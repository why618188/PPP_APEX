import os
import re
import math
import time
import torch
import random
import numpy as np

from functools import partial
from tqdm import tqdm
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    load_peft_weights,
    set_peft_model_state_dict
)


from inference import extract_features
from inference2 import batch_prob_infer_fn, inference_output_prob


def find_all_linear_names(model):
    # cls = bnb.nn.Linear8bitLt
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def tokenize(prompt, type):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < CUTOFF_LEN
            and type == "target"
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if len(result["input_ids"]) >= CUTOFF_LEN and type == "target":
        result["input_ids"][CUTOFF_LEN - 1] = tokenizer.eos_token_id
        result["attention_mask"][CUTOFF_LEN - 1] = 1

    result["prompt_lengths"] = len(result["input_ids"])

    return result


def concatenate(tokenized_text1, tokenized_text2):
    input_ids = tokenized_text1["input_ids"] + tokenized_text2["input_ids"]
    attention_mask = tokenized_text1["attention_mask"] + tokenized_text2["attention_mask"]
    prompt_length = tokenized_text1["prompt_lengths"]

    concatenated_text = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_length": prompt_length,
        "labels": input_ids.copy()
    }

    return concatenated_text


def generate_and_tokenize_prompt(data_point):
    data_point = data_point["conversation"][0]
    instruction = data_point['system']
    input_text = data_point["input"]
    input_text = instruction + input_text
    input_text = tokenizer.bos_token + input_text
    target_text = data_point["output"] + tokenizer.eos_token

    tokenized_input_text = tokenize(input_text, type="input")
    tokenized_target_txt = tokenize(target_text, type="target")

    return concatenate(tokenized_input_text, tokenized_target_txt)


@torch.no_grad()
def dpp_calculate_scores(model, data):
    scores = []
    for i in tqdm(range(len(data)), desc="Calculating Scores"):
        input_ids = torch.tensor([data[i]["input_ids"]])
        attention_mask = torch.tensor([data[i]["attention_mask"]])
        prompt_length = torch.tensor([data[i]["prompt_length"]])
        score = batch_prob_infer_fn(model, input_ids, attention_mask, prompt_length).detach().cpu().type(torch.half)
        scores.append(score)
    return np.array(scores)


@torch.no_grad()
def dpp_extract_features(model, data):
    embeds = []
    for i in tqdm(range(len(data)), desc="Extracting Features"):
        datapoint = {}
        datapoint["input_ids"] = torch.tensor([data[i]["input_ids"]])
        datapoint["attention_mask"] = torch.tensor([data[i]["attention_mask"]])
        embeds.append(extract_features(model, datapoint))
    embeds = np.concatenate(embeds, axis=0)
    return embeds


@torch.no_grad()
def dpp(kernel_matrix, max_length, epsilon=1E-20):
    """
    Fast greedy implementation of DPP algorithm.
    From: https://github.com/laming-chen/fast-map-dpp.
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def generate_kernel_matrix(feature_vectors, scores, is_score=True):
    mean = feature_vectors.mean(axis=0)
    std = feature_vectors.std(axis=0)
    feature_vectors = (feature_vectors - mean) / std
    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    if is_score:
        similarities = np.dot(feature_vectors, feature_vectors.T)
        kernel_matrix = scores.reshape((-1, 1)) * similarities * scores.reshape((1, -1))
    else:
        kernel_matrix = np.dot(feature_vectors, feature_vectors.T)
    return kernel_matrix


@torch.no_grad()
def select_top1000(model, data, epoch):
    scores = dpp_calculate_scores(model, data)
    percentile = 100 - 20 * epoch
    threshold = np.percentile(scores, percentile)
    mask = scores >= threshold
    indices = np.where(mask)[0]
    eligible_data = [data[int(i)] for i in indices]
    feature_vectors = dpp_extract_features(model, eligible_data)

    selected_data = []
    while True:
        kernel = generate_kernel_matrix(feature_vectors, scores=None, is_score=False)
        new_selected_indices = dpp(kernel, 1000 - len(selected_data))
        new_selected_data = [eligible_data[i] for i in new_selected_indices]
        selected_data += new_selected_data
        if len(selected_data) >= 1000:
            break
        all_indices = np.arange(feature_vectors.shape[0])
        remaining_indices = np.setdiff1d(all_indices, new_selected_indices)
        feature_vectors = feature_vectors[remaining_indices]
        eligible_data = [eligible_data[i] for i in remaining_indices]

    print(f"Generate {len(selected_data)} samples successfully!")
    return selected_data


def train(model, batch_size, data, tokenizer, output_dir, val_set_size):
    train_data = data['train'].map(generate_and_tokenize_prompt)
    model.train()

    for epoch in range(1):
        epoch += 5
        selected_data = select_top1000(model, train_data, epoch)
        trainer = Trainer(
            model=model,
            train_dataset=selected_data,
            args=TrainingArguments(
                num_train_epochs=1,  # Set to 1 since we're manually looping over epochs
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=4e-4,
                weight_decay=1e-4,
                gradient_accumulation_steps=1,
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=2000 if val_set_size > 0 else None,
                save_steps=2000,
                output_dir=output_dir,
                report_to="tensorboard",
                logging_steps=10,
                save_total_limit=3,
                load_best_model_at_end=True if val_set_size > 0 else False,
                optim="adamw_torch"
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                              padding=True)
        )
        trainer.train()
        model.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch_{epoch}_model"))

    return 0


def validate(model, data_test_raw, epoch, output_dir):
    sr = 0
    data_test_raw = data_test_raw['train']
    with open(os.path.join(output_dir, f"epoch_{epoch}_response-label.txt"), "w") as f:
        with tqdm(total=len(data_test_raw), desc="Validating", unit="data") as pbar:
            f.write("Success Rate:\n")
            for j in range(len(data_test_raw)):
                data_test = data_test_raw[j]['conversation'][0]
                System = data_test['system']
                messages = [{"role": "system", "content": System}]
                Input = data_test['input']
                label = data_test['output']
                messages.append({"role": "user", "content": Input})
                response = model.chat(tokenizer, messages)
                if j % 50 == 0:
                    f.write(f"response of example {j // 50}:\n" + response + "\n")
                    f.write(f"label of example {j // 50}:\n" + label + "\n")
                if response == label:
                    sr += 1
                pbar.update(1)
            success_rate = sr / len(data_test_raw)
            f.seek(0)
            f.write(f"Success Rate: {success_rate}\n")
    return success_rate


def test(start_epoch, model, test_data, output_dir):
    model.eval()
    epoch = start_epoch
    print(f"Test of epoch {epoch}!")
    sr = validate(model, test_data, epoch, output_dir)
    print(f"Success rate of epoch {epoch} is: {sr}.")


if __name__ == '__main__':

    # hyperparameters
    CUTOFF_LEN = 2048
    VAL_SET_SIZE = 0
    BATCH_SIZE = 5
    NUM_EPOCHS = 1
    START_EPOCH = 5    # manually modify if resume

    # paths
    DATA_PATH = "/home/hywang/projects/d2pruning/train.json"
    TEST_PATH = "/home/hywang/projects/d2pruning/test.json"
    model_path = "/home/hywang/projects/d2pruning/Baichuan2-7B-Chat"
    OUTPUT_DIR = "/home/hywang/projects/d2pruning/DPP/model6"
    adapter_path = "/home/hywang/projects/d2pruning/DPP/model6/epoch_4_model"  # manually modify if resume

    # load peft model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='right')
    tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 quantization_config=BitsAndBytesConfig(
                                                     load_in_4bit=True,
                                                     bnb_4bit_compute_dtype=torch.float16,
                                                     bnb_4bit_use_double_quant=True,
                                                     bnb_4bit_quant_type='nf4'
                                                 ),
                                                 use_cache=False,
                                                 device_map="auto")
    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)
    config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=modules,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    adapter_weights = load_peft_weights(adapter_path)    # if resume
    set_peft_model_state_dict(model, adapter_weights)    # if resume

    # datasets
    data = load_dataset("json", data_files=DATA_PATH)
    test_data = load_dataset("json", data_files=TEST_PATH)

    """Remember to choose only one mode!"""
    # train mode
    train(model, BATCH_SIZE, data, tokenizer, OUTPUT_DIR, VAL_SET_SIZE)

    # log mode
    # log_td(START_EPOCH, model, data, OUTPUT_DIR)

    # evaluate mode
    # eval(START_EPOCH, model, test_data, OUTPUT_DIR)

    # extract features mode
    # save_features(model, data)    # Remember to set START_EPOCH as last epoch!
