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


from inference import extract_features, batch_prob_infer_fn


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
    for i in range(len(data)):
        input_ids = torch.tensor([data[i]["input_ids"]])
        attention_mask = torch.tensor([data[i]["attention_mask"]])
        prompt_length = torch.tensor([data[i]["prompt_length"]])
        prob = batch_prob_infer_fn(model, input_ids, attention_mask, prompt_length).detach().cpu().type(torch.half)
        scores.append(prob)
    return np.array(scores)


@torch.no_grad()
def dpp_extract_features(model, data):
    embeds = []
    for i in range(len(data)):
        datapoint = {}
        datapoint["input_ids"] = torch.tensor([data[i]["input_ids"]])
        datapoint["attention_mask"] = torch.tensor([data[i]["attention_mask"]])
        embeds.append(extract_features(model, datapoint))
    embeds = np.concatenate(embeds, axis=0)
    mean = embeds.mean(axis=0)
    std = embeds.std(axis=0)
    embeds = (embeds - mean) / std
    return embeds


@torch.no_grad()
def dpp(kernel_matrix, max_length, epsilon=1E-10):
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


@torch.no_grad()
def select_top10(model, data, timestep):
    random.seed(timestep)
    random_indices = random.sample(range(len(data)), 100)
    random_selected_data = [data[i] for i in random_indices]

    t = time.time()
    scores = dpp_calculate_scores(model, random_selected_data)
    print(f"Scores are calculated successfully in {time.time() - t}!")

    t = time.time()
    feature_vectors = dpp_extract_features(model, random_selected_data)
    print(f"Features are extracted successfully in {time.time() - t}!")

    t = time.time()
    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    similarities = np.dot(feature_vectors, feature_vectors.T)
    kernal_matrix = scores.reshape((100, 1)) * similarities * scores.reshape((1, 100))
    print(f"Kernel matrix is generated successfully in {time.time() - t}!")

    selected_indices = dpp(kernal_matrix, 10)
    selected_data = [random_selected_data[int(i)] for i in selected_indices]
    print(f"Generate {len(selected_data)} samples successfully!")
    return selected_data


def train(model, batch_size, data, tokenizer, output_dir, val_set_size):
    train_data = data['train'].map(generate_and_tokenize_prompt)
    model.train()

    for timestep in tqdm(range(400), desc="Training Progress", unit="step"):
        timestep += 101
        train_step_data = select_top10(model, train_data, timestep)
        trainer = Trainer(
            model=model,
            train_dataset=train_step_data,
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

        if timestep % 100 == 0:
            model.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch_{timestep // 100}_model"))

    return 0


@torch.no_grad()
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


@torch.no_grad()
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
    OUTPUT_DIR = "/home/hywang/projects/d2pruning/DPP/model3"
    adapter_path = "/home/hywang/projects/d2pruning/DPP/model3/epoch_1_model"  # manually modify if resume

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
