import os
import random
import torch
import numpy as np

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
        "prompt_lengths": prompt_length,
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


def select_top10(model, data, irrloss, timestep):
    random.seed(timestep)
    random_indices = random.sample(range(len(data)), 100)
    selected_data = [data[i] for i in random_indices]
    selected_irrloss = np.array([irrloss[i] for i in random_indices])
    selected_trainloss = []

    with torch.no_grad():
        for i in range(100):
            datapoint = {}
            datapoint["input_ids"] = torch.tensor(selected_data[i]["input_ids"]).reshape(1, -1)
            datapoint["attention_mask"] = torch.tensor(selected_data[i]["attention_mask"]).reshape(1, -1)
            labels = torch.tensor(selected_data[i]["labels"]).reshape(1, -1)
            output = model(**datapoint, labels=labels)
            selected_trainloss.append(output["loss"].item())

    selected_trainloss = np.array(selected_trainloss)
    rho_loss = selected_trainloss - selected_irrloss

    sorted_indices = np.argsort(rho_loss)[::-1]
    top10_indices = sorted_indices[:10]
    top10_data = [selected_data[i] for i in top10_indices]

    for i in top10_indices:
        print(rho_loss[i], selected_trainloss[i], selected_irrloss[i])
    top10_origin_indices = [random_indices[i] for i in top10_indices]
    print(top10_origin_indices)

    return top10_data


def train(model, batch_size, data, tokenizer, output_dir, val_set_size):
    train_data = data['train'].map(generate_and_tokenize_prompt)
    irrloss = np.load("PTP/irrloss.npy")
    model.train()

    for timestep in tqdm(range(500), desc="Training Progress", unit="step"):
        timestep += 1
        train_step_data = select_top10(model, train_data, irrloss, timestep)
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


if __name__ == '__main__':
    # hyperparameters
    CUTOFF_LEN = 2048
    VAL_SET_SIZE = 0
    BATCH_SIZE = 5
    NUM_EPOCHS = 1
    START_EPOCH = 1    # manually modify if resume

    # paths
    DATA_PATH = "/home/hywang/projects/d2pruning/train.json"
    OUTPUT_DIR = "/home/hywang/projects/d2pruning/PTP/Target_Model"
    model_path = "/D2P/Baichuan2-7B-Chat"
    # adapter_path = "/home/hywang/projects/d2pruning/PTP/Target_Model/epoch_1_model"    # manually modify if resume

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

    # adapter_weights = load_peft_weights(adapter_path)    # if resume
    #set_peft_model_state_dict(model, adapter_weights)    # if resume

    # datasets
    data = load_dataset("json", data_files=DATA_PATH)

    train(model, BATCH_SIZE, data, tokenizer, OUTPUT_DIR, VAL_SET_SIZE)
