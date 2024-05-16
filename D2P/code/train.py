import os
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

from core.training import TrainingDynamicsLogger
from inference import batch_prob_infer_fn, extract_features


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


def generate_and_tokenize_prompt_only(data_point):
    data_point = data_point["conversation"][0]
    instruction = data_point['system']
    input_text = data_point["input"]
    input_text = instruction + input_text
    input_text = tokenizer.bos_token + input_text

    return tokenizer(input_text, truncation=True, max_length=CUTOFF_LEN, padding=False, return_tensors='pt')


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


def train(num_epochs, start_epoch, model, batch_size, data, tokenizer, output_dir, val_set_size):
    train_data = data['train'].map(generate_and_tokenize_prompt)
    model.train()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        trainer = Trainer(
            model=model,
            train_dataset=train_data,
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
        print(f"Training of epoch {epoch}!")
        trainer.train()
        model.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch_{epoch}_model"))

    return 0


def log_td(start_epoch, model, data, output_dir):
    train_data = data['train'].map(generate_and_tokenize_prompt)
    model.eval()
    epoch = start_epoch
    td_logger = TrainingDynamicsLogger()
    print(f"Logging of epoch {epoch}!")

    for step in tqdm(range(len(train_data["input_ids"])), desc="Logging", unit="step"):
        idx = torch.tensor([step])
        input_ids = torch.tensor([train_data["input_ids"][step]])
        attention_mask = torch.tensor([train_data["attention_mask"][step]])
        prompt_lengths = torch.tensor([train_data["prompt_lengths"][step]])
        log_tuple = {
            'epoch': epoch,
            'iteration': None,
            'idx': idx,
            'output': batch_prob_infer_fn(model, input_ids, attention_mask, prompt_lengths).detach().cpu().type(torch.half)
        }
        td_logger.log_tuple(log_tuple)

    td_logger.save_training_dynamics(os.path.join(output_dir, f"td_{epoch}.pkl"))
    print(f"Logging completed for epoch {epoch}.")


def eval(start_epoch, model, test_data, output_dir):
    model.eval()
    epoch = start_epoch
    print(f"Test of epoch {epoch}!")
    sr = validate(model, test_data, epoch, output_dir)
    print(f"Success rate of epoch {epoch} is: {sr}.")


def save_features(model, data):
    model.eval()
    train_data = data['train'].map(generate_and_tokenize_prompt_only, remove_columns=['conversation'])
    embeds = []
    for step in tqdm(range(len(train_data["input_ids"])), desc="Extract Features", unit="step"):
        datapoint = {}
        datapoint["input_ids"] = torch.tensor(train_data[step]["input_ids"])
        datapoint["attention_mask"] = torch.tensor(train_data[step]["attention_mask"])
        embeds.append(extract_features(model, datapoint))

    embeds = np.concatenate(embeds, axis=0)
    mean_values = np.mean(embeds, axis=0)
    std_devs = np.std(embeds, axis=0)
    embeds = (embeds - mean_values) / std_devs
    np.save(os.path.join("/D2P", "train-features.npy"), embeds)


if __name__ == '__main__':

    # hyperparameters
    CUTOFF_LEN = 2048
    VAL_SET_SIZE = 0
    BATCH_SIZE = 7
    NUM_EPOCHS = 1
    START_EPOCH = 5    # manually modify if resume

    # paths
    DATA_PATH = "/home/hywang/projects/d2pruning/train.json"
    TEST_PATH = "/home/hywang/projects/d2pruning/test.json"
    model_path = '/D2P/Warm_up_model'
    OUTPUT_DIR = "/D2P/D2P_Round1_Model"
    adapter_path = "/D2P/D2P_Round1_Model/epoch_3_model"  # manually modify if resume

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
    train(NUM_EPOCHS, START_EPOCH, model, BATCH_SIZE, data, tokenizer, OUTPUT_DIR, VAL_SET_SIZE)

    # log mode
    # log_td(START_EPOCH, model, data, OUTPUT_DIR)

    # evaluate mode
    # eval(START_EPOCH, model, test_data, OUTPUT_DIR)

    # extract features mode
    # save_features(model, data)    # Remember to set START_EPOCH as last epoch!
