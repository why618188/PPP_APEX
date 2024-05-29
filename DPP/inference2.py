import gc
import torch
import numpy as np
from typing import List


@torch.no_grad()
def build_chat_input_llama(model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    def _parse_messages(messages, split_role="user"):
        system, text = "", ""
        for i, message in enumerate(messages):
            if message["role"] == "system":
                system = message["content"]
            if message["role"] == split_role:
                text = message["content"]
        return system, text

    max_tokens = max_new_tokens or model.generation_config.max_length
    system, text = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_text_tokens = max_tokens - len(system_tokens)
    text_tokens = tokenizer.encode(text)

    input_tokens = system_tokens + text_tokens
    input_tokens = input_tokens[-max_tokens:]  # truncate left

    return input_tokens


@torch.no_grad()
def batch_prob_infer_fn(
    model,
    batch_input_ids, 
    batch_attention_mask, 
    prompt_lengths,  
    context_len=None,
):

    if context_len and batch_input_ids.shape[1] > context_len:
        batch_input_ids = batch_input_ids[:, -context_len:]
        batch_attention_mask = batch_attention_mask[:, -context_len:]
        prompt_lengths = [max(0, pl - (batch_input_ids.shape[1] - context_len)) for pl in prompt_lengths]

    model.eval()

    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
    batch_probs = torch.ones(batch_input_ids.size(0), device=batch_input_ids.device)
    for i in range(batch_input_ids.size(0)):
        prompt_length = prompt_lengths[i]
        output_probs = torch.softmax(outputs.logits[i, prompt_length-1:-1, :], dim=-1)  
        token_indices = batch_input_ids[i, prompt_length:].unsqueeze(-1) 
        if token_indices.max() >= output_probs.shape[-1]:
            print(f"Index out of bounds: {token_indices.max()} not in [0, {output_probs.shape[-1]-1}]")
            continue  
        token_probs = torch.gather(output_probs, 1, token_indices).squeeze(-1) 

        valid_mask = batch_attention_mask[i, prompt_length:]  
        token_probs[~valid_mask.bool()] = 1
        action_prob = token_probs.prod()
 
        num_valid_tokens = valid_mask.sum()
        if num_valid_tokens > 0:
            action_prob **= (1.0 / num_valid_tokens)
        batch_probs[i] = action_prob

    return batch_probs


@torch.no_grad()
def inference_output_prob(
    model, tokenizer, messages, actions, device, context_len, max_batch_size=1
):
    num_action = len(actions)
    assert not model.config.is_encoder_decoder
    
    prompt_ids = build_chat_input_llama(model, tokenizer, messages)
    tokenized_actions = tokenizer(actions)
    input_ids, attention_mask = [], []
    
    max_action_length = sum(len(ids) for ids in tokenized_actions.input_ids)
    max_length = len(prompt_ids) + max_action_length  
    tokenizer.add_special_tokens({'pad_token': 'null'})

    prompt_len = []
    for i in range(num_action):
        past_input_ids = prompt_ids + [i for list in tokenized_actions.input_ids[:i] for i in list]
        current_input_ids = {'input_ids': past_input_ids + tokenized_actions.input_ids[i]}
        current_input_ids = tokenizer.pad(current_input_ids, max_length=max_length, padding="max_length")
        prompt_len.append(len(past_input_ids))
        input_ids.append(current_input_ids['input_ids'])
        attention_mask.append(current_input_ids['attention_mask'])

    input_ids = torch.as_tensor(input_ids, device=device)
    attention_mask = torch.as_tensor(attention_mask, device=device)

    assert not torch.any(input_ids[:, 0] == tokenizer.pad_token_id)

    rets = []
    with torch.no_grad():
        for i_l in range(0, num_action, max_batch_size):
            i_r = min(i_l + max_batch_size, num_action)
            batch_action_probs = batch_prob_infer_fn(
                model, 
                input_ids[i_l:i_r],
                attention_mask[i_l: i_r],
                prompt_len[i_l:i_r],
                context_len
            )
            rets.extend(batch_action_probs.cpu().tolist())
    return rets
