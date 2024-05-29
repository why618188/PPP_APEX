import torch
import numpy as np
from typing import List


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
        output_probs = torch.softmax(outputs.logits[i, prompt_length - 1:-1, :], dim=-1)
        token_indices = batch_input_ids[i, prompt_length:].unsqueeze(-1)
        if token_indices.max() >= output_probs.shape[-1]:
            print(f"Index out of bounds: {token_indices.max()} not in [0, {output_probs.shape[-1] - 1}]")
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
def extract_features(model, input):
    output = model(**input, output_hidden_states=True)
    temp = output["hidden_states"][-1]
    features = temp[:, 0, :].detach().cpu().numpy()
    return features
