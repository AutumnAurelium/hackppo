import torch

def reward_fn(recipe: "recipes.hackppo.rlvr_full_finetune.PPORLVRFullFinetuneRecipeSingleDevice", token_ids: torch.Tensor, input_pos_ids: torch.Tensor, mask: torch.Tensor):
    text = recipe._tokenizer.decode(token_ids)
    print(text)
    return sum([1 if x == 'a' else 0 for x in text])

def reward_fn_example():
    return reward_fn
