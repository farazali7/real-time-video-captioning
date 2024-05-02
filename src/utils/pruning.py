import pickle
import torch
from torch.nn import Module
from torch.nn.utils import prune
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import re

from src.utils.io import load_kd_student_model, load_pruned_model

from config import cfg


def global_prune_model(model: Module, ratio: float) -> Module:
    """
    Global unstructured pruning of given model based on specified lowest L1 norm and ratio of parameters to remove.

    Args:
        model: Original model
        ratio: Percentage of parameters to prune from entire network if global strategy, or per layer
        if local strategy

    Returns:
        Pruned model instance
    """
    # Build tuple of parameters and weight name
    # Need to evaluate model modules so convert strings of the form:
    # chars.#.chars -> chars[#].chars
    pattern = r'(.+)\.(\d)\.(.+)'
    params = [p for p in model.named_parameters() if 'weight' in p[0]]
    for idx, (name, p) in enumerate(params):
        # First fix the naming of the weight variables ('weight' for all modules except MHA)
        if 'in_proj_weight' in name:  # For Multi-Head Attention layers
            param_name = 'in_proj_weight'
        else:
            param_name = 'weight'
        name = f'model.{name.replace(f".{param_name}", "")}'

        # Now replace chars.#.chars -> chars[#].chars for later eval() call to work
        match = re.match(pattern, name)
        if match:
            prefix, num, suffix = match.groups()
            name = f'{prefix}[{num}].{suffix}'

        # Evaluate name as actual nn.Module
        params[idx] = (eval(name), param_name)

    prune.global_unstructured(parameters=params, pruning_method=prune.L1Unstructured, amount=ratio)

    # Make it permanent (i.e., shift pruned named parameter to original parameter naming and drop mask)
    for module, param_name in params:
        prune.remove(module=module, name=param_name)

    return model


if __name__ == "__main__":
    # Load configuration
    train_args = cfg['TRAIN']

    # Dataset preparation
    data_path = cfg['DATA']['VIDEOS_PATH']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    random_state = cfg['SEED']

    vocab_length = len(tokenizer.vocab)

    student_model_def = train_args['STUDENT_MODEL_DEF']

    student_model_args = {
        **cfg['MODEL'][student_model_def],
        'vocab_length': vocab_length,
        'cls_token_id': tokenizer.cls_token_id,
        'sep_token_id': tokenizer.sep_token_id
    }

    ckpt_path = 'results/2l_ce_kl_epoch19_100424.ckpt'

    # # Create a series of pruned models, ranging from 10-50% pruning
    # ratios = np.arange(0.1, 0.6, 0.1)
    # for ratio in ratios:
    #     # Load a fresh pretrained student model
    #     pretrained_student = load_kd_student_model(ckpt_path, student_model_args)
    #
    #     # Prune the model
    #     pruned = global_prune_model(pretrained_student, ratio=ratio)
    #
    #     # Save the model for future experimentation
    #     model_path = f'results/pruned/2l_ce_kl_e19_{round(ratio, 1)}pr.pth'
    #     torch.save(pruned.state_dict(), model_path)

    m = load_pruned_model('results/pruned/2l_ce_kl_e19_0.1pr.pth', student_model_args)

    print('Done')
