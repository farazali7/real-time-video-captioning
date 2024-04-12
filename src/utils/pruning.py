import pickle
import torch
from torch.nn import Module
from torch.nn.utils import prune
import pandas as pd
from transformers import BertTokenizer

from src.utils.io import load_kd_student_model

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
    prune.global_unstructured(parameters=model, pruning_method=prune.l1_unstructured, amount=ratio)

    # Make it permanent (i.e., shift pruned named parameter to original parameter naming and drop mask)
    prune.remove(module=model, name='')

    return model


if __name__ == "__main__":
    # Load configuration
    train_args = cfg['TRAIN']

    # Dataset preparation
    data_path = cfg['DATA']['VIDEOS_PATH']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    random_state = cfg['SEED']

    vocab_length = len(tokenizer.vocab)

    student_model_def, teacher_model_def = train_args['STUDENT_MODEL_DEF'], train_args['TEACHER_MODEL_DEF']

    student_model_args = {
        **cfg['MODEL'][student_model_def],
        'vocab_length': vocab_length,
        'cls_token_id': tokenizer.cls_token_id,
        'sep_token_id': tokenizer.sep_token_id
    }

    teacher_model_args = cfg['MODEL'][teacher_model_def]

    ckpt_path = 'results/2l_ce_kl_epoch19_100424.ckpt'
    student_model = load_kd_student_model(ckpt_path, student_model_args)

    print('Done')
