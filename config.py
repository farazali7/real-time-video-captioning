'''
CONFIGURATION SETTINGS FOR PROJECT
'''
import torch

cfg = {
    'DATA': {
    },
    'CALLBACK': {
    },
    'LOGGER': {
        'save_dir': 'results/',
        'name': 'imglogs'
    },
    'TRAIN': {
        'model_def': 'ModelV1',
        'TRAINER': {
            'max_epochs': 100,
            'precision': 16,
            'enable_checkpointing': True,
            'accelerator': 'gpu',
            'devices': int(torch.cuda.device_count()),
            'strategy': 'ddp_find_unused_parameters_true'
        },
        'LR': 1e-5,
        'BATCH_SIZE': 16,
        'DATA_PERC': 1.0
    },
    'MODEL': {
        'ModelV1': {
            'TinyVIT': {
                'embed_dim': 2048,
                'hidden_dim': 512,
                'n_heads': 4,
                'n_layers': 4,
                'dropout': 0.3,
                'attention_type': 'fast',
                'n_features': 256
            },
            'DECODER': {
                'outchn': 1,
                'in_ft': 1,
                'img_size': 256,
                'apply_attention': True,
                'embed_dim': 256,
                'n_heads': 4,
            }
        },
    },
    'WANDB': {
        "MODE": 'online'
    }
}