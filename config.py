'''
CONFIGURATION SETTINGS FOR PROJECT
'''
import torch

cfg = {
    'SEED': 5,
    'DATA': {
        'VIDEOS_PATH': 'data/MSRVTT/videos/all',
        'CAPTIONS_PATH': 'data/labels/labels.csv',
        'ENCODED_CAPTION_IDS': 'data/labels/encoded_captions.pkl'
    },
    'CALLBACK': {
        'dirpath': 'results/',
        'save_top_k': 2,
        'monitor': 'val_loss'
    },
    'LOGGER': {
        'save_dir': 'results/',
        'name': 'captions'
    },
    'TRAIN': {
        'STUDENT_MODEL_DEF': 'StudentCandidateV1',
        'TEACHER_MODEL_DEF': 'GenerativeImageTextTeacher',
        'TRAINER': {
            'max_epochs': 100,
            'precision': 16,
            'enable_checkpointing': True,
            'strategy': 'auto'
        },
        'LR': 1e-5,
        'BATCH_SIZE': 2,
    },
    'MODEL': {
        # STUDENT MODELS
        'StudentCandidateV1': {
            'image_enc_name': 'tiny_vit_21m_224.dist_in22k_ft_in1k',
            'd_model': 576,
            'n_head': 8,
            'd_ffn': 1024,
            'dropout': 0.2,
            'num_decoder_layers': 4
        },
        # TEACHER MODELS
        'GenerativeImageTextTeacher': {
            'param_path': 'data/teacher_configs/GIT_LARGE_MSRVTT/parameter.yaml',
            'pretrained_weights': 'results/model.pt'
        }
    },
    'WANDB': {
        "MODE": 'disabled'  # One of {'online', 'offline', 'disabled'}
    }
}