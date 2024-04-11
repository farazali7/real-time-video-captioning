# Results Folder

Stores the results offline in the same resporitory against a `wandb_run_name` in the `results/run/` directory. You can open your experiment name and view your student and teacher model configurations, as well as training step metrics and validation results that are generated at the end of each epoch.


## Table of Contents

- [Results Folder](#results-folder)
    - [Repository Structure](#repository-structure)
    - [Runs Folder](#runs-folder)
    - [GIT Model Weights (model.pt)](#git-model-weights-modelpt)


## Repository Structure

```
results
├── run
|     ├── run_name_1
|     │     └── results_and_metrics.txt
|     |     └── desc.txt <-- You need to manually write this if needed (optional).
|     |     └── *.ckpt (*Distillation model checkpoints* saved after each epoch).
|     |     └── test__results_and_metrics.txt
|     |     └── validation__and_metrics.txt
|     └── run_name_2
|           └── ...
├── model.pt <-- To be installed by you, this is the teacher model weights.
└── .gitkeep
```

## Runs Folder

Contains each run information during a training run. Creates a unique `wandb_run_name` for each run and stores all important information about the run, configuration, teacher and student model information, as well as training, testing and validation metrics. It also stored the final epoch model checkpoint.


## GIT Model weights (model.pt)

This will be used to obtain teacher model results and activations. Weights will be freezed during training and inference for the teacher model. If this isn't installed already, follow the steps below.

Click this [link](https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_MSRVTT/snapshot/model.pt) to install the model weights. Move the weights to the `results` directory as shown in the repository structure. This is needed to get teacher ouputs after freezing weights to use multiple sets of activations and output logits for knowledge distillation.