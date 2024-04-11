# Real-Time Video Captioning

This repository contains the source code of "Real-Time Video Captioning Systems", which is our project for the course _CSC2231HS — Topics in Computer Systems: Visual and Mobile Computing Systems_ course at the _University of Toronto_. This repository takes inspiration from and also utilizes certain parts from the [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) research by Microsoft.

Contributors: [Faraz Ali](https://github.com/farazali7), [Paanugoban Kugathasan](https://github.com/Paanugoban) and [Gautam Chettiar](https://github.com/chettiargautam)


## Table of Contents

- [Real-Time Video Captioning](#real-time-video-captioning)
  - [Setup](#setup)
    - [Clone the Repository](#clone-the-repository)
  - [Virtual Environment Setup](#virtual-environment-setup-recommended-but-optional)
    - [Using `venv` from `pip`](#using-venv-from-pip)
      - [Windows](#windows)
      - [macOS/Linux](#macoslinux)
    - [Using `conda`](#using-conda)
      - [Windows, macOS, and Linux](#windows-macos-and-linux)
    - [Notes](#notes)
  - [Install the Required Packages](#install-the-required-packages)
    - [Using `pip`](#using-pip-for-venv-environments)
    - [Using `conda`](#using-conda-for-anaconda-miniconda-environments)
    - [Note](#note)
  - [GenerativeImage2Text](#generativeimage2text)
  - [Repository Structure](#repository-structure)
  - [Installing the MSRVTT dataset](#installing-the-msrvtt-dataset)
  - [Installing the GIT Model Weights (model.pt)](#installing-the-git-model-weights-modelpt)
  - [Installing PyTorch with CUDA Support](#installing-pytorch-with-cuda-support)
    - [Testing PyTorch CUDA Installation](#testing-pytorch-cuda-installation)
  - [Setting Up Weights & Biases (wandb)](#setting-up-weights--biases-wandb)
    - [Installation](#installation)
    - [Configuration](#configuration)
    - [Usage](#usage)
    - [Note](#note-1)
  - [`config.py` File Description](#configpy-file-description)
  - [Training](#training)
  - [Inference](#inference)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)


## Setup

### Clone the repository

Clone the repository into your machine using the following commands:   
```shell
git clone https://github.com/farazali7/real-time-video-captioning.git
cd real-time-video-captioning
```


## Virtual Environment Setup (recommended but optional)

Setting up a virtual environment is crucial for managing project-specific dependencies and avoiding conflicts between different projects. Below are the steps to set up a virtual environment using `venv` (part of the Python Standard Library) and `conda` (part of the Anaconda distribution).

### Using `venv` from `pip`

#### Windows

1. Open your command prompt (cmd) or PowerShell.
2. Navigate to your project's directory.
3. Create a virtual environment named `env`:

   ```cmd
   python3 -m venv env
   ```

4. Activate the virtual environment:

   - **Command Prompt:**

     ```cmd
     .\env\Scripts\activate
     ```

   - **PowerShell:**

     ```powershell
     .\env\Scripts\Activate.ps1
     ```

#### macOS/Linux

1. Open your terminal.
2. Navigate to your project's directory.
3. Create a virtual environment named `env`:

   ```bash
   python3 -m venv env
   ```

4. Activate the virtual environment:

   ```bash
   source env/bin/activate
   ```

### Using `conda`

#### Windows, macOS, and Linux

1. Open your Anaconda Prompt (Windows) or terminal (macOS/Linux).
2. Create a new conda environment named `myenv`:

   ```bash
   conda create --name myenv python=3.9
   ```

   - Replace `myenv` with your desired environment name and `3.9` with the required Python version.

3. Activate the newly created conda environment:

   ```bash
   conda activate myenv
   ```

### Notes

- After activating your virtual environment (`venv` or `conda`), you can install packages using `pip install <package_name>` for `venv` environments or `conda install <package_name>` for conda environments.
- To deactivate any virtual environment and return to the global Python environment, run `deactivate` for `venv` environments or `conda deactivate` for conda environments.
- Ensure your Python and Anaconda installations are up to date to avoid potential issues with creating or managing virtual environments.


## Install the Required Packages

After setting up and activating your virtual environment, you can install the required packages. Use the appropriate set of commands based on your virtual environment setup (`venv` with `pip` or `conda` for Anaconda/Miniconda environments).

### Using `pip` (for `venv` environments)

1. Install the packages from the `requirements.txt` file:

   ```shell
   pip install -r requirements.txt
   ```

2. Run the setup script (optional, avoid if breaks):

   ```shell
   python3 setup.py build develop
   ```

### Using `conda` (for Anaconda/Miniconda environments)

If you're using a `conda` environment and prefer to manage packages with `conda`, you may need to find equivalent conda packages for the contents of your `requirements.txt`. However, for packages or dependencies not available through conda channels, you can still use `pip` within your conda environment.

1. For packages available via conda, install them using:

   ```shell
   conda install package_name
   ```

   Replace `package_name` with the actual names of the packages you need.

2. For any remaining packages listed in `requirements.txt` that are not available through conda, or for which you prefer the pip version, use:

   ```shell
   pip install package_name
   ```

3. Run the setup script (optional, avoid if breaks):

   ```shell
   python3 setup.py build develop
   ```

### Note

- When using `conda`, it's a good practice to first search for the package using `conda search package_name` to see if it's available through conda channels. If not, then resort to `pip`.
- Always ensure your environment is activated (`source env/bin/activate` for `venv` or `conda activate env_name` for conda) before running these commands.
- To deactivate the `conda` environment and return to the global Python environment, use:

   ```bash
   conda deactivate
   ```


## GenerativeImage2Text

This [repository](https://github.com/microsoft/GenerativeImage2Text) contains the source code of the research mentioned earlier. The GIT model is used as the teacher model for knowledge distillation tasks and hence is required for running experiments.

While the dependency is provided in the `requirements.txt` file, if you are using a virtual environment and will install dependencies using either `pip` or `conda`, it is most likely going to install them as modules to `../envs/venv/site-packages.` and hence will require the code to be run as a module. If this for some reason doesn't work as desired, then follow the steps below.

1. Ensure you are in the root directory:

   ```bash
   pwd
   ```

   Which should return a path similar to `../real-time-video-captioning/`

2. Clone the GIT repository directly:

   ```bash
   git clone https://github.com/microsoft/GenerativeImage2Text.git
   ```

3. Within the folder that gets installed, you can move the `generativeimage2text` subfolder to the root directory and delete the rest of the repository. Do remember to also install any dependencies that their model uses either from their source `requirements.txt` or by manually installing them.


## Repository Structure

To keep things clear, an expected repository structure has been provided below:

```
.
├── .idea
├── .gitignore
├── config.py (important, consists of all teacher, student models and run configs)
├── data
│   └── labels
│   └── MSRVTT (needs to be installed)
│   └── teacher_configs
├── lightning_logs
│   └── ...
├── MANIFEST.in
├── output (don't need to create this explicitly)
│   └── clip
│        └── ViT-L-14.pt (installs after first run)
├── __pycache__
│   └── ...
├── README.md (current)
├── requirements.txt
├── results
│   └── run (creates after first run)
│        └── ...
│   └── model.pt (Needs to be installed)
├── setup.py
├── generativeimage2text (optional to manually clone if installing dependencies fails)
│   └── ...
├── src
│   └── ...
├── tests
│   └── ...
└── wandb
    └── ...
```


## Installing the MSRVTT dataset.

Install the `MSRVTT` dataset from this [link](https://cove.thecvf.com/datasets/839). Unzip the contents and move the `MSRVTT` folder to the `data` directory as shown in the repository structure. Its contents should look like this:

```
data
└── MSRVTT
      ├── annotation
      │   └── ...
      ├── high-quality
      │   └── ...
      ├── structured-symlinks (generated on run)
      │   └── ...
      ├── videos
      │   └── ...
```


## Installing the GIT model weights (model.pt)

Click this [link](https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_MSRVTT/snapshot/model.pt) to install the model weights. Move the weights to the `results` directory as shown in the repository structure. This is needed to get teacher ouputs after freezing weights to use multiple sets of activations and output logits for knowledge distillation.


## Installing PyTorch with CUDA Support

To leverage GPU capabilities for deep learning tasks, ensure you have a CUDA-compatible GPU and the corresponding CUDA Toolkit installed. Follow the steps below to install PyTorch with CUDA support:

1. Navigate to the [PyTorch Get Started page](https://pytorch.org/get-started/locally/).
2. Select the appropriate configuration for your system, including your OS, package manager (`conda` or `pip`), Python version, and the CUDA version installed on your system.
3. Follow the generated command to install PyTorch. It will look something like this for `pip`:

   ```shell
   pip install torch torchvision torchaudio cudatoolkit=11.3 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   
   Or this for Conda:

   ```shell
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   ```
   
   Make sure to replace `11.3` with the version of CUDA Toolkit you have installed.

### Testing PyTorch CUDA Installation

After installing PyTorch, you can test if it's using GPU capabilities by checking CUDA availability:

```python3
import torch

# Returns True if CUDA is available, and PyTorch is using CUDA 
cuda_available = torch.cuda.is_available()

# Print CUDA availability status
print(f"CUDA available: {cuda_available}")

# If CUDA is available, print the name of the GPU(s)
if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPU(s) available: {gpu_count}")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Please check your installation.")
```

Copy and run the above Python code in your environment to verify the availability of GPU support in PyTorch.


## Setting Up Weights & Biases (wandb)

Weights & Biases (wandb) is a tool for experiment tracking, model optimization, and dataset versioning. Follow the instructions below to set up wandb in your local environment.

### Installation

Install `wandb` using pip with the following command (it is there in the `requirements.txt` file either way):

```shell
pip install wandb
```

### Configuration

After installation, you need to log in to wandb and set up your API key:

1. If you don't have a Weights & Biases account, create one at [wandb.ai](https://wandb.ai/).
2. Once you have your account, find your API key in your wandb profile settings under the API keys section.
3. Log in to wandb by running the following command and entering your API key when prompted:

```shell
wandb login
```

You can also log in silently by running:

```shell
wandb login your-api-key
```

Replace `your-api-key` with the actual API key you obtained from your wandb profile.

### Usage

To start tracking a new experiment, simply add `wandb.init()` to your training script:

```python
import wandb

# Initialize a new run
wandb.init(project="my_project_name", entity="my_wandb_username")

# Optionally, you can log hyperparameters using wandb.config, but don't
wandb.config.batch_size = 64
wandb.config.learning_rate = 0.001

# Log metrics inside your training loop
wandb.log({"loss": loss_value})
```

Replace `my_project_name` with the name of your project and `my_wandb_username` with your wandb username.

For detailed documentation and additional configuration options, visit the [wandb documentation](https://docs.wandb.ai/).

### Note

- Keep your API key private; do not include it in scripts or upload it to public repositories.
- Configure your `.gitignore` file to ignore the `wandb` directory that is created after your first run. This directory contains local copies of your experiment data.


## `config.py` File Description

This `config.py` file contains the configuration settings used for setting up and running the knowledge distillation task, which involves a student model learning from a teacher model. Below are the descriptions of the different sections and their respective parameters:

### Seed
- `SEED`: The seed for random number generation used to ensure reproducibility.

### Data Paths
- `DATA`: This section contains paths to the data resources.
  - `VIDEOS_PATH`: The directory containing video files for the dataset.
  - `CAPTIONS_PATH`: Path to the CSV file with video captions.
  - `ENCODED_CAPTION_IDS`: Path to the pickle file with encoded captions.

### Callback Configuration
- `CALLBACK`: Settings related to saving the training checkpoints.
  - `dirpath`: The directory where model checkpoints are saved.
  - `filename`: The naming scheme for the checkpoint files.
  - `save_top_k`: The number of top-k checkpoints to save.
  - `monitor`: The metric to monitor for checkpointing.
  - `mode`: The mode ('max' or 'min') that decides which checkpoints to save.

### Logger Settings
- `LOGGER`: Configuration for the logging mechanism.
  - `save_dir`: The directory to save logs.
  - `name`: The name for the log directory.

### Training Parameters
- `TRAIN`: Configuration for the training process.
  - `STUDENT_MODEL_DEF`: The definition of the student model.
  - `TEACHER_MODEL_DEF`: The definition of the teacher model.
  - `TRAINER`: Parameters for the PyTorch Lightning Trainer.
    - `max_epochs`: The maximum number of training epochs.
    - `precision`: The precision setting for training (e.g., 16 for FP16).
    - `enable_checkpointing`: Whether to enable model checkpointing.
    - `strategy`: The strategy for distributed training.
  - `LR`: The learning rate for the optimizer.
  - `BATCH_SIZE`: The batch size for training.

### Model Definitions
- `MODEL`: The specific configurations for student and teacher models.
  - `StudentCandidateV1`: Configurations for the student model variant.
    - `image_enc_name`: Name of the image encoder.
    - `d_model`: The dimension of the model.
    - `n_head`: The number of heads in the multi-head attention mechanism.
    - `d_ffn`: The dimension of the feed-forward network.
    - `dropout`: The dropout rate.
    - `num_decoder_layers`: The number of decoder layers.
  - `GenerativeImageTextTeacher`: Configurations for the teacher model.
    - `param_path`: Path to the YAML file with parameters for the teacher model.
    - `pretrained_weights`: Path to the pretrained weights file.

### Weights & Biases Configuration
- `WANDB`: Settings for the Weights & Biases logging tool.
  - `MODE`: The mode for WANDB which can be 'online', 'offline', or 'disabled'.

Please ensure that these configurations are correctly set before initiating the training process to guarantee that the environment is correctly prepared for the knowledge distillation task.


## Training

- Once the setup has been completed, you can change the configuration `config.py` file to have the parameters required. The training script `src/train.py` contains initialization for the student and teacher models and a distillation trainer. 

- The distillation trainer in its `training_step` currently supports 4 loss functions, namely the Cross-Entropy Loss between the final output logits of the ground truth and the student model, the KL Divergence Loss, the Encoder Feature Activation Loss and the Decoder Feature Activation Loss between the student and the teacher model, which is calculated using the Mean Squared Error function as `torch.nn.MSELoss()`.

- It is recommended to start training with the default `config` file provided, for compounding loss terms that include KL Divergence loss, it is recommended to have a Batch Size equal to or greater than 8, and if it doesn't include the KL Divergence Loss, then its recommended to have a Batch Size greater than 12 (empirical). It is also recommended to keep the batch size higher if possible as higher batch sizes have been attributed to greater success chances in generic NLP tasks.

- For the Decoder Feature Map/Activation Losses, keep in mind that this would depend on the `num_decoder_layers` term of the student model, and you would need to manually configure the changes to this loss term to match the layers as neded. If not sure how to implement, avoid that loss term.

The script can be run as follows:

```bash
cd real-time-video-captioning
python3 -m src.train
```

If you use the _University of Toronto_ compute server or something similar, you can utilize the slurm cluster to allocate GPU memory to your run, which should look something like this

```bash
srun --partition {partition_name} -c {num_cores} --mem={memory} --gres={gpu} python3 -m src.train
```

This should read all the necessary configurations from the `config` file to initialize most if not all of the class instances in order to run the training script. If you have `wandb` configured, then it should log your run information in the `wandb` directory that is one level below the root. Each run will subsequently also be logged locally in the `results/run/{wandb_run_name}` directory which will be unique for each run.

This allows the option of running multiple tasks at the same time by just changing the config file after executing one run, allowing to save time and utilize compute if still avaiable.

After the run, the `callback` feature should save your latest model checkpoint(s) for the architecture specified in the `config` file. It is important to keep track of the architecture, as you will need it along with the `*.ckpt` checkpoint file for inference. Functionality has already been included in the `src/models/model.py` file to keep track of the architecture and log that in the same run directory mentioned earlier.

Each run should show you the losses you want to log or print (edit this in `src/models/model.py`) until the `max_epochs` count.


## Inference

This can only be done after training one successful model/task. You would need one existing model checkpoint for this in the form of `results/run/{wandb_run_name}/*.ckpt`. Inference is run against a particular `wandb` experiment run and its name. 

Update the `config` file to the same params included during that particular experiment training, and then use that experiment name to run inference. Mismatch in the config settings and model checkpoint will cause errors due to incoherent model `state_dict` params.

The inference is carried out in the following way:

```bash
cd real-time-video-captioning
python3 -m src.inference {wandb_run_name}
```

And that `wandb_run_name` can be found after atleast running the training once in the following directory:

```
.
├── results
│   └── run (creates after first run)
│        └── <wandb_run_name> <-- This is what we want
|             └── ...
```

This should load just the student model weights, freeze them using `model.eval()`, and then run the inference and compare with the ground truth captions from the test dataloader.


## Acknowledgements

We would like to extend our sincere gratitude to all those who contributed to the successful realization of the "Real-Time Video Captioning Systems" project. Special thanks to our course instructors and peers in the _CSC2231HS — Topics in Computer Systems: Visual and Mobile Computing Systems_ course at the _University of Toronto_, whose guidance and insightful feedback were invaluable.

Our appreciation also goes out to the [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) research team at Microsoft for their pioneering work and open-source code, which played a pivotal role in the development of our project.

This project was created for educational purposes and any content from external sources has been duly credited within the repository. All utilized datasets, models, and software are acknowledged and attributed to their respective authors and licenses, as detailed in the respective documentation and license files.


## Citation

If you find our work useful in your research, please consider citing:
@misc{
   real_time_video_captioning,
   title={Real-Time Video Captioning Systems},
   author={Ali, Faraz and Kugathasan, Paanugoban and Chettiar, Gautam},
   year={2024},
   publisher={GitHub},
   journal={GitHub repository},
   howpublished={https://github.com/farazali7/real-time-video-captioning}
}