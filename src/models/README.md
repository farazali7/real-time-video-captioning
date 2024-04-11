# `model.py` - Model Definitions

The `model.py` file contains the core model definitions for the project, including custom implementations for various neural network architectures and the main distillation trainer module used for knowledge distillation.


## Key Components

- **TinyVIT**: A simple class wrapping around a pretrained Vision Transformer (ViT) from the `timm` library, extracting feature maps for images.
- **StudentCandidateV1**: Defines the student model for knowledge distillation, utilizing the TinyVIT encoder and a Transformer decoder.
- **PositionalEncoding**: Implements the positional encoding used in Transformers.
- **GenerativeImageTextModel**: Represents the reimplementation of the GIT model, which outputs the final logits, visual features, and decoder hidden states.
- **GenerativeImageTextTeacher**: A wrapper class that includes the teacher model along with its tokenizer, handling the forwarding of outputs and teacher-specific activations.
- **DistillationTrainer**: A PyTorch Lightning module tailored for training with knowledge distillation. This includes loss function definitions, training, validation, and test steps, along with hooks for capturing intermediate activations.


## Distillation Training

The `DistillationTrainer` class is the central training module that combines student and teacher models. It orchestrates the knowledge distillation process by comparing the student's outputs with those of the teacher. The trainer uses several loss functions to align the student's behavior with the teacher's, including feature map losses and KL divergence for logits alignment.


## Activation Hooks

The module employs hooks to capture intermediate activations from both the student and teacher models. These activations are essential for feature-based distillation, allowing the student model to learn from the teacher's internal representations.


## Usage

To utilize these models for training and inference, instantiate the `StudentCandidateV1` and `GenerativeImageTextTeacher` with the required parameters, and pass them to a `DistillationTrainer` instance. During training, the trainer will leverage the `forward` methods of these models to perform knowledge distillation.


## Additional Notes

- The `GenerativeImageTextModel` class provides custom implementations for decoding steps, including greedy and beam search decoding, which are crucial during the inference phase.
- The code includes device-specific settings to ensure compatibility with different hardware accelerations like CUDA and MPS.

Make sure to configure the model parameters and training settings according to your project needs.