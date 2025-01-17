# NCF_Model

I am making part of my NCF model public for the purposes of reuse and building in public. Majority of the business logic that the NCF is used for is located in the @LookAlive (A company I have co-founded) repository which we have chosen to make private as it will be a monetized application.

# Neural Collaborative Filtering (NCF) Implementation

A PyTorch implementation of Neural Collaborative Filtering (NCF) for building recommendation systems. This implementation includes both Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) components (as expected from the paper), with support for distributed training, cross-validation, and a very configurable model architecture.

This implementation is based on the paper ["Neural Collaborative Filtering"](https://arxiv.org/abs/1708.05031) by He et al., published at WWW 2017. If you use this code, please cite:

```bibtex
@inproceedings{he2017neural,
  title={Neural Collaborative Filtering},
  author={He, Xiangnan and Liao, Lizi and Zhang, Hanwang and Nie, Liqiang and Hu, Xia and Chua, Tat-Seng},
  booktitle={Proceedings of the 26th International Conference on World Wide Web},
  pages={173--182},
  year={2017}
}
```

The NCF framework is designed to learn the complex user-item interactions by replacing the inner product with a neural architecture that can learn an arbitrary function from data. The model leverages the flexibility and non-linearity of neural networks to learn the interaction function from data, rather than using a fixed inner product. This allows it to express and learn more complex patterns compared to traditional matrix factorization techniques.

## Features

- Neural Collaborative Filtering model implementation
- Support for both CPU and GPU training
- Distributed training support for scaling to multiple GPUs
- Cross-validation functionality
- Configurable model architecture
- Checkpoint saving and loading
- Batch normalization and dropout for better regularization

## Requirements

- Python 3.6+
- PyTorch
- CUDA (optional, for GPU support)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ncf
```

2. Install the required dependencies:
```bash
pip install torch numpy argparse
```

## Usage

### Basic Training

To train the model with default parameters:

```bash
python main.py
```

### Custom Training

You can customize various parameters:

```bash
python main.py \
    --num_users 1000 \
    --num_items 2000 \
    --embedding_dim 8 \
    --layers 64 32 16 8 \
    --batch_size 256 \
    --epochs 10 \
    --learning_rate 0.001
```

### Cross-Validation

To perform cross-validation:

```bash
python main.py --cross_validate --n_splits 5
```

### Distributed Training

For multi-GPU training:

```bash
python main.py --distributed --world_size <num_gpus>
```

I have been training on a gcp instance with 4 3080s. I also train on my local machine which a single 12GB 3060 and it works fine just a bit slow.

## Parameters

- `--num_users`: Number of users in the dataset
- `--num_items`: Number of items in the dataset
- `--embedding_dim`: Dimension of embedding vectors
- `--layers`: MLP layer dimensions (space-separated)
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimization
- `--device`: Device to use (cuda/cpu)
- `--checkpoint_dir`: Directory to save model checkpoints
- `--cross_validate`: Enable cross-validation
- `--n_splits`: Number of cross-validation folds
- `--distributed`: Enable distributed training
- `--world_size`: Number of distributed processes
- `--dist_url`: URL for distributed training
- `--dist_backend`: Distributed backend (nccl/gloo)

## Model Architecture

The NCF model combines two components:
1. Generalized Matrix Factorization (GMF)
2. Multi-Layer Perceptron (MLP)

Both components use embedding layers for users and items, followed by a neural network architecture that learns the interaction between users and items.

## Contributing

Feel free to submit issues and enhancement requests!
