# Functional Neural Field (FCNF) Pretraining

A PyTorch framework for training Functional Autoencoders and Neural Fields on physical datasets. This repository supports various architectures including SIREN, Transolver, and Perceiver, with integrated quantization (VQ/FSQ) capabilities.

## 🚀 Features
- **Multiple Architectures**: Supports SIREN, Transolver, MLP, and Perceiver-based autoencoders.
- **Quantization**: Built-in support for Vector Quantization (VQ) and Finite Scalar Quantization (FSQ).
- **Physical Datasets**: Specialized loaders for Kolmogorov Flow and Inlet Flow data.
- **Lightning Powered**: Training logic implemented using PyTorch Lightning for scalability and reproducible experiments.
- **Flexible Configuration**: Uses Hydra for granular control over model, data, and training parameters.

## 📂 Project Structure
- `src/fcnf/`: Model definitions and components.
- `src/engine/`: Training logic (`FAETrainer`).
- `src/utils/`: Data processing and utilities.
- `configs/`: YAML configuration files for experiments.
- `scripts/`: Python scripts to launch training.
- `bash/`: Helper shell scripts for common tasks.

## 🛠️ Getting Started

### Installation
Ensure you have the required dependencies installed:
```bash
pip install torch lightning omegaconf hydra-core einops wandb h5py
```

### Training
To start a training run, you can use the provided bash scripts or call the training script directly with Hydra:

**Using Bash script:**
```bash
bash bash/fcnf.sh
```

**Using Python script directly:**
```bash
python3 scripts/train_dim_red.py --config-name fcnf_HIT
```

### Configuration
Configurations are managed via Hydra in the `configs/` directory. You can override any parameter through the CLI:
```bash
python3 scripts/train_dim_red.py --config-name fcnf_HIT training.batch_size=128 model.latent_features=256
```

## 📊 Logging & Visualization
The project integrates with **Weights & Biases (WandB)** for experiment tracking. Ensure you have your `WANDB_API_KEY` set. Visualizations of the data can be found in `dataset/see_the_data.ipynb`.

## ⚖️ License
[Specify License here, e.g., MIT]
