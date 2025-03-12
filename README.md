# Variational Autoencoder (VAE) in PyTorch

## Overview

This repository contains an implementation of a **Variational Autoencoder (VAE)** using PyTorch, based on the original paper by **Kingma and Welling (2013)**.

## Features

- Implementation of a **fully connected VAE** in PyTorch.
- Training on the **MNIST dataset**.
- Logging with **TensorBoard**.
- Reconstruction of images for visualization.

## Project Structure
```
├── configs/
│   ├── hparams.yaml       # Hyperparameters configuration
│   ├── paths.yaml         # Paths for saving models/logs
│
├── src/
    ├── dataset.py         # Data loading utilities
    ├── infer.py           # Inference script
    ├── model.py           # VAE model
    ├── train.py           # Training script
```

## Model Architecture

The VAE consists of:
- An **encoder** that maps input images to a latent space.
- A **reparameterization trick** to sample from the latent distribution.
- A **decoder** that reconstructs images from the latent space.

## Usage

### Training
To train the VAE, run:
```bash
python train.py
```
This script will:
- Train the model for the specified number of epochs.
- Save the trained model in `MODEL_DIR`.
- Log the loss values in TensorBoard.

### Inference
To visualize reconstructed images from the trained model, run:
```bash
python infer.py
```
This script will:
- Load the trained model.
- Display original vs. reconstructed images from the test set.

## References

- **Auto-Encoding Variational Bayes** – Kingma & Welling (2013) [[Paper](https://arxiv.org/abs/1312.6114)]