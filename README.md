# Variational Autoencoder (VAE) in PyTorch

## Overview

This repository contains an implementation of a **Variational Autoencoder (VAE)** using PyTorch, based on the original paper by **Kingma and Welling (2013)**.

## Features

- Implementation of a **VAE** in PyTorch.
- Training on the **MNIST dataset**.
- Logging with TensorBoard.
- Generation of images with visualization in Plotly.

## Usage

### Training
To train the VAE, run:
```bash
python train.py
```
This script will:
- Train the model.
- Save the trained model in `MODEL_DIR`.
- Log the loss values in TensorBoard.

To see the plotted loss functions, run ```tensorboard --logdir=logs``` in the terminal.

### Inference
To visualize reconstructed images from the trained model, run:
```bash
python infer.py
```
This script will:
- Load the trained model.
- Display generated images.

## References

- **Auto-Encoding Variational Bayes** – Kingma & Welling (2013) [[Paper](https://arxiv.org/abs/1312.6114)]