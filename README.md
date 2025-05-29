# Variational Autoencoder (VAE) in PyTorch

## Overview

This repository contains an implementation of a **Variational Autoencoder (VAE)** using PyTorch, based on the original paper by **Kingma and Welling (2013)**, with a few modern enhancements.

## Features

- Implementation of a **VAE** in PyTorch.
- Training on the **MNIST and CIFAR10** datasets.
- Logging with TensorBoard.

## Usage

To train the VAE, run:
```bash
python run.py
```
This script will:
- Train the model.
- Save the trained model in `WEIGHTS_DIR`.
- Log the loss values and generated samples in TensorBoard.

To see the plotted loss functions and the generated samples, run ```tensorboard --logdir=logs``` in the terminal.

## References

- **Auto-Encoding Variational Bayes** – Kingma & Welling (2013) [[Paper](https://arxiv.org/abs/1312.6114)]
