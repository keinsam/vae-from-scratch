import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import get_dataloaders
from model import VAE

# Load configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

INPUT_DIM = config["model"]["input_dim"]
HIDDEN_DIM = config["model"]["hidden_dim"]
LATENT_DIM = config["model"]["output_dim"]
BATCH_SIZE = config["train"]["batch_size"]
EPOCHS = config["train"]["epochs"]
LEARNING_RATE = config["train"]["learning_rate"]

# Load dataloaders
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)


