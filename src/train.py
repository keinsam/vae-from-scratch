import yaml
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import get_dataloaders
from model import VAE

# Load hyperparameters file
with open("hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)

# Load hyperparameters
INPUT_DIM = hparams["model"]["input_dim"]
HIDDEN_DIM = hparams["model"]["hidden_dim"]
LATENT_DIM = hparams["model"]["latent_dim"]
BATCH_SIZE = hparams["train"]["batch_size"]
EPOCHS = hparams["train"]["epochs"]
LEARNING_RATE = hparams["train"]["learning_rate"]

# Load dataloaders
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# Initialize model
model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

