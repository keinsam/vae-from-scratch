import yaml
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from dataset import MNIST, CIFAR10
from model import VAE
from train import train_vae


# Load paths
with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)
MODEL_NAME = Path(paths["model_name"])
DATA_DIR = Path(paths["data_dir"])
LOGS_DIR = Path(paths["logs_dir"])
WEIGHTS_DIR = Path(paths["weights_dir"])
MODEL_PATH = WEIGHTS_DIR.joinpath(f"{MODEL_NAME}.pth")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Load hyperparameters
with open("configs/hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)
CHANNEL_DIM = hparams["model"]["channel_dim"]
HIDDEN_DIM = hparams["model"]["hidden_dim"]
LATENT_DIM = hparams["model"]["latent_dim"]
BATCH_SIZE = hparams["train"]["batch_size"]
NB_EPOCHS = hparams["train"]["nb_epochs"]
LEARNING_RATE = hparams["train"]["learning_rate"]
DATASET_NAME = hparams["dataset"]["name"]
SUBSET_SIZE = hparams["dataset"]["subset_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=LOGS_DIR.joinpath(MODEL_NAME))
flat_hparams = {k: v for section in hparams.values() for k, v in section.items()}
writer.add_hparams(flat_hparams, {})

# Load dataset
transforms = Compose([
    Resize(32),
    ToTensor(),
    # Normalize([0.5 for _ in range(CHANNEL_DIM)], [0.5 for _ in range(CHANNEL_DIM)]),
])
if DATASET_NAME == "mnist" :
    dataset = MNIST(DATA_DIR, train=True, download=True, transform=transforms, subset_size=SUBSET_SIZE)
if DATASET_NAME == "cifar10" :
    dataset = CIFAR10(DATA_DIR, train=True, download=True, transform=transforms, subset_size=SUBSET_SIZE)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model and optimizer
model = VAE(channel_dim=CHANNEL_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)




if __name__ == "__main__" :
    train_vae(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=DEVICE,
        nb_epochs=NB_EPOCHS,
        model_path=MODEL_PATH,
        writer=writer
    )