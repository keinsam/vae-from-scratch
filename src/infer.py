from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE
from dataset import get_dataloaders
import matplotlib.pyplot as plt

# Load hyperparameters
import yaml
with open("hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)

INPUT_DIM = hparams["model"]["input_dim"]
HIDDEN_DIM = hparams["model"]["hidden_dim"]
LATENT_DIM = hparams["model"]["latent_dim"]
BATCH_SIZE = hparams["train"]["batch_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load paths
model_name = "vae_v0"
model_dir = Path("models")
model_path = model_dir.joinpath(f"{model_name}.pth")

# Load the trained model
model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load dataloaders
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# Function to visualize the original and reconstructed images
def show_reconstructed_images(model, dataloader, num_images=8):
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= 1:  # We only want to process one batch
                break
            x = x.to(DEVICE)
            x_recon, _, _ = model(x.view(-1, INPUT_DIM))  # Get reconstructed images

            # Reshape reconstructed images
            x_recon = x_recon.view(-1, 28, 28)
            x = x.view(-1, 28, 28)

            # Plot the original and reconstructed images
            fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
            for j in range(num_images):
                axes[0, j].imshow(x[j].cpu().numpy(), cmap="gray")
                axes[0, j].axis('off')
                axes[1, j].imshow(x_recon[j].cpu().numpy(), cmap="gray")
                axes[1, j].axis('off')
            plt.show()

# Visualize the reconstructed images
show_reconstructed_images(model, test_loader)

