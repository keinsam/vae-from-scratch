from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE
from dataset import get_dataloaders
import matplotlib.pyplot as plt

# Load hyperparameters
import yaml
with open("configs/hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)

INPUT_DIM = hparams["model"]["input_dim"]
HIDDEN_DIM = hparams["model"]["hidden_dim"]
LATENT_DIM = hparams["model"]["latent_dim"]
BATCH_SIZE = hparams["train"]["batch_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load paths
with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

MODEL_NAME = Path(paths["model_name"])
MODEL_DIR = Path(paths["model_dir"])
MODEL_PATH = MODEL_DIR.joinpath(f"{MODEL_NAME}.pth")

# Load the trained model
model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def generate_images(model, num_images=8):
    model.eval()
    with torch.no_grad():
        # Sample latent vectors from the prior distribution
        z = torch.randn(num_images, LATENT_DIM).to(DEVICE)  # (batch_size, latent_dim)
        
        # Generate images from the decoder
        generated_images = model.decode(z).view(-1, 28, 28)  # Reshape if working with 28x28 images

        # Plot generated images
        fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
        for j in range(num_images):
            axes[j].imshow(generated_images[j].cpu().numpy(), cmap="gray")
            axes[j].axis('off')
        plt.show()

# Generate and visualize images
generate_images(model, num_images=4)

