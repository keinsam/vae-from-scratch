import yaml
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloaders
from model import VAE

# Load hyperparameters
with open("configs/hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)

INPUT_DIM = hparams["model"]["input_dim"]
HIDDEN_DIM = hparams["model"]["hidden_dim"]
LATENT_DIM = hparams["model"]["latent_dim"]
BATCH_SIZE = hparams["train"]["batch_size"]
NB_EPOCHS = hparams["train"]["nb_epochs"]
LEARNING_RATE = hparams["train"]["learning_rate"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load paths
with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

MODEL_NAME = Path(paths["model_name"])
LOG_DIR = Path(paths["log_dir"])
MODEL_DIR = Path(paths["model_dir"])
MODEL_PATH = MODEL_DIR.joinpath(f"{MODEL_NAME}.pth")
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=LOG_DIR.joinpath(MODEL_NAME))
writer.add_hparams({"input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM, "latent_dim": LATENT_DIM,
                    "batch_size": BATCH_SIZE, "nb_epochs": NB_EPOCHS, "learning_rate": LEARNING_RATE,},
                    {})

# Load dataloaders
train_loader = get_dataloaders(batch_size=BATCH_SIZE)

# Initialize model and optimizer
model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NB_EPOCHS):
    model.train()
    epoch_recon, epoch_kl = 0, 0
    for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NB_EPOCHS}"):
        x = x.to(DEVICE).view(-1, INPUT_DIM)
        # Forward pass
        x_recon, mu, sigma = model.forward(x)
        sigma = torch.clamp(sigma, min=1e-6)  # Stability
        # Compute loss
        reconstruction_loss = nn.BCELoss(reduction='sum')(x_recon, x) / BATCH_SIZE
        kl_divergence = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) / BATCH_SIZE
        loss = reconstruction_loss + kl_divergence
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Logging
        epoch_recon += reconstruction_loss.item()
        epoch_kl += kl_divergence.item()
    # Compute average loss
    avg_recon = epoch_recon / len(train_loader)
    avg_kl = epoch_kl / len(train_loader)
    print(f"Epoch {epoch+1}: Total Loss: {avg_recon + avg_kl:.4f}")
    # Log to TensorBoard
    writer.add_scalar('Epoch/Reconstruction', avg_recon, epoch)
    writer.add_scalar('Epoch/KL_Divergence', avg_kl, epoch)
    writer.add_scalar('Epoch/Total', avg_recon + avg_kl, epoch)

# Save model
torch.save(model.state_dict(), MODEL_PATH)

# Close TensorBoard writer
writer.close()
