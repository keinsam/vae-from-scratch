import yaml
from pathlib import Path
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataloaders
from model import VAE

# Load hyperparameters
with open("hparams.yaml", "r") as f:
    hparams = yaml.safe_load(f)

INPUT_DIM = hparams["model"]["input_dim"]
HIDDEN_DIM = hparams["model"]["hidden_dim"]
LATENT_DIM = hparams["model"]["latent_dim"]
BATCH_SIZE = hparams["train"]["batch_size"]
NB_EPOCHS = hparams["train"]["nb_epochs"]
LEARNING_RATE = hparams["train"]["learning_rate"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load paths
model_name = "vae_v0"
log_dir = Path("logs")
model_dir = Path("models")
model_path = model_dir.joinpath(f"{model_name}.pth")
log_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=log_dir.joinpath(model_name))

# Load dataloaders
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# Initialize model and optimizer
model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NB_EPOCHS):
    epoch_loss = 0
    for i, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{NB_EPOCHS}"):
        # Forward pass
        x = x.to(DEVICE).view(-1, INPUT_DIM)
        x_recon, mu, sigma = model.forward(x)
        # Compute loss
        reconstruction_loss = nn.BCELoss(reduction="sum")(x_recon, x)
        kl_divergence = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        loss = reconstruction_loss + kl_divergence
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # Log to TensorBoard
        writer.add_scalar("Loss/reconstruction", reconstruction_loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar("Loss/kl_divergence", kl_divergence.item(), epoch * len(train_loader) + i)
        writer.add_scalar("Loss/total", loss.item(), epoch * len(train_loader) + i)
    # Compute average loss
    avg_loss = epoch_loss / len(train_loader)
    writer.add_scalar("Loss/average", avg_loss, epoch)
    print(f"Epoch {epoch + 1}/{NB_EPOCHS}, Average Loss: {avg_loss}")

# Save model
torch.save(model.state_dict(), model_path)

# Close TensorBoard writer
writer.close()
