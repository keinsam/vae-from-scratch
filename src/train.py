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
torch.save(model.state_dict(), MODEL_PATH)

# Close TensorBoard writer
writer.close()
