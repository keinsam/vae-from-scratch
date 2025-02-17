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
EPOCHS = hparams["train"]["epochs"]
LEARNING_RATE = hparams["train"]["learning_rate"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=Path("logs"))

# Load data
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# Initialize model, optimizer, and loss function
model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.BCELoss(reduction="mean")

# Training loop
for epoch in range(EPOCHS):
    for i, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        # Forward pass
        x = x.to(DEVICE).view(-1, INPUT_DIM)
        x_recon, mu, sigma = model.forward(x)
        # Compute loss
        reconstruction_loss = loss_function(x_recon, x)
        kl_divergence = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) / 2
        # Backward pass
        loss = reconstruction_loss + kl_divergence
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Log to TensorBoard
        writer.add_scalar("Loss/reconstruction", reconstruction_loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar("Loss/kl_divergence", kl_divergence.item(), epoch * len(train_loader) + i)
        writer.add_scalar("Loss/total", loss.item(), epoch * len(train_loader) + i)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")


# Save model
model_path = Path("models/model.pth")
torch.save(model.state_dict(), model_path)

# Close TensorBoard writer
writer.close()
