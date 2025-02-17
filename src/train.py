import yaml
import wandb  # Import W&B
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

# Initialize Weights & Biases (W&B) for logging
wandb.init(project="vae-training", config=hparams)

# Load dataloaders
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# Initialize model
model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss function
loss_function = nn.MSELoss()

# Training loop with logging
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        x, _ = batch  # Assuming (data, labels) from DataLoader
        x_reconstructed, _, _ = model(x)
        
        loss = loss_function(x_reconstructed, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Log the loss
    wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Save the model and log it in W&B
model_path = "vae_model.pth"
torch.save(model.state_dict(), model_path)
wandb.save(model_path)
wandb.finish()
