import yaml
from pathlib import Path
import torch
from model import VAE
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load hyperparameters
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

# Generate and visualize images
with torch.no_grad():
    # Sample latent vectors from the prior distribution
    z = torch.randn(10, LATENT_DIM).to(DEVICE)
    # Generate images from the decoder
    generated_images = model.decode(z).view(-1, 28, 28)
    # Visualize generated images
    fig = make_subplots(rows=1, cols=10, subplot_titles=[f"Image {i+1}" for i in range(10)])
    for j in range(10):
        fig.add_trace(go.Heatmap(z=generated_images[j].cpu().numpy(), colorscale='gray', showscale=False),
                      row=1, col=j+1)
    fig.update_layout(height=300, width=1200, title_text="VAE Generated Images")
    fig.show()
