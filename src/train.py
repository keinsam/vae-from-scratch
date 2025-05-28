from tqdm import tqdm
import torch
from torch import nn
import torchvision


def train_vae(model, dataloader, optimizer, device, nb_epochs, model_path, writer=None):
    criterion = nn.BCELoss(reduction='sum')
    model.to(device)

    step = 0
    model.train()
    for epoch in range(nb_epochs):
        epoch_recon, epoch_kl = 0, 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{nb_epochs}")
        for x, _ in loop:
            x = x.to(device)

            # Forward pass
            x_recon, mu, sigma = model(x)
            
            # Compute losses
            reconstruction_loss = criterion(x_recon, x)
            kl_divergence = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = reconstruction_loss + kl_divergence

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            epoch_recon += reconstruction_loss.item()
            epoch_kl += kl_divergence.item()

            if writer is not None:
                writer.add_scalar("VAE/Reconstruction_Loss", reconstruction_loss.item(), global_step=step)
                writer.add_scalar("VAE/KL_Divergence", kl_divergence.item(), global_step=step)
                writer.add_scalar("VAE/Total_Loss", loss.item(), global_step=step)
            step += 1
        
        avg_recon = epoch_recon / len(dataloader.dataset)
        avg_kl = epoch_kl / len(dataloader.dataset)
        total = avg_recon + avg_kl

        print(f"Epoch [{epoch+1}/{nb_epochs}] - Total Loss: {total:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
        
        if writer is not None and (epoch + 1) % 2 == 0:
            with torch.no_grad():
                z = torch.randn(32, model.latent_dim).to(device)
                fixed_samples = model.decoder(z)
                grid = torchvision.utils.make_grid(fixed_samples, nrow=8, normalize=True)
                writer.add_image("VAE/Samples", grid, global_step=step)


    # Save model
    torch.save(model.state_dict(), model_path)
    
    if writer is not None:
        writer.close()
