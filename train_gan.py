# train_gan.py
import os
import torch
from torch.utils.data import DataLoader
from models.dual_pix2pix_pipeline import DualPix2PixPipeline
from dataset import EvolutionaryPairDataset
from evolution_lines import evolution_lines
from tqdm import tqdm
from torchvision.utils import save_image
import yaml

def generate_and_save_samples(model, x_t, x_t1, epoch, output_dir, device):
    """Generates and saves samples using the pix2pix generators."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        x_t_sample = x_t[:4].to(device)
        x_t1_sample = x_t1[:4].to(device)

        # Forward prediction: x_t -> x_t1
        x_t1_pred = model.forward_generator(x_t_sample)
        # Backward prediction: x_t+1 -> x_t
        x_t_pred = model.backward_generator(x_t1_sample)

        # Save comparisons
        save_image(torch.cat([x_t_sample, x_t1_pred, x_t1_sample], -1),
                   os.path.join(output_dir, f"e{epoch}_fwd.png"), normalize=True)
        save_image(torch.cat([x_t1_sample, x_t_pred, x_t_sample], -1),
                   os.path.join(output_dir, f"e{epoch}_bwd.png"), normalize=True)

        # Bonus: Try unseen evolutions to see how well the model generalizes.
        x_t2_pred = model.forward_generator(x_t1_sample)
        x_t0_pred = model.backward_generator(x_t_sample)
        save_image(torch.cat([x_t1_sample, x_t2_pred], -1),
                   os.path.join(output_dir, f"e{epoch}_fwd_new.png"), normalize=True)
        save_image(torch.cat([x_t_sample, x_t0_pred], -1),
                   os.path.join(output_dir, f"e{epoch}_bwd_new.png"), normalize=True)

def train():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Dataset
    dataset = EvolutionaryPairDataset(evolution_lines=evolution_lines,
                                      image_size=config["image_size"],
                                      sprite_type_path=config["sprite_type_path"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # Model
    model = DualPix2PixPipeline().to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(model.get_generator_params(), lr=config["lr"], betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model.get_discriminator_params(), lr=config["lr"], betas=(0.5, 0.999))

    # Checkpoint loading and saving
    start_epoch = 0
    checkpoint_dir = os.path.join("checkpoints_gan", config["sprite_type_path"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for latest checkpoint to resume
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if checkpoint_files:
        latest_checkpoint = sorted(checkpoint_files)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")

    lambda_recon = 10.0  # As recommended in pix2pix paper
    lambda_cycle = config["lambda_cycle"]

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        g_loss_acc, d_loss_acc = 0.0, 0.0

        for x_t, x_t1 in tqdm(dataloader, desc=f"Epoch {epoch}"):
            x_t, x_t1 = x_t.to(device), x_t1.to(device)
            
            # Create adversarial ground truths
            valid = torch.ones(x_t.size(0), 1, 8, 8, device=device) # PatchGAN output size
            fake = torch.zeros(x_t.size(0), 1, 8, 8, device=device)

            # --- Train Generators ---
            optimizer_G.zero_grad()
            
            # Forward generation
            fake_t1 = model.forward_generator(x_t)
            # Backward generation
            fake_t = model.backward_generator(x_t1)

            # Adversarial loss for generators
            loss_G_fwd = model.adversarial_loss(model.forward_discriminator(x_t, fake_t1), valid)
            loss_G_bwd = model.adversarial_loss(model.backward_discriminator(x_t1, fake_t), valid)

            # Reconstruction loss (L1)
            loss_L1_fwd = model.reconstruction_loss(fake_t1, x_t1)
            loss_L1_bwd = model.reconstruction_loss(fake_t, x_t)

            # Cycle consistency loss
            reconstructed_t = model.backward_generator(fake_t1)
            loss_cycle_t = model.cycle_loss(reconstructed_t, x_t)
            
            reconstructed_t1 = model.forward_generator(fake_t)
            loss_cycle_t1 = model.cycle_loss(reconstructed_t1, x_t1)
            
            # Total Generator Loss
            loss_G = (loss_G_fwd + loss_G_bwd) + \
                     (loss_L1_fwd + loss_L1_bwd) * lambda_recon + \
                     (loss_cycle_t + loss_cycle_t1) * lambda_cycle

            loss_G.backward()
            optimizer_G.step()
            
            # --- Train Discriminators ---
            optimizer_D.zero_grad()

            # Forward discriminator loss
            loss_D_fwd_real = model.adversarial_loss(model.forward_discriminator(x_t, x_t1), valid)
            loss_D_fwd_fake = model.adversarial_loss(model.forward_discriminator(x_t, fake_t1.detach()), fake)
            loss_D_fwd = (loss_D_fwd_real + loss_D_fwd_fake) * 0.5
            
            # Backward discriminator loss
            loss_D_bwd_real = model.adversarial_loss(model.backward_discriminator(x_t1, x_t), valid)
            loss_D_bwd_fake = model.adversarial_loss(model.backward_discriminator(x_t1, fake_t.detach()), fake)
            loss_D_bwd = (loss_D_bwd_real + loss_D_bwd_fake) * 0.5
            
            # Total Discriminator Loss
            loss_D = loss_D_fwd + loss_D_bwd

            loss_D.backward()
            optimizer_D.step()

            g_loss_acc += loss_G.item()
            d_loss_acc += loss_D.item()

        avg_g_loss = g_loss_acc / len(dataloader)
        avg_d_loss = d_loss_acc / len(dataloader)
        print(f"Epoch {epoch}: G Loss={avg_g_loss:.4f} | D Loss={avg_d_loss:.4f}")

        # Save sample outputs for visualization
        sample_dir = os.path.join("samples_gan", config["sprite_type_path"])
        generate_and_save_samples(model, x_t, x_t1, epoch, sample_dir, device)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch} at {checkpoint_path}")

if __name__ == "__main__":
    train()