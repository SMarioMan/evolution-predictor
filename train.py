# train.py
import os
import torch
from torch.utils.data import DataLoader
from models.dual_diffusion_pipeline import DualDiffusionPipeline
from dataset import EvolutionaryPairDataset
from evolution_lines import evolution_lines
from tqdm import tqdm
from torchvision.utils import save_image
import yaml


def generate_and_save_samples(model, x_t, x_t1, epoch, output_dir, device):
    """
    Generates forward and backward predictions and saves side-by-side comparisons to disk.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Only take a small number of samples to visualize
        x_t_sample = x_t[:4].to(device)
        x_t1_sample = x_t1[:4].to(device)

        # Forward prediction: x_t → x_t+1
        x_t1_pred = model.generate(model.forward_model, x_t_sample)

        # Backward prediction: x_t+1 → x_t
        x_t_pred = model.generate(model.backward_model, x_t1_sample)

        # Save comparisons for forward prediction
        for i in range(x_t_sample.size(0)):
            grid = torch.stack([
                x_t_sample[i].cpu().clamp(0, 1),        # input x_t
                x_t1_pred[i].cpu().clamp(0, 1),         # predicted x_t+1
                x_t1_sample[i].cpu().clamp(0, 1),       # ground truth x_t+1
            ])
            save_image(grid, os.path.join(output_dir, f"e{epoch}_fwd_{i}.png"), nrow=3)

        # Save comparisons for backward prediction
        for i in range(x_t_sample.size(0)):
            grid = torch.stack([
                x_t1_sample[i].cpu().clamp(0, 1),       # input x_t+1
                x_t_pred[i].cpu().clamp(0, 1),          # predicted x_t
                x_t_sample[i].cpu().clamp(0, 1),        # ground truth x_t
            ])
            save_image(grid, os.path.join(output_dir, f"e{epoch}_bwd_{i}.png"), nrow=3)


def train():
    # Load hyperparameters and settings from config.yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Device selection (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Load dataset from the specified directory
    dataset = EvolutionaryPairDataset(evolution_lines=evolution_lines,
                                      image_size=config["image_size"],
                                      sprite_type_path=config["sprite_type_path"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize the dual diffusion pipeline model
    model = DualDiffusionPipeline(image_size=config["image_size"]).to(device)

    # Define optimizer (Adam works well for diffusion models)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
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
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}.")

    # Number of training epochs
    num_epochs = config["epochs"]

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0

        # Iterate through batches of image pairs
        for x_t, x_t1 in tqdm(dataloader, desc=f"Epoch {epoch}"):
            x_t, x_t1 = x_t.to(device), x_t1.to(device)

            optimizer.zero_grad()

            # Perform a training step and compute all components of the loss
            loss, loss_f, loss_b, loss_c = model.training_step(
                x_t, x_t1, lambda_cycle=config["lambda_cycle"]
            )

            # Backpropagate and optimize
            loss.backward()
            optimizer.step()

            # Accumulate total loss for reporting
            total_loss += loss.item()

        # Print losses for monitoring
        print(
            f"Epoch {epoch}: Total Loss={total_loss:.4f} | "
            f"Forward={loss_f:.4f} | Backward={loss_b:.4f} | Cycle={loss_c:.4f}"
        )

        # Save checkpoint with model + optimizer + epoch info
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        # Save sample outputs for visualization
        sample_dir = os.path.join("samples", config["sprite_type_path"])
        generate_and_save_samples(model, x_t, x_t1, epoch, sample_dir, device)

    # Save final model separately
    save_path = os.path.join("checkpoints", config["sprite_type_path"])
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "final.pth"))
    print(f"Model saved to {config['save_path']}")


if __name__ == "__main__":
    train()
