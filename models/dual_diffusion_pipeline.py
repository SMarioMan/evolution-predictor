# dual_diffusion_pipeline.py
import torch
from diffusers import UNet2DModel, DDPMScheduler
from torch import nn

class DualDiffusionPipeline(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        # Forward model: Predicts x_t+1 from x_t (progression)
        self.forward_model = UNet2DModel(
            sample_size=image_size,             # Image dimensions (e.g., 64x64)
            in_channels=3,                      # RGB input
            out_channels=3,                     # RGB output
            layers_per_block=2,
            block_out_channels=(64, 128, 256),  # Channels in each layer
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D")
        )

        # Backward model: Predicts x_t from x_t+1 (regression)
        self.backward_model = UNet2DModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D")
        )

        # Scheduler for diffusion noise steps (forward and reverse processes)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def forward_step(self, model, x_input, x_target):
        """
        One training step for a diffusion model.
        Args:
            model: The diffusion model (forward or backward).
            x_input: Input (e.g., x_t or x_t+1).
            x_target: Target to predict (e.g., x_t+1 or x_t).
        Returns:
            MSE loss between model prediction and added noise.
        """
        noise = torch.randn_like(x_target)  # Random Gaussian noise
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (x_target.size(0),), device=x_target.device)

        # Add noise to the target image
        noisy = self.noise_scheduler.add_noise(x_target, noise, timesteps)

        # Predict the noise from the noisy image
        model_pred = model(noisy, timesteps, return_dict=False)[0]

        # Return MSE loss between predicted and true noise
        return nn.functional.mse_loss(model_pred, noise)

    def cycle_loss(self, x_t, x_t1):
        """
        Computes a cycle consistency loss:
        Forward model generates x_t+1, backward model reconstructs x_t.
        Args:
            x_t: Image at time t.
            x_t1: Image at time t+1.
        Returns:
            L2 loss between reconstructed x_t and true x_t.
        """
        with torch.no_grad():
            # Predict x_t+1 from x_t
            pred_t1 = self.generate(self.forward_model, x_t)

            # Reconstruct x_t from generated x_t+1
            pred_t = self.generate(self.backward_model, pred_t1)

        # L2 loss between reconstructed x_t and original x_t
        return nn.functional.mse_loss(pred_t, x_t)

    def generate(self, model, reference_tensor):
        """
        Generate an image by reversing the diffusion process.

        Args:
            model: The UNet2DModel (either forward or backward).
            reference_tensor: Used only for shape/device of the initial noise.

        Returns:
            A generated image (same shape as reference_tensor).
        """
        # Start with pure Gaussian noise of the same shape
        sample = torch.randn_like(reference_tensor)

        for t in reversed(range(self.noise_scheduler.config.num_train_timesteps)):
            t_tensor = torch.tensor([t] * sample.size(0), device=sample.device)

            # Predict noise and take a reverse diffusion step
            pred_noise = model(sample, t_tensor)[0]
            sample = self.noise_scheduler.step(pred_noise, t, sample).prev_sample

        return sample

    def training_step(self, x_t, x_t1, lambda_cycle=0.2):
        """
        Perform one training step with both models and include cycle consistency loss.
        Args:
            x_t: Image at time t.
            x_t1: Image at time t+1.
            lambda_cycle: Weighting for cycle consistency loss.
        Returns:
            Total loss and individual components for logging.
        """
        # Forward and backward model loss
        loss_f = self.forward_step(self.forward_model, x_t, x_t1)
        loss_b = self.forward_step(self.backward_model, x_t1, x_t)

        # Optional cycle consistency loss
        loss_cycle = self.cycle_loss(x_t, x_t1)

        # Total loss with regularization
        total_loss = loss_f + loss_b + lambda_cycle * loss_cycle

        return total_loss, loss_f.item(), loss_b.item(), loss_cycle.item()
