# utils.py
import os
import torchvision.utils as vutils


def save_image_batch(tensor, path, nrow=4):
    """
    Saves a batch of image tensors as a single image grid.

    Args:
        tensor (Tensor): A batch of images with shape (B, C, H, W).
        path (str): Path to save the resulting image grid.
        nrow (int): Number of images per row in the grid.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the image grid with normalization for better visibility
    vutils.save_image(tensor, path, nrow=nrow, normalize=True)