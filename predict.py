# predict.py
import torch
import argparse
import yaml
from models.dual_diffusion_pipeline import DualDiffusionPipeline
from PIL import Image
import torchvision.transforms as T


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(config, device):
    model = DualDiffusionPipeline(image_size=config["image_size"]).to(device)
    model.load_state_dict(torch.load(config["save_path"], map_location=device))
    model.eval()
    return model


def preprocess_image(image_path, image_size):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # Add batch dimension


def postprocess_and_save(tensor_img, output_path):
    to_pil = T.ToPILImage()
    img = to_pil(tensor_img.squeeze(0).cpu().clamp(0, 1))
    img.save(output_path)
    print(f"Saved output image to {output_path}")


def main():
    # Setup CLI arguments
    parser = argparse.ArgumentParser(description="Generate evolutionary prediction using a trained diffusion model.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image (x_t or x_t+1)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image")
    parser.add_argument("--direction", type=str, choices=["forward", "backward"], default="forward",
                        help="Direction of prediction: 'forward' for x_t → x_t+1, 'backward' for x_t+1 → x_t")

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(config, device)

    # Prepare input
    x_input = preprocess_image(args.input, config["image_size"]).to(device)

    # Run generation
    with torch.no_grad():
        if args.direction == "forward":
            output = model.generate(model.forward_model, x_input)
        else:
            output = model.generate(model.backward_model, x_input)

    # Save output
    postprocess_and_save(output, args.output)


if __name__ == "__main__":
    main()

# Usage examples:
# Forward evolution (x_t -> x_t+1)
# python predict.py --input sprites/charmeleon.png --output output/charizard_pred.png --direction forward
# Backward evolution (x_t+1 -> x_t)
# python predict.py --input sprites/charizard.png --output output/charmeleon_pred.png --direction backward
