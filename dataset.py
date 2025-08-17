# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class EvolutionaryPairDataset(Dataset):
    def __init__(self, evolution_lines, root_dir="pokemon_sprites", image_size=128, sprite_type_path='home/normal/1x'):
        """
        Custom Dataset for loading Pokémon evolutionary image pairs.

        Args:
            evolution_lines (List[List[str]]): A list of lists, where each inner list
                                               is an evolutionary chain of Pokémon names.
            root_dir (str): The base directory where sprites were saved (e.g., 'pokemon_sprites').
            image_size (int): Images will be verified to be (image_size x image_size).
            sprite_type_path (str): The sub-path within the root_dir to find the sprites.
                                    The default 'home/normal/1x' matches the scraper's output
                                    for standard sprites.
        """
        self.evolution_lines = evolution_lines
        self.root_dir = root_dir
        self.sprite_type_path = sprite_type_path
        self.image_size = image_size
        
        # --- Identify all available sprites ---
        self.available_sprites = self._get_available_sprites()
        print(f"Found {len(self.available_sprites)} available sprites.")

        # --- Create a flat list of all possible (pre-evolution, post-evolution) pairs ---
        self.pairs = set()  # Initialize as a set to store unique pairs
        skipped_pairs_count = 0
        for line in self.evolution_lines:
            # For a line like ['a', 'b', 'c'], create pairs ('a', 'b') and ('b', 'c')
            for i in range(len(line) - 1):
                pre_evo = line[i]
                post_evo = line[i+1]

                pre_evo_filename = f"{pre_evo}.png"
                post_evo_filename = f"{post_evo}.png"

                if pre_evo_filename in self.available_sprites and post_evo_filename in self.available_sprites:
                    self.pairs.add((pre_evo, post_evo))
                else:
                    skipped_pairs_count += 1
                    if pre_evo_filename not in self.available_sprites:
                        print(f"Skipping pair ({pre_evo}, {post_evo}): Missing sprite for {pre_evo}")
                    if post_evo_filename not in self.available_sprites:
                        print(f"Skipping pair ({pre_evo}, {post_evo}): Missing sprite for {post_evo}")
        # Convert to a flat list
        self.pairs = list(self.pairs)
        
        print(f"Generated {len(self.pairs)} valid evolutionary pairs.")
        if skipped_pairs_count > 0:
            print(f"Skipped {skipped_pairs_count} pairs due to missing sprites.")
        
        # See if we have any unused sprites in the dataset
        self.validate_missing_evos()

        # Determine proper padding to fit in the GAN model
        left_pad = int((128-image_size)/2)
        right_pad = int((128-image_size)/2)
        if image_size % 2 != 0:
            right_pad += 1

        # Define image preprocessing transforms
        self.transform = T.Compose([
            T.Pad(padding=(left_pad, left_pad, right_pad, right_pad), fill=0),
            # T.Resize((image_size, image_size)), # Not needed since all sprites are the same size (hopefully).
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1] for Tanh
        ])

    def _get_available_sprites(self):
        """
        Scans the sprite directory and returns a set of available sprite filenames.
        """
        sprite_dir = os.path.join(self.root_dir, self.sprite_type_path)
        if not os.path.exists(sprite_dir):
            print(f"Warning: Sprite directory not found: {sprite_dir}")
            return set()
        
        available_files = set()
        for filename in os.listdir(sprite_dir):
            if filename.endswith(".png"): # Assuming all sprites are .png files
                available_files.add(filename)
        return available_files

    def __len__(self):
        """
        Returns:
            int: Total number of evolutionary pairs.
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Loads a single evolutionary pair.

        Args:
            idx (int): Index of the evolutionary pair.

        Returns:
            Tuple[Tensor, Tensor]: (pre-evolution image, post-evolution image)
        """
        # Get the names for the pre- and post-evolution Pokémon
        pre_evo_name, post_evo_name = self.pairs[idx]

        # --- Dynamically construct the file paths ---
        pre_evo_path = os.path.join(self.root_dir, self.sprite_type_path, f"{pre_evo_name}.png")
        post_evo_path = os.path.join(self.root_dir, self.sprite_type_path, f"{post_evo_name}.png")

        # Load and preprocess both images
        pre_evo_img = self.load_and_validate_image(pre_evo_path)
        post_evo_img = self.load_and_validate_image(post_evo_path)

        return pre_evo_img, post_evo_img
    
    def load_and_validate_image(self, image_path):
        """
        Loads an image, converts it to RGB, and validates its size.
        Raises a ValueError if the image size does not match self.image_size.
        """
        img = Image.open(image_path)
        
        # If the image has an alpha channel, composite it onto a white background
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use the alpha channel as mask
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get the actual size of the image (width, height)
        actual_width, actual_height = img.size

        if actual_width != self.image_size or actual_height != self.image_size:
            raise ValueError(
                f"Image at {image_path} has mismatched size. "
                f"Expected ({self.image_size}, {self.image_size}), "
                f"but got ({actual_width}, {actual_height})."
            )
        
        # Apply the remaining transforms
        return self.transform(img)
    
    def validate_missing_evos(self):
        pokemon_in_evolution_lines = set()
        for line in self.evolution_lines:
            for pokemon in line:
                pokemon_in_evolution_lines.add(pokemon)

        available_sprite_names = {os.path.splitext(name)[0] for name in self.available_sprites}

        unrepresented_sprites = available_sprite_names - pokemon_in_evolution_lines

        if unrepresented_sprites:
            print("\nPokémon sprites found in the directory but NOT represented in any evolutionary line:")
            for pokemon_name in sorted(list(unrepresented_sprites)):
                print(f"- {pokemon_name}")
        else:
            print("\nAll Pokémon sprites found in the directory are represented in an evolutionary line.")
        

# Example Usage (for testing purposes)
if __name__ == '__main__':
    print("\n--- Initializing Dataset ---")
    from evolution_lines import evolution_lines
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    dataset = EvolutionaryPairDataset(evolution_lines=evolution_lines,
                                      image_size=config["image_size"],
                                      sprite_type_path=config["sprite_type_path"])

    print(f"\nDataset length: {len(dataset)}")

    if len(dataset) > 0:
        print("\n--- Testing __getitem__ ---")
        try:
            for i in range(len(dataset)):
                pre_evo_img, post_evo_img = dataset[i]
                print(f"Successfully loaded pair: {dataset.pairs[i]}")
        except IndexError:
            print("No valid pairs to test __getitem__.")
    else:
        print("No valid evolutionary pairs were found in the dataset.")