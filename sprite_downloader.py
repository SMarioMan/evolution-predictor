import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def download_pokemon_sprites():
    """
    Downloads all Pokemon sprites from pokemondb.net, preserving the original folder structure.
    """
    # Base URL for the website
    base_url = "https://pokemondb.net"
    # The page that contains links to all Pokemon
    sprites_page_url = urljoin(base_url, "/sprites")

    # Create a directory to save the sprites
    save_dir = "pokemon_sprites"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    try:
        # --- 1. Get the main sprites page ---
        print(f"Fetching main sprites page: {sprites_page_url}")
        main_page_response = requests.get(sprites_page_url)
        main_page_response.raise_for_status()  # Raise an exception for bad status codes

        # --- 2. Parse the main page to find links to individual Pokemon pages ---
        soup = BeautifulSoup(main_page_response.content, "html.parser")
        pokemon_links = []
        # Find all 'a' tags with class 'infocard' which link to pokemon pages
        for a_tag in soup.find_all("a", class_="infocard"):
            href = a_tag.get("href")
            if href and href.startswith("/sprites/"):
                pokemon_links.append(urljoin(base_url, href))

        print(f"Found {len(pokemon_links)} Pokemon pages to scrape.")

        # --- 3. Loop through each Pokemon page and download sprites ---
        for i, pokemon_page_url in enumerate(pokemon_links):
            try:
                print(f"\n--- Processing Pokemon {i+1}/{len(pokemon_links)}: {pokemon_page_url} ---")

                # --- 4. Get the individual Pokemon page ---
                pokemon_page_response = requests.get(pokemon_page_url)
                pokemon_page_response.raise_for_status()

                # --- 5. Parse the Pokemon page to find sprite images ---
                pokemon_soup = BeautifulSoup(pokemon_page_response.content, "html.parser")
                # Find all 'img' tags whose 'src' attribute points to a sprite
                sprite_img_tags = pokemon_soup.find_all("img", src=lambda s: s and "img.pokemondb.net/sprites/" in s)

                if not sprite_img_tags:
                    print(f"No sprites found on {pokemon_page_url}")
                    continue

                print(f"Found {len(sprite_img_tags)} sprites to download.")

                # --- 6. Download each sprite and preserve folder structure ---
                for img_tag in sprite_img_tags:
                    sprite_url = img_tag.get("src")
                    if not sprite_url:
                        continue

                    try:
                        # Find the path relative to the 'sprites' directory
                        # e.g., for '.../sprites/home/normal/bulbasaur.png', get 'home/normal/bulbasaur.png'
                        if 'sprites/' in sprite_url:
                            # Using split with a limit of 1 to handle potential 'sprites' in filenames
                            relative_path = sprite_url.split('sprites/', 1)[1]

                            # Create the full local path including subdirectories
                            save_path = os.path.join(save_dir, relative_path)

                            # Create the necessary subdirectories if they don't exist
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)

                            # Download the image
                            print(f"Downloading {sprite_url} to {save_path}...")
                            sprite_response = requests.get(sprite_url)
                            sprite_response.raise_for_status()

                            # Save the image to the local directory
                            with open(save_path, "wb") as f:
                                f.write(sprite_response.content)
                        else:
                            print(f"Skipping URL (unexpected format): {sprite_url}")

                    except (requests.exceptions.RequestException, IndexError) as e:
                        print(f"Failed to download or save sprite from {sprite_url}: {e}")

            except requests.exceptions.RequestException as e:
                print(f"Could not process {pokemon_page_url}: {e}")

        print("\nAll sprites have been downloaded successfully!")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the main page: {e}")

if __name__ == "__main__":
    download_pokemon_sprites()
