import requests
import json

def get_all_evolution_chain_urls():
    """
    Fetches all evolution chain URLs from the PokeAPI.
    """
    base_url = "https://pokeapi.co/api/v2/evolution-chain/"
    evolution_chain_urls = []
    offset = 0
    limit = 100  # Number of evolution chains to fetch per request

    while True:
        # Construct the URL with offset and limit
        url = f"{base_url}?offset={offset}&limit={limit}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()

            # Extract URLs from the results
            for result in data['results']:
                evolution_chain_urls.append(result['url'])

            # Check if there are more results
            if data['next']:
                offset += limit
            else:
                break  # No more pages
        except requests.exceptions.RequestException as e:
            print(f"Error fetching evolution chain list: {e}")
            break
    return evolution_chain_urls

def extract_evolution_lines(chain_node, current_path, all_lines):
    """
    Recursively extracts all possible evolutionary lines from an evolution chain node.

    Args:
        chain_node (dict): The current node in the evolution chain (e.g., chain['chain'] or evolves_to item).
        current_path (list): The list of Pokémon names in the current evolutionary path.
        all_lines (list): The master list to store all complete evolutionary lines.
    """
    # Get the name of the current Pokémon in the chain
    pokemon_name = chain_node['species']['name']
    
    # Append the current Pokémon to the path
    new_path = current_path + [pokemon_name]

    # If there are no further evolutions, this path is a complete evolutionary line
    if not chain_node['evolves_to']:
        all_lines.append(new_path)
    else:
        # If there are evolutions, recursively call for each branch
        for evolution in chain_node['evolves_to']:
            extract_evolution_lines(evolution, new_path, all_lines)

def get_evolution_lines():
    """
    Fetches all evolution chains and processes them to generate a list of
    all unique evolutionary lines.
    """
    all_evolution_lines = []
    
    # Get all evolution chain URLs
    chain_urls = get_all_evolution_chain_urls()
    print(f"Found {len(chain_urls)} evolution chains. Fetching data...")

    for i, url in enumerate(chain_urls):
        try:
            response = requests.get(url)
            response.raise_for_status()
            chain_data = response.json()
            
            # Start the recursive extraction from the root of the chain
            extract_evolution_lines(chain_data['chain'], [], all_evolution_lines)
            
            # Optional: Print progress
            if (i + 1) % 50 == 0 or (i + 1) == len(chain_urls):
                print(f"Processed {i + 1}/{len(chain_urls)} chains.")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching evolution chain from {url}: {e}")
            continue # Continue to the next URL even if one fails

    # Remove duplicate lines (e.g., due to different paths leading to the same end)
    # Convert inner lists to tuples for set uniqueness, then back to lists
    unique_lines = sorted(list(set(tuple(line) for line in all_evolution_lines)))
    
    # Convert tuples back to lists for the final output
    return [list(line) for line in unique_lines]

if __name__ == "__main__":
    evolution_lines = get_evolution_lines()

    # Print the list in the desired Python format
    print("# A list of lists, where each inner list is an evolutionary line.")
    print("# All names should be lowercase and match the filenames (e.g., 'nidoran-m').")
    print("evolution_lines = [")
    for line in evolution_lines:
        # Format each inner list nicely
        formatted_line = ", ".join(f"'{pokemon}'" for pokemon in line)
        print(f"    [{formatted_line}],")
    print("]")
