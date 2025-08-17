# Pokémon Evolution Predictor

An attempt to use ML to synthesize new evolutions and baby Pokémon by identifying existing patterns in evolution data.

This work was inspired by [Tom7](http://tom7.org/)'s [Uppestcase and Lowestcase Letters [advances in derp learning]](https://youtu.be/HLRdruqQfRk).

In practice, the generated evolutions can sometimes have compelling portraits, but the actual color data is nearly always a mess.

### Example input-output pairs:

![](examples/evolve1.png)
![](examples/evolve2.png)

## Base dependencies:
```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
pip3 install diffusers transformers accelerate
pip3 install datasets pillow scipy tqdm matplotlib
```

## Download images:
```bash
pip install requests beautifulsoup4
python3 sprite_downloader.py
```
This dataset is currently 263 MB and takes about 70 minutes to download on a reasonably fast Internet connection.
All sprites from all generations of games are downloaded, but the script can be modified if you only want sprites from specific games.

## Download evolutionary lines:

Normally not needed, as they are already provided in [evolution_lines.py](evolution_lines.py).
Use this only if new Pokémon are released, to update [evolution_lines.py](evolution_lines.py).
Output is printed to screen and must be carefully transferred over. 
It does not simply override the file because it may be desirable to add custom evolutionary line data, such as for Mega Evolutions and Gigantamax forms.
```bash
pip install requests
python3 make_evolution_lines.py
```

## Train:

Make sure the desired sprites have been downloaded, set up your configuration in [config.yml](config.yml), then run:
```bash
python3 train_gan.py
```

NOTE: [`train.py`](train.py) and [`dual_pix2pix_pipeline.py`](models/dual_pix2pix_pipeline.py) is old code for pix2pix and no longer works.

## Generate:

```bash
python3 predict.py --input <path to input image> --output <path to output image> --direction forward
# Or
python3 predict.py --input <path to input image> --output <path to output image> --direction backward
```

## Alternatives

[Poke2Poke: Generating Pokemon Evolutions](https://medium.com/lcc-unison/poke2poke-generating-pokemon-evolutions-370522583391)