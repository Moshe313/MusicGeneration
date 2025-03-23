# Generate Music Using Discrete Diffusion

## Creators

Omer Aviv, Moshe Buchris, Rom Hirsch, Omer Pilosof

## Pepar
[SEDD_Music](SEDD_Music.pdf)

## Design Choices

This codebase is built modularly to promote future research (as opposed to a more compact framework, which would be better for applications). The primary files are 

1. ```noise_lib.py```: the noise schedule
2. ```graph_lib```: the forward diffusion process
3. ```sampling.py```: the sampling strategies
4. ```model/```: the model architecture

## Installation

Simply run

```
conda env create -f environment.yml
```

which will create a ```sedd``` environment with packages installed. Note that this installs with CUDA 11.8, and different CUDA versions must be installed manually. The biggest factor is making sure that the ```torch``` and ```flash-attn``` packages use the same CUDA version (more found [here](https://github.com/Dao-AILab/flash-attention)).


### Run Sampling

We can run sampling using a the script 

```
python run_sample.py

```
edit the following function: 
destination_path - where save the samples 

model_path - where the weights are saved

steps_time - the number of steps in the diffusion process

number_samples - the number of samples to generate

start_save - the starting index for the samples it affect the sample file name

```
gen_sample(destination_path, steps_time,
                   model_path, number_samples, start_save=0)
```

## Training New Models
### Aduio To Token 
Download the model [wavtokenizer_medium_music_audio_320_24k](https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/blob/main/wavtokenizer_medium_music_audio_320_24k.ckpt) and store it under "models" folder.

To use WavTokenizer, Enter the Tokenizer folder and install it using:

```
conda create -n wavtokenizer python=3.9
conda activate wavtokenizer
pip install -r requirements.txt
```

To encode audio files in directory enter the "EncodeAll.py" file, and change the folder_file and save_path variables:
```
folder_path = r"C:\path-to-your-audio-files-folder"
save_path = r"C:\path-to-the-results-tokens-folder\token-"
```

### Run Training

We provide training code, which can be run with the command

```
python run_train.py
```
This creates a new directory `direc=exp_local/DATE/TIME` with the following structure (compatible with running sampling experiments locally)
```
├── direc
│   ├── .hydra
│   │   ├── config.yaml
│   │   ├── ...
│   ├── checkpoints
│   │   ├── checkpoint_*.pth
│   ├── checkpoints-meta
│   │   ├── checkpoint.pth
│   ├── samples
│   │   ├── iter_*
│   │   │   ├── sample_*.txt
│   ├── logs
```
Here, `checkpoints-meta` is used for reloading the run following interruptions, `samples` contains generated images as the run progresses, and `logs` contains the run output. Arguments can be added with `ARG_NAME=ARG_VALUE`, with important ones being:
```
ngpus                     the number of gpus to use in training (using pytorch DDP)
training.accum            number of accumulation steps, set to 1 for small and 2 for medium (assuming an 8x80GB node)
noise.type                one of geometric, loglinear 
graph.type                one of uniform, absorb
model                     one of small, medium
model.scale_by_sigma      set to False if graph.type=uniform (not yet configured)
```
Some example commands include
```
# training hyperparameters for SEDD absorb
python train.py noise_lib=loglinear graph.type=absorb model=medium training.accum=2
# training hyperparameters for SEDD uniform
python train.py noise_lib=geometric graph.type=uniform model=small model.scale_by_sigma=False
```

## Metrics and Evaluation
### FAD
Installation:
```
pip install frechet_audio_distance
```

To get the frechet audio distance (FAD) of both VGGish and PANN districution, enter the "run.py" file:
```
path_to_background_set = r"C:\path-to-your-BACKGROUND-audio-files-folder"
path_to_eval_set = r"C:\path-to-your-GENERATED-audio-files-folder"
```
Save the distributions to save time at the next comparing:
```
background_embds_path = r"C:\path-to-save-the-BACKGROUND-files-distribution\Background_embeddings_"
eval_embds_path = r"C:\path-to-save-the-GENERATED-files-distribution\Generated_embeddings_"
```
The result will printed and saved.
## Other Features

### SLURM compatibility

To train on slurm, simply run 
```
python train.py -m args
```
## change dataset 

we create new function for load the dataset of music call MusicDataset.py 
for changing to another dataset you need to change hardcoded path in the function 
default path is ./OurWavTokenizer/Tokens



## Citation

@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```
## Acknowledgements

This repository builds heavily off of [score sde](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).
