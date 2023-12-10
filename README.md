# tempnetic
Tempo estimation

## Installation

Install the package in editable mode:
```
python -m venv env
source env/bin/activate
pip install jsonargparse[signatures]>=4.18.0
pip install -e .
```

## Inference

```
```

## Training
```
CUDA_VISIBLE_DEVICES=4 python scripts/main.py fit \
-c cfg/trainer.yaml \
-c cfg/model.yaml \
-c cfg/data.yaml \
```


## Overview

Brainstorming: 
- What if the tempo changes during the song?
- Should we estimate beats first and use beats to compute tempo?
- If we estimate the tempo directly, should we treat as classification or regression problem?
- Should we add data augmentation due to small dataset size? 
    - We need to do some data exploration to see if the dataset is balanced or not.
    - What is the distribution of tempos in the dataset?
    - What is the min and max tempo in the dataset?
- If we use data augmentation, what kind of augmentation should we use?
- Are there any non-neural network methods that we can use as a baseline?
    - Looks like librosa has a method based on onset detection.
- We will want a train/test split? 
- What sample rate? 

Model design:
- Input: spectrogram
- Output: tempo? beats? 
- Types of models
    - 2d CNN
    - LSTM
    - Transformer
    - Mamba (SSM)
- Can we use features from a pre-trained model?

Considerations:
- Performance (how do we measure this?)
- Generalization (does it work on real songs?)
- Efficiency (can we run it on CPU?)
- Causality and real-time operation


## Resources

- [Tempo Estimation @ MIREX](https://www.music-ir.org/mirex/wiki/2014:Audio_Tempo_Estimation)