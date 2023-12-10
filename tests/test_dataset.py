import torch

from tempnetic.data import AudioFileDataModule

if __name__ == "__main__":
    root_dir = "/import/c4dm-datasets-ext/tempnetic-dataset/GTZAN"

    dm = AudioFileDataModule(root_dir)
    dm.setup()

    train_dataset = dm.train_dataset

    for bidx, batch in enumerate(train_dataset):
        audio, bpm = batch
        print(audio.shape, bpm.shape)
