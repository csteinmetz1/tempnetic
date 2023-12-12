import os
import glob
import torch
import random
import librosa
import pedalboard
import torchaudio
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from typing import List


class AudioFileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: List[dict],
        num_passes: int = 1,
        time_stretch: bool = False,
        level_randomization: bool = False,
        use_tempogram: bool = False,
    ):
        super().__init__()
        self.examples = examples
        self.num_passes = num_passes
        self.time_stretch = time_stretch
        self.level_randomization = level_randomization
        self.use_tempogram = use_tempogram

    def __len__(self):
        return len(self.examples) * self.num_passes

    def __getitem__(self, idx):
        # since the dataset is small we can pass through it multiple times
        idx = idx % len(self.examples)
        example = self.examples[idx]

        audio = example["audio"]
        bpm = example["bpm"]
        sample_rate = example["sample_rate"]
        tempogram = example["tempogram"]

        length = audio.shape[1]

        # time stretch
        if self.time_stretch:
            stretch_factor = np.random.uniform(0.7, 1.3)
            transient_mode = np.random.choice(["crisp", "smooth", "mixed"])
            audio_out = torch.from_numpy(
                pedalboard.time_stretch(
                    audio_out.numpy(),
                    sample_rate,
                    stretch_factor=stretch_factor,
                    transient_mode=transient_mode,
                    high_quality=False,
                )
            )
            bpm = bpm * stretch_factor

            # if shorter than the original length, repeat pad
            if audio_out.shape[1] < length:
                num_repeats = length // audio_out.shape[1] + 1
                audio_out = audio_out.repeat(1, num_repeats)
                audio_out = audio_out[:, :length]
            elif audio_out.shape[1] > length:
                audio_out = audio_out[:, :length]

        # level randomization
        if self.level_randomization:
            gain_db = np.random.uniform(-12, 0.0)
            gain_lin = 10 ** (gain_db / 20)
            audio_out *= gain_lin

        return audio, tempogram, bpm


class AudioFileDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        length: int = 262144,
        batch_size: int = 32,
        num_workers: int = 8,
        num_passes: int = 50,
        min_bpm: float = 40,
        max_bpm: float = 250,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        # find all the audio files in the root directory
        audio_filepaths = glob.glob(
            os.path.join(self.hparams.root_dir, "audio", "**", "*.wav"),
            recursive=True,
        )

        # since the dataset is small, we can load all the files in memory
        genres = set()
        examples = []
        print("Loading dataset...")
        for audio_idx, audio_filepath in enumerate(tqdm(audio_filepaths)):
            # get the song ID
            song_id = os.path.basename(audio_filepath).replace(".wav", "")
            genre = os.path.basename(os.path.dirname(audio_filepath))

            if genre not in genres:
                genres.add(genre)

            # get the bpm file
            bpm_filepath = os.path.join(
                self.hparams.root_dir,
                "tempo",
                f"""gtzan_{song_id.replace(".", "_")}.bpm""",
            )

            # check if bpm file exist
            if not os.path.exists(bpm_filepath):
                print(f"Skipping {song_id} as no BPM file exists.")
                continue

            with open(bpm_filepath, "r") as f:
                bpm = float(f.read())
                bpm = torch.tensor([bpm]).float()

            # check if the bpm is within the specified range
            if bpm < self.hparams.min_bpm or bpm > self.hparams.max_bpm:
                print(f"Skipping {song_id} as BPM ({bpm}) is outside range.")
                continue

            # get the audio
            try:
                audio, sr = torchaudio.load(audio_filepath)
            except:
                print(f"Skipping {song_id} as audio cannot be loaded.")
                continue

            # convert stereo to mono
            if audio.shape[0] == 2:
                audio = torch.mean(audio, dim=0, keepdim=True)

            audio = audio / torch.max(torch.abs(audio))  # peak normalization

            # split into non-overlapping chunks
            num_chunks = audio.shape[1] // self.hparams.length

            for chunk_idx in range(num_chunks):
                start = chunk_idx * self.hparams.length
                end = start + self.hparams.length
                chunk = audio[:, start:end]

                # compute tempogram of chunk
                oenv = librosa.onset.onset_strength(
                    y=chunk.numpy(),
                    sr=sr,
                    hop_length=512,
                )
                tempogram = librosa.feature.tempogram(
                    onset_envelope=oenv,
                    sr=sr,
                    hop_length=512,
                )
                tempogram = torch.from_numpy(tempogram).float()

                examples.append(
                    {
                        "song_id": song_id,
                        "audio_filepath": audio_filepath,
                        "bpm": bpm,
                        "audio": chunk,
                        "tempogram": tempogram,
                        "genre": genre,
                        "sample_rate": sr,
                    }
                )

        if len(examples) == 0:
            raise Exception("No examples loaded.")

        print(f"Loaded {len(examples)} examples.")

        # split into train and validation
        # ensure that each genre is represented in both train and validation
        # and that there is no overlap between songs in train and validation
        train_examples = []
        val_examples = []

        for genre in genres:
            genre_examples = [e for e in examples if e["genre"] == genre]
            random.shuffle(genre_examples)
            genre_train_examples = genre_examples[: int(len(genre_examples) * 0.8)]
            genre_val_examples = genre_examples[int(len(genre_examples) * 0.8) :]

            train_examples.extend(genre_train_examples)
            val_examples.extend(genre_val_examples)

            print(
                "genre:",
                genre,
                "train:",
                len(genre_train_examples),
                "val:",
                len(genre_val_examples),
            )

        self.train_dataset = AudioFileDataset(train_examples, self.hparams.num_passes)
        self.val_dataset = AudioFileDataset(val_examples)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
        )
