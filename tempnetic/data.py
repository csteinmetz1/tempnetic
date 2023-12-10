import os
import glob
import torch
import torchaudio
import pytorch_lightning as pl

from tqdm import tqdm
from typing import List


class AudioFileDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[dict]):
        super().__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        audio = example["audio"]
        bpm = example["bpm"]

        # convert stereo to mono
        if audio.shape[0] == 2:
            audio = torch.mean(audio, dim=0, keepdim=True)

        return audio, bpm


class AudioFileDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        length: int = 262144,
        batch_size: int = 32,
        num_workers: int = 8,
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
        for audio_filepath in tqdm(audio_filepaths):
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

            # get the beats
            # beats_filepath = os.path.join(root_dir, "beats", f"{song_id}.beats")

            # get the audio
            try:
                audio, sr = torchaudio.load(audio_filepath)
            except:
                print(f"Skipping {song_id} as audio cannot be loaded.")
                continue

            # split into non-overlapping chunks
            num_chunks = audio.shape[1] // self.hparams.length

            for chunk_idx in range(num_chunks):
                start = chunk_idx * self.hparams.length
                end = (chunk_idx + 1) * self.hparams.length
                chunk = audio[:, start:end]
                examples.append(
                    {
                        "song_id": song_id,
                        "audio_filepath": audio_filepath,
                        "bpm": bpm,
                        "audio": chunk,
                        "genre": genre,
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

        self.train_dataset = AudioFileDataset(train_examples)
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
