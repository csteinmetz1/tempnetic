import torch
import librosa
import torchaudio
import numpy as np
import pytorch_lightning as pl


class System(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        weight_decay: float = 0.001,
        lr: float = 0.001,
        min_bpm: float = 40,
        max_bpm: float = 250,
        use_tempogram: bool = False,
        sample_rate: int = 22050,
    ):
        # use min and max bpm from madmom if not specified
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def forward(self, x, tempogram=None):
        return torch.sigmoid(self.model(x, tempogram=tempogram))

    def estimate_tempo(self, x: torch.Tensor, sr: int):
        # convert sample rate if necessary
        if sr != self.hparams.sample_rate:
            x = torchaudio.functional.resample(x, sr, self.hparams.sample_rate)

        # convert stereo to mono
        if x.shape[0] == 2:
            x = torch.mean(x, dim=0, keepdim=True)

        # normalize
        x = x / torch.max(torch.abs(x))

        if self.hparams.use_tempogram:
            oenv = librosa.onset.onset_strength(
                y=x.numpy(),
                sr=self.hparams.sample_rate,
                hop_length=512,
            )
            tempogram = librosa.feature.tempogram(
                onset_envelope=oenv,
                sr=self.hparams.sample_rate,
                hop_length=512,
            )
            tempogram = torch.from_numpy(tempogram).float()
            tempogram = tempogram.unsqueeze(0)
        else:
            tempogram = None

        # pass through model
        with torch.no_grad():
            pred_bpm = self.forward(x.unsqueeze(0), tempogram=tempogram)

        # rescale the predicted bpm
        pred_bpm = (
            pred_bpm * (self.hparams.max_bpm - self.hparams.min_bpm)
            + self.hparams.min_bpm
        )

        return pred_bpm

    def common_step(self, batch, mode="train"):
        """Common step for both training and validation"""
        audio, tempogram, bpm = batch

        # scale bpm to [0, 1]
        bpm_scaled = (bpm - self.hparams.min_bpm) / (
            self.hparams.max_bpm - self.hparams.min_bpm
        )

        # forward pass
        if self.hparams.use_tempogram:
            pred_bpm_scaled = self.forward(audio, tempogram=tempogram)
        else:
            pred_bpm_scaled = self.forward(audio)

        # compute loss
        loss = torch.nn.functional.mse_loss(pred_bpm_scaled, bpm_scaled)

        # rescale the predicted bpm
        pred_bpm = (
            pred_bpm_scaled * (self.hparams.max_bpm - self.hparams.min_bpm)
            + self.hparams.min_bpm
        )

        # compute accuracy
        # ensure predicted bpm is within 8% of the ground truth
        error = torch.abs(pred_bpm - bpm)
        accuracy = torch.mean((error <= 0.08 * bpm).float())

        # log metrics
        self.log(
            f"{mode}_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            f"{mode}_accuracy",
            accuracy,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

        return loss, pred_bpm, bpm

    def training_step(self, batch, batch_idx):
        loss, pred_bpm, bpm = self.common_step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_bpm, bpm = self.common_step(batch, mode="val")
        return pred_bpm, bpm
