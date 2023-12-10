import torch
import pytorch_lightning as pl


class System(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        weight_decay: float = 0.001,
        lr: float = 0.001,
        min_bpm: float = 24,
        max_bpm: float = 320,
    ):
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

    def common_step(self, batch, mode="train"):
        """Common step for both training and validation"""
        audio, bpm = batch

        # normalize the bpm between 0 and 1
        nom_bpm = (bpm - self.hparams.min_bpm) / (
            self.hparams.max_bpm - self.hparams.min_bpm
        )

        # forward pass
        pred_nom_bpm = torch.sigmoid(self.model(audio))

        # compute loss
        loss = torch.nn.functional.mse_loss(pred_nom_bpm, nom_bpm)

        # log metrics
        self.log(
            f"{mode}_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

        # denormalize the predicted bpm
        pred_bpm = (
            pred_nom_bpm * (self.hparams.max_bpm - self.hparams.min_bpm)
            + self.hparams.min_bpm
        )

        return loss, pred_bpm, bpm

    def training_step(self, batch, batch_idx):
        loss, pred_bpm, bpm = self.common_step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, red_bpm, bpm = self.common_step(batch, mode="val")
        return red_bpm, bpm
