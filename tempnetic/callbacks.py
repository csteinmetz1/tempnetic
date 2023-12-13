import os
import shutil
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT


class MoveConfigCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        full_run_dir = trainer.logger.experiment.dir
        run_id = full_run_dir.split(os.sep)[-2].split("-")[-1]
        src_dir = os.path.join(trainer.log_dir, "config.yaml")
        dest_dir = os.path.join(
            trainer.log_dir, "tempnetic", run_id, "checkpoints", "config.yaml"
        )

        run_dir = os.path.join(trainer.log_dir, "tempnetic", run_id)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        shutil.copyfile(src_dir, dest_dir)


def plot_xy(
    pred_bpms,
    bpms,
    min_bpm: float,
    max_bpm: float,
    axs=None,
    title: str = "XY Plot",
):
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    # plot scatter
    axs.scatter(pred_bpms, bpms, color="tab:blue", s=4)

    # plot straight line
    x = [min_bpm, max_bpm]
    y = [min_bpm, max_bpm]
    plt.plot(x, y, color="gray", linewidth=2, linestyle="--")
    # add line for + 8% error and - 8% error
    # x = [min_bpm * 1.08, max_bpm * 1.08]
    # y = [min_bpm * 1.08, max_bpm * 1.08]
    # plt.plot(x, y, color="lightgray", linewidth=2, linestyle="-")
    # x = [min_bpm * 0.92, max_bpm * 0.92]
    # y = [min_bpm * 0.92, max_bpm * 0.92]
    # plt.plot(x, y, color="lightgray", linewidth=2, linestyle="--")

    plt.grid(c="lightgray")
    axs.set_title(f"{title}")
    axs.set_xlabel("Predicted BPM")
    axs.set_ylabel("True BPM")
    plt.tight_layout()


class XYPlotCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.pred_bpms = []
        self.bpms = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pred_bpm, bpm = outputs
        pred_bpm = pred_bpm.detach().cpu().numpy()
        bpm = bpm.detach().cpu().numpy()

        # iterate over batch elements
        for i in range(len(pred_bpm)):
            self.pred_bpms.append(pred_bpm[i])
            self.bpms.append(bpm[i])

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        plot_xy(
            self.pred_bpms,
            self.bpms,
            pl_module.hparams.min_bpm,
            pl_module.hparams.max_bpm,
            axs=axs,
        )

        # log confusion matrix to wandb
        trainer.logger.experiment.log({"xy-valid": plt})
        plt.close("all")

        # reset outputs
        self.pred_bpms = []
        self.bpms = []
