import os
import shutil
import pytorch_lightning as pl


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
