import torch
import librosa
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from tqdm import tqdm

from tempnetic.system import System
from tempnetic.callbacks import plot_xy
from tempnetic.data import AudioFileDataModule
from tempnetic.models.mobilenet import SpectrogramMobileNetV2


def estimate_tempo(x: torch.Tensor, sr: int):
    onset_env = librosa.onset.onset_strength(y=x.numpy(), sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    return tempo


if __name__ == "__main__":
    # in the evaluation script we will load our pretrained model
    # and run inference on audio files from the validiation set
    # in addition, we will consider a baseline tempo estimation model from librosa.

    # 48r709u0

    pl.seed_everything(42)  # make sure we use the same seed as in training

    # load the pretrained model ayduycjk
    ckpt_path = "/import/c4dm-datasets-ext/tempnetic-logs/tempnetic/ayduycjk/checkpoints/last.ckpt"
    model = SpectrogramMobileNetV2(1, 22050)
    system = System.load_from_checkpoint(ckpt_path, model=model)
    system.cpu()
    system.eval()

    # load the validation dataset
    root_dir = "/import/c4dm-datasets-ext/tempnetic-dataset/GTZAN-extended"

    dm = AudioFileDataModule(root_dir, batch_size=1, num_passes=1, num_workers=1)
    dm.setup()

    metrics = {"librosa": [], "tempnetic": []}
    predictions = {"librosa": [], "tempnetic": []}
    bpms = []
    val_dataset = dm.val_dataset

    pbar = tqdm(val_dataset)
    num_exmaples = len(pbar)

    for bidx, batch in enumerate(pbar):
        audio, tempogram, bpm = batch

        bpms.append(bpm)

        # estimate tempo with tempnetic
        with torch.no_grad():
            pred_tempo = system.estimate_tempo(audio, 22050)

        predictions["tempnetic"].append(pred_tempo)

        # compute accuracy within +/- 8% of ground truth
        accuracy = torch.abs(pred_tempo - bpm) <= 0.08 * bpm
        metrics["tempnetic"].append(accuracy)

        # estimate tempo with librosa
        pred_tempo = torch.from_numpy(estimate_tempo(audio, 22050))
        predictions["librosa"].append(pred_tempo)

        # compute accuracy within +/- 8% of ground truth
        accuracy = torch.abs(pred_tempo - bpm) <= 0.08 * bpm
        metrics["librosa"].append(accuracy)

        # print mean accuracy
        pbar.set_description(
            f"librosa: {torch.stack(metrics['librosa']).float().mean():0.4f} tempnetic: {torch.stack(metrics['tempnetic']).float().mean():0.4f}"
        )

        if bidx > num_exmaples:
            break

    # plot the results
    for model, pred_bpms in predictions.items():
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))
        plot_xy(pred_bpms, bpms, 50, 250, axs=axs, title=model)
        plt.savefig(f"outputs/{model}_xy.png", dpi=300)
        plt.close()
