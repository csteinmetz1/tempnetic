import torch
import argparse
import torchaudio
import pytorch_lightning as pl

from tempnetic.system import System
from tempnetic.callbacks import plot_xy
from tempnetic.data import AudioFileDataModule
from tempnetic.models.mobilenet import SpectrogramMobileNetV2

CKPT_PATH = (
    "/import/c4dm-datasets-ext/tempnetic-logs/tempnetic/ayduycjk/checkpoints/last.ckpt"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to input audio file")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    args = parser.parse_args()

    # load model
    model = SpectrogramMobileNetV2(1, 22050)
    system = System.load_from_checkpoint(args.ckpt_path, model=model)
    system.cpu()
    system.eval()

    # load audio
    audio, sr = torchaudio.load(args.input)

    # estimate tempo
    with torch.no_grad():
        pred_tempo = system.estimate_tempo(audio, sr)

    print(f"Estimated tempo: {pred_tempo.item():.2f} BPM")
