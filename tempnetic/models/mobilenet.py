import torch
import torchvision
import torchlibrosa


class SpectrogramMobileNetV2(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        sample_rate: int,
        window_length: int = 1024,
        hop_length: int = 512,
        mel_bins: int = 64,
        fmin: int = 20,
        fmax: int = 11025,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model = torchvision.models.mobilenet_v2(weights="DEFAULT")
        self.model.classifier = torch.nn.Linear(1280, embed_dim)

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spec_extractor = torchlibrosa.stft.Spectrogram(
            n_fft=window_length,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.logmel_extractor = torchlibrosa.stft.LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_length,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.spec_augmenter = torchlibrosa.augmentation.SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=64,
            freq_stripes_num=2,
        )

    def forward(self, x: torch.Tensor, tempogram: torch.Tensor = None):
        if tempogram is None:
            bs, chs, seq_len = x.shape
            assert chs == 1  # must be mono
            # remove channel dimension
            x = x.squeeze(1)
            x = self.spec_extractor(x)
            x = self.logmel_extractor(x)
        else:
            bs, chs, bpms, frames = tempogram.shape
            x = tempogram

        # augment spectrogram
        if self.training:
            x = self.spec_augmenter(x)

        # repeat channel dimension
        x = x.repeat(1, 3, 1, 1)

        # forward pass through model
        x = self.model(x)
        return x
