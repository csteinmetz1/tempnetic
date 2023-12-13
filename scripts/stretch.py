import os
import glob
import torch
import pedalboard
import torchaudio

from tqdm import tqdm
from multiprocessing import Pool

# create an extended dataset by time stretching the audio files


def stretch(audio_filepath, root_dir, audio_output_dir, tempo_output_dir):
    song_id = os.path.basename(audio_filepath).replace(".wav", "")
    genre = os.path.basename(os.path.dirname(audio_filepath))
    os.makedirs(os.path.join(audio_output_dir, genre), exist_ok=True)

    try:
        audio, sr = torchaudio.load(audio_filepath)
    except:
        print(f"Skipping {audio_filepath} as audio cannot be loaded.")
        return

    # get the bpm file
    bpm_filepath = os.path.join(
        root_dir,
        "tempo",
        f"""gtzan_{song_id.replace(".", "_")}.bpm""",
    )

    # check if bpm file exist
    if not os.path.exists(bpm_filepath):
        print(f"Skipping {song_id} as no BPM file exists.")
        return

    with open(bpm_filepath, "r") as f:
        bpm = float(f.read())
        bpm = torch.tensor([bpm]).float()

    stretch_factors = [0.7, 0.8, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    for idx, stretch_factor in enumerate(stretch_factors):
        print(song_id, stretch_factor)

        audio_out = torch.from_numpy(
            pedalboard.time_stretch(
                audio.numpy(),
                sr,
                stretch_factor=stretch_factor,
                high_quality=True,
            )
        )
        stretched_bpm = bpm * stretch_factor

        output_bpm_filepath = os.path.join(
            tempo_output_dir,
            f"""gtzan_{song_id.replace(".", "_")}_{idx}.bpm""",
        )

        with open(output_bpm_filepath, "w") as f:
            f.write(str(stretched_bpm.item()))

        output_audio_filepath = os.path.join(
            audio_output_dir,
            genre,
            f"""{song_id}_{idx}.wav""",
        )
        torchaudio.save(output_audio_filepath, audio_out, sr)


if __name__ == "__main__":
    root_dir = "/import/c4dm-datasets-ext/tempnetic-dataset/GTZAN"
    output_dir = "/import/c4dm-datasets-ext/tempnetic-dataset/GTZAN-extended"
    os.makedirs(output_dir, exist_ok=True)

    audio_output_dir = os.path.join(output_dir, "audio")
    tempo_output_dir = os.path.join(output_dir, "tempo")
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(tempo_output_dir, exist_ok=True)

    # find all the audio files in the root directory
    audio_filepaths = glob.glob(
        os.path.join(root_dir, "audio", "**", "*.wav"),
        recursive=True,
    )

    # multiprocessing for speed
    with Pool(32) as p:
        p.starmap(
            stretch,
            [
                (audio_filepath, root_dir, audio_output_dir, tempo_output_dir)
                for audio_filepath in audio_filepaths
            ],
        )
