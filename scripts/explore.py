import os
import glob
import matplotlib.pyplot as plt

if not os.path.exists("outputs"):
    os.mkdir("outputs")


def plot_bpm_histogram(bpm_filepaths, name: str = "bpm_histogram"):
    dataset = {}
    bpm_filepaths = sorted(bpm_filepaths)
    for bpm_filepath in bpm_filepaths:
        song_id = os.path.basename(bpm_filepath).replace(".bpm", "")
        with open(bpm_filepath, "r") as f:
            bpm = float(f.read())
            dataset[song_id] = {"bpm": bpm}

    # create a histogram of the BPMs
    plt.hist([v["bpm"] for v in dataset.values()], bins=25, width=10, zorder=3)
    plt.xlabel("BPM")
    plt.ylabel("Count")
    plt.title("Histogram of BPMs")
    plt.grid(c="lightgray", zorder=1)
    plt.tight_layout()
    plt.savefig(f"outputs/{name}.png", dpi=300)


if __name__ == "__main__":
    # first let's load of the BPM files
    bpm_filepaths = glob.glob(
        "/import/c4dm-datasets-ext/tempnetic-dataset/GTZAN/tempo/*.bpm"
    )
    plot_bpm_histogram(bpm_filepaths, name="bpm_histogram")

    extended_bpm_filepaths = glob.glob(
        "/import/c4dm-datasets-ext/tempnetic-dataset/GTZAN-extended/tempo/*.bpm"
    )
    plot_bpm_histogram(extended_bpm_filepaths, name="bpm_histogram_extended")
