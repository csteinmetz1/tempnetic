import os
import glob
import matplotlib.pyplot as plt

if not os.path.exists("outputs"):
    os.mkdir("outputs")

# first let's load of the BPM files
bpm_filepaths = glob.glob(
    "/import/c4dm-datasets-ext/tempnetic-dataset/GTZAN/tempo/*.bpm"
)
bpm_filepaths = sorted(bpm_filepaths)

dataset = {}

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
plt.savefig("outputs/bpm_histogram.png", dpi=300)
