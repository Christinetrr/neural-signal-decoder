
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

all_trials = []

# Load data
for participant, session in enumerate(["Session 1", "Session 2"], start=1):
    for path in sorted(Path("data", session).glob("*LR*.mat")):
        run_data = loadmat(path)["runData"][0, 0]
        labels = [str(label.item()) for label in run_data["label"].flatten()]
        fs = run_data["fs"].item()

        #PREPROCESSING
        # Select EEG channels
        excluded_channels = ["VEO", "HEO", "EKG", "EMG"]
        eeg_indices = [i for i, ch in enumerate(labels) if ch not in excluded_channels]

        # Bandpass filter (4–40 Hz)
        sos = butter(4, [4, 40], btype="bandpass", fs=fs, output="sos")

        for trial, target, outcome in zip(
            run_data["trials"].flatten(),
            run_data["target"].flatten(),
            run_data["outcome"].flatten(),
        ):
            eeg = trial[eeg_indices, :].copy()
            all_trials.append({
                "participant": participant,
                "run": path.stem,
                "eeg": sosfiltfilt(sos, eeg, axis=1),
                "target": int(target),
                "outcome": int(outcome),
                "labels": [labels[i] for i in eeg_indices],
            })

print("Total LR trials:", len(all_trials))
