
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, freqz 
from statsmodels.regression.linear_model import burg

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
                "fs": fs,
            })

train_trials = []
test_trials = []

# Split trials into training and testing sets
for participant in [1, 2]:
    trials = [t for t in all_trials if t["participant"] == participant]
    runs = sorted(set(t["run"] for t in trials))

    for trial in trials:
        if trial["run"] == runs[-1]:
            test_trials.append(trial)
        else:
            train_trials.append(trial)

#extract avg power band in 10.5-13.5 Hz for C3 and C4 channels
#output: [log(avg_power_C3), log(avg_power_C4)] for each trial
def extract_features(trial, band=(10.5, 13.5)):
    eeg = trial["eeg"]
    labels = trial["labels"]

    c3 = labels.index("C3")
    c4 = labels.index("C4")

    c3_neighbors = [labels.index(ch) for ch in ["FC3", "CP3", "C1", "C5"]]
    c4_neighbors = [labels.index(ch) for ch in ["FC4", "CP4", "C2", "C6"]]

    filtered = np.array([
        eeg[c3] - eeg[c3_neighbors].mean(axis=0),
        eeg[c4] - eeg[c4_neighbors].mean(axis=0)
    ])

    window_size = 160
    step_size = 40
    freqs = np.arange(0, 30.1, 0.2)
    powers = []

    for i in range(0, filtered.shape[1] - window_size + 1, step_size):
        window = filtered[:, i:i + window_size]
        psd = []

        for channel in window:
            ar, variance = burg(channel, order=8)

            if not np.isfinite(variance) or variance <= 0:
                break

            _, response = freqz([1], np.r_[1, -ar], worN=freqs, fs=trial["fs"])
            psd.append(variance * np.abs(response) ** 2)

        if len(psd) != 2:
            continue

        band_idx = (freqs >= band[0]) & (freqs <= band[1])
        powers.append(np.mean(np.array(psd)[:, band_idx], axis=1))

    if not powers:
        return np.array([np.nan, np.nan])

    return np.log(np.mean(powers, axis=0) + 1e-12)

x_train = np.array([extract_features(t) for t in train_trials])
x_test = np.array([extract_features(t) for t in test_trials])

y_train = np.array([t["target"] for t in train_trials])
y_test = np.array([t["target"] for t in test_trials])

valid = np.isfinite(x_train).all(axis=1)

x_train = x_train[valid]
y_train = y_train[valid]
train_trials = [t for t, keep in zip(train_trials, valid) if keep]

for participant in [1, 2]:
    train_idx = np.array([t["participant"] == participant for t in train_trials])
    test_idx = np.array([t["participant"] == participant for t in test_trials])

    model = LogisticRegression()
    model.fit(x_train[train_idx], y_train[train_idx])

    predictions = model.predict(x_test[test_idx])
    accuracy = accuracy_score(y_test[test_idx], predictions)




bands = [(8, 11), (11, 14), (14, 17), (17, 20), (20, 30)]

best_bands = {}

for participant in [1, 2]:
    trials = [t for t in train_trials if t["participant"] == participant]
    runs = sorted(set(t["run"] for t in trials))

    train = [t for t in trials if t["run"] != runs[-1]]
    val = [t for t in trials if t["run"] == runs[-1]]

    best_accuracy = -1

    for band in bands:
        x = np.array([extract_features(t, band) for t in train])
        y = np.array([t["target"] for t in train])

        x_val = np.array([extract_features(t, band) for t in val])
        y_val = np.array([t["target"] for t in val])

        model = LogisticRegression()
        model.fit(x, y)

        accuracy = accuracy_score(y_val, model.predict(x_val))
        print(f"Participant {participant}, {band} Hz: {accuracy:.2%}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_bands[participant] = band

    print(f"Best band for Participant {participant}: {best_bands[participant]}")



# Evaluate selected frequency bands
for participant in [1, 2]:
    train = [t for t in train_trials if t["participant"] == participant]
    test = [t for t in test_trials if t["participant"] == participant]

    band = best_bands[participant]

    x = np.array([extract_features(t, band) for t in train])
    y = np.array([t["target"] for t in train])

    x_eval = np.array([extract_features(t, band) for t in test])
    y_eval = np.array([t["target"] for t in test])

    model = LogisticRegression()
    model.fit(x, y)

    predictions = model.predict(x_eval)
    accuracy = accuracy_score(y_eval, predictions)

    print(f"Participant {participant}, {band} Hz: {accuracy:.2%}")