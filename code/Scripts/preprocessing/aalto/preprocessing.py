import csv
import gc
import glob
import json
import os
import warnings

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress pandas ParserWarnings for cleaner output
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)


def create_sliding_windows(session_features, user_label, window_size, stride):
    """
    Create sliding windows from session features.
    """
    windows = []
    labels = []

    if len(session_features) < window_size:
        return windows, labels

    scaler = StandardScaler()
    if np.std(session_features, axis=0).any() == 0:
        normalised_features = session_features - np.mean(session_features, axis=0)
    else:
        normalised_features = scaler.fit_transform(session_features)

    for i in range(0, len(normalised_features) - window_size + 1, stride):
        windows.append(normalised_features[i : i + window_size])
        labels.append(user_label)

    return windows, labels


def process_and_save(input_dir, output_file, window_size, stride, split_ratio=0.8):
    """
    Main function to process and save preprocessed data.

    Args:
        input_dir: Directory containing raw keystroke data files
        output_file: Base path for output files (will be suffixed with :train and :test)
        window_size: Size of sliding window for feature extraction
        stride: Step size for sliding window
        split_ratio: Ratio of data to use for training (default: 0.8)
    """
    print("==== Preprocessing ====")
    print(f"Window Size: {window_size}")
    print(f"Stride: {stride}")
    print(f"Train/Test Split: {split_ratio * 100:.0f}:{(1 - split_ratio) * 100:.0f}")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

    if window_size <= 1:
        print("Window size must be greater than 1.")
        return

    raw_files = glob.glob(os.path.join(input_dir, "*_keystrokes.txt"))
    if not raw_files:
        print("No input files found.")
        return

    all_participants = set()
    for raw_file in raw_files:
        try:
            df = pd.read_csv(
                raw_file,
                sep="\t",
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
                encoding="latin-1",
                on_bad_lines="warn",
            )
            if "PARTICIPANT_ID" in df.columns:
                all_participants.update(df["PARTICIPANT_ID"].unique())
        except Exception as e:
            print(f"Error reading {raw_file}: {e}")

    if not all_participants:
        print("No participants found.")
        return

    user_map = {str(user_id): i for i, user_id in enumerate(sorted(all_participants))}
    print(f"Found {len(user_map)} users.")

    all_windows = []
    all_labels = []
    success_count = 0
    skip_count = 0

    for i, raw_file in enumerate(raw_files):
        filename = os.path.basename(raw_file)
        print(f"[{i + 1:6d}/{len(raw_files):6d}] processing {filename}...", flush=True)
        if (i + 1) % 5000 == 0:
            print(f"  -> Processed {i + 1} files so far...")
        try:
            df = pd.read_csv(
                raw_file,
                sep="\t",
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
                encoding="latin-1",
                on_bad_lines="error",
            )
            if not all(
                col in df.columns
                for col in [
                    "PARTICIPANT_ID",
                    "TEST_SECTION_ID",
                    "PRESS_TIME",
                    "RELEASE_TIME",
                ]
            ):
                print(f"Missing columns in {raw_file}")
                continue

            participant_id = str(df["PARTICIPANT_ID"].iloc[0])
            user_label = user_map[participant_id]

            for _, session_group in df.groupby(["TEST_SECTION_ID"]):
                session_group = session_group.sort_values(by=["PRESS_TIME"])

                session_group["HOLD_TIME"] = (
                    session_group["RELEASE_TIME"] - session_group["PRESS_TIME"]
                )
                session_group["NEXT_PRESS_TIME"] = session_group["PRESS_TIME"].shift(-1)
                session_group["FLIGHT_TIME"] = (
                    session_group["NEXT_PRESS_TIME"] - session_group["RELEASE_TIME"]
                )

                session_features = session_group[["HOLD_TIME", "FLIGHT_TIME"]].values

                windows, labels = create_sliding_windows(
                    session_features, user_label, window_size, stride
                )
                all_windows.extend(windows)
                all_labels.extend(labels)
                success_count += 1
        except Exception as e:
            print(f"  SKIPPED: {os.path.basename(raw_file)} - {e}")
            skip_count += 1
            continue
        del df
        gc.collect()

    if not all_windows:
        print("no Windows could be created")
        return

    x = np.array(all_windows)
    y = np.array(all_labels)

    print(f"Final data shape: x={x.shape}, y={y.shape}")

    # Split into train/test using stratified split
    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=1 - split_ratio, stratify=y, random_state=42
    )
    print(f"Train: {x_train.shape[0]} samples, Test: {x_test.shape[0]} samples")

    # Save train data
    train_file = f"{output_file}:train"
    print(f"Saving train data to {train_file}")
    with h5py.File(train_file, "w") as f:
        f.create_dataset("x", data=x_train)
        f.create_dataset("y", data=y_train)
        f.attrs["user_map"] = json.dumps(user_map)
        f.attrs["window_size"] = window_size
        f.attrs["stride"] = stride
        f.attrs["split"] = "train"

    # Save test data
    test_file = f"{output_file}:test"
    print(f"Saving test data to {test_file}")
    with h5py.File(test_file, "w") as f:
        f.create_dataset("x", data=x_test)
        f.create_dataset("y", data=y_test)
        f.attrs["user_map"] = json.dumps(user_map)
        f.attrs["window_size"] = window_size
        f.attrs["stride"] = stride
        f.attrs["split"] = "test"

    print(f"\n=== Summary ===")
    print(f"Files processed: {success_count}")
    print(f"Files skipped: {skip_count}")
    print(f"Total files: {len(raw_files)}")
    print("pre-processing complete")
