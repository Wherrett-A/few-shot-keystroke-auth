import csv
import gc
import glob
import json
import os
import warnings
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple
import logging

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def create_sliding_windows(
    session_features: np.ndarray,
    user_label: int,
    window_size: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[int]]:
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


def process_file(args: Tuple) -> Optional[List[Tuple]]:
    """
    Process a single file and return session data.

    Args:
        args: Tuple of (file_path, user_map, window_size, stride)

    Returns:
        List of (session_key, windows, user_label) tuples, or None on failure
    """
    file_path, user_map, window_size, stride = args
    try:
        df = pd.read_csv(
            file_path,
            sep="\t",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            encoding="latin-1",
            on_bad_lines="warn",
        )
        required_cols = [
            "PARTICIPANT_ID",
            "TEST_SECTION_ID",
            "PRESS_TIME",
            "RELEASE_TIME",
        ]
        if not all(col in df.columns for col in required_cols):
            return None

        participant_id = str(df["PARTICIPANT_ID"].iloc[0])

        if participant_id not in user_map:
            logger.warning(
                "File %s: Unknown PARTICIPANT_ID '%s'",
                os.path.basename(file_path),
                participant_id,
            )
            return None

        user_label = user_map[participant_id]

        df["PRESS_TIME"] = pd.to_numeric(df["PRESS_TIME"], errors="coerce")
        df["RELEASE_TIME"] = pd.to_numeric(df["RELEASE_TIME"], errors="coerce")
        df = df.dropna(subset=["PRESS_TIME", "RELEASE_TIME"])

        if len(df) == 0:
            logger.warning(
                "File %s: No valid timestamp data",
                os.path.basename(file_path),
            )
            return None

        sessions = []
        for test_section_id, session_group in df.groupby(["TEST_SECTION_ID"]):
            session_group = session_group.sort_values(by=["PRESS_TIME"])

            if len(session_group) == 0:
                continue

            session_group["HOLD_TIME"] = (
                session_group["RELEASE_TIME"] - session_group["PRESS_TIME"]
            )

            session_group["NEXT_PRESS_TIME"] = session_group["PRESS_TIME"].shift(-1)
            session_group["FLIGHT_TIME"] = (
                session_group["NEXT_PRESS_TIME"] - session_group["RELEASE_TIME"]
            )
            session_group = session_group.dropna(subset=["FLIGHT_TIME"])
            session_features = session_group[["HOLD_TIME", "FLIGHT_TIME"]].values

            windows, labels = create_sliding_windows(
                session_features, user_label, window_size, stride
            )
            if windows:
                session_key = (participant_id, test_section_id)
                sessions.append((session_key, windows, user_label))

        return sessions if sessions else None
    except Exception as e:
        logger.warning("Failed to process %s: %s", os.path.basename(file_path), e)
        return None


def process_and_save(
    input_dir: str,
    output_file: str,
    window_size: int,
    stride: int,
    split_ratio: float = 0.8,
    chunk_size: int = 100,
    num_workers: int = 1,
) -> None:
    """
    Main function to process and save preprocessed data with chunked HDF5 writing.

    Args:
        input_dir: Directory containing raw keystroke data files
        output_file: Base path for output files (will be suffixed with :train and :test)
        window_size: Size of sliding window for feature extraction
        stride: Step size for sliding window
        split_ratio: Ratio of data to use for training (default: 0.8)
        chunk_size: Number of sessions to process at once (default: 100)
        num_workers: Number of worker processes (default: 1 = single-threaded)
    """
    print("==== Preprocessing ====")
    print(f"Window Size: {window_size}")
    print(f"Stride: {stride}")
    print(f"Train/Test Split: {split_ratio * 100:.0f}:{(1 - split_ratio) * 100:.0f}")
    print(f"Chunk Size: {chunk_size} sessions")
    print(f"Workers: {num_workers}")
    logger.debug(
        "Preprocessing config: window_size=%s, stride=%s, split_ratio=%s, chunk_size=%s, num_workers=%s",
        window_size,
        stride,
        split_ratio,
        chunk_size,
        num_workers,
    )

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")
        logger.debug("Output directory ensured: %s", output_dir)

    if window_size <= 1:
        raise ValueError(f"window_size must be > 1, got {window_size}")

    if not 0 < split_ratio <= 1:
        raise ValueError(f"split_ratio must be in (0, 1], got {split_ratio}")

    raw_files = glob.glob(os.path.join(input_dir, "*_keystrokes.txt"))
    if not raw_files:
        raise FileNotFoundError(
            f"No valid input files found in {input_dir}. "
            f"Expected files matching '*_keystrokes.txt'. "
            f"Please check the directory path and ensure data files exist."
        )

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
            logger.debug("Read file %s with columns: %s", raw_file, list(df.columns))
            if "PARTICIPANT_ID" in df.columns:
                all_participants.update(df["PARTICIPANT_ID"].unique())
                logger.debug(
                    "Found PARTICIPANT_ID column with %d unique IDs in %s",
                    df["PARTICIPANT_ID"].nunique(),
                    raw_file,
                )
        except Exception as e:
            print(f"Error reading {raw_file}: {e}")

    if not all_participants:
        raise ValueError(
            f"No valid participant data found in {input_dir}. "
            f"Ensure files contain 'PARTICIPANT_ID' column."
        )

    user_map = {str(user_id): i for i, user_id in enumerate(sorted(all_participants))}
    print(f"Found {len(user_map)} users.")
    logger.debug("Mapped %d users to numeric labels", len(user_map))

    sessions_data = []
    success_count = 0
    skip_count = 0

    effective_workers = min(num_workers, cpu_count()) if num_workers > 1 else 1

    if effective_workers == 1:
        for i, raw_file in enumerate(raw_files):
            filename = os.path.basename(raw_file)
            print(
                f"[{i + 1:6d}/{len(raw_files):6d}] processing {filename}...", flush=True
            )
            if (i + 1) % 5000 == 0:
                print(f"  -> Processed {i + 1} files so far...")
            try:
                df = pd.read_csv(
                    raw_file,
                    sep="\t",
                    quoting=csv.QUOTE_NONE,
                    escapechar="\\",
                    encoding="latin-1",
                    on_bad_lines="warn",
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
                    missing = [
                        col
                        for col in [
                            "PARTICIPANT_ID",
                            "TEST_SECTION_ID",
                            "PRESS_TIME",
                            "RELEASE_TIME",
                        ]
                        if col not in df.columns
                    ]
                    print(
                        f"  SKIPPED: {filename} - missing required columns: {missing}"
                    )
                    skip_count += 1
                    continue

                participant_id = str(df["PARTICIPANT_ID"].iloc[0])
                user_label = user_map[participant_id]

                for test_section_id, session_group in df.groupby(["TEST_SECTION_ID"]):
                    session_group = session_group.sort_values(by=["PRESS_TIME"])

                    session_group["HOLD_TIME"] = (
                        session_group["RELEASE_TIME"] - session_group["PRESS_TIME"]
                    )
                    session_group["NEXT_PRESS_TIME"] = session_group[
                        "PRESS_TIME"
                    ].shift(-1)
                    session_group["FLIGHT_TIME"] = (
                        session_group["NEXT_PRESS_TIME"] - session_group["RELEASE_TIME"]
                    )
                    session_group = session_group.dropna(subset=["FLIGHT_TIME"])

                    session_features = session_group[
                        ["HOLD_TIME", "FLIGHT_TIME"]
                    ].values

                    windows, labels = create_sliding_windows(
                        session_features, user_label, window_size, stride
                    )
                    if windows:
                        session_key = (participant_id, test_section_id)
                        sessions_data.append((session_key, windows, user_label))
                        success_count += 1
            except Exception as e:
                print(f"  SKIPPED: {os.path.basename(raw_file)} - {e}")
                logger.exception("Error processing %s", os.path.basename(raw_file))
                skip_count += 1
                continue
            del df
            gc.collect()
    else:
        print(f"Processing {len(raw_files)} files with {effective_workers} workers...")
        process_args = [(f, user_map, window_size, stride) for f in raw_files]
        with Pool(processes=effective_workers) as pool:
            results = pool.map(process_file, process_args)

        for result in results:
            if result is None:
                skip_count += 1
            else:
                sessions_data.extend(result)
                success_count += len(result)

    if not sessions_data:
        print("no Windows could be created")
        logger.debug("No Windows could be created from the input data")
        return

    print(f"Total sessions collected: {len(sessions_data)}")
    logger.debug("Total sessions collected: %d", len(sessions_data))

    print("Splitting sessions (not windows) to prevent data leakage...")
    session_indices = list(range(len(sessions_data)))
    session_labels = [sessions_data[i][2] for i in session_indices]

    train_session_indices, test_session_indices = train_test_split(
        session_indices,
        test_size=1 - split_ratio,
        stratify=session_labels,
        random_state=42,
    )

    print(
        f"Train sessions: {len(train_session_indices)}, Test sessions: {len(test_session_indices)}"
    )

    train_file = f"{output_file}:train"
    test_file = f"{output_file}:test"

    total_train_samples = 0
    total_test_samples = 0

    print(f"Writing train data incrementally to {train_file}...")
    with h5py.File(train_file, "w") as f:
        dset_x = f.create_dataset(
            "x",
            shape=(0, window_size, 2),
            maxshape=(None, window_size, 2),
            dtype="float32",
        )
        dset_y = f.create_dataset(
            "y",
            shape=(0,),
            maxshape=(None,),
            dtype="int32",
        )

        for chunk_start in range(0, len(train_session_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(train_session_indices))
            chunk_indices = train_session_indices[chunk_start:chunk_end]

            chunk_windows = []
            chunk_labels = []
            for idx in chunk_indices:
                windows = sessions_data[idx][1]
                label = sessions_data[idx][2]
                chunk_windows.extend(windows)
                chunk_labels.extend([label] * len(windows))

            if not chunk_windows:
                continue

            chunk_x = np.array(chunk_windows, dtype=np.float32)
            chunk_y = np.array(chunk_labels, dtype=np.int32)

            current_size = dset_x.shape[0]
            new_size = current_size + len(chunk_x)
            dset_x.resize(new_size, axis=0)
            dset_y.resize(new_size, axis=0)
            dset_x[current_size:new_size] = chunk_x
            dset_y[current_size:new_size] = chunk_y

            total_train_samples += len(chunk_x)

            del chunk_windows, chunk_labels, chunk_x, chunk_y
            gc.collect()

            if (chunk_end) % (chunk_size * 5) == 0 or chunk_end == len(
                train_session_indices
            ):
                print(
                    f"  -> Written {chunk_end}/{len(train_session_indices)} train sessions..."
                )
                logger.debug(
                    "Written %d/%d train sessions in current chunk",
                    chunk_end,
                    len(train_session_indices),
                )

        f.attrs["user_map"] = json.dumps(user_map)
        f.attrs["window_size"] = window_size
        f.attrs["stride"] = stride
        f.attrs["split"] = "train"

    print(f"Writing test data incrementally to {test_file}...")
    with h5py.File(test_file, "w") as f:
        dset_x = f.create_dataset(
            "x",
            shape=(0, window_size, 2),
            maxshape=(None, window_size, 2),
            dtype="float32",
        )
        dset_y = f.create_dataset(
            "y",
            shape=(0,),
            maxshape=(None,),
            dtype="int32",
        )

        for chunk_start in range(0, len(test_session_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(test_session_indices))
            chunk_indices = test_session_indices[chunk_start:chunk_end]

            chunk_windows = []
            chunk_labels = []
            for idx in chunk_indices:
                windows = sessions_data[idx][1]
                label = sessions_data[idx][2]
                chunk_windows.extend(windows)
                chunk_labels.extend([label] * len(windows))

            if not chunk_windows:
                continue

            chunk_x = np.array(chunk_windows, dtype=np.float32)
            chunk_y = np.array(chunk_labels, dtype=np.int32)

            current_size = dset_x.shape[0]
            new_size = current_size + len(chunk_x)
            dset_x.resize(new_size, axis=0)
            dset_y.resize(new_size, axis=0)
            dset_x[current_size:new_size] = chunk_x
            dset_y[current_size:new_size] = chunk_y

            total_test_samples += len(chunk_x)

            del chunk_windows, chunk_labels, chunk_x, chunk_y
            gc.collect()

            if (chunk_end) % (chunk_size * 5) == 0 or chunk_end == len(
                test_session_indices
            ):
                print(
                    f"  -> Written {chunk_end}/{len(test_session_indices)} test sessions..."
                )
                logger.debug(
                    "Written %d/%d test sessions in current chunk",
                    chunk_end,
                    len(test_session_indices),
                )

        f.attrs["user_map"] = json.dumps(user_map)
        f.attrs["window_size"] = window_size
        f.attrs["stride"] = stride
        f.attrs["split"] = "test"

    print(f"\n=== Summary ===")
    print(f"Files processed: {success_count}")
    print(f"Files skipped: {skip_count}")
    print(f"Total files: {len(raw_files)}")
    print(f"Train: {total_train_samples} samples, Test: {total_test_samples} samples")
    print("pre-processing complete")
    logger.debug("Preprocessing completed successfully.")
