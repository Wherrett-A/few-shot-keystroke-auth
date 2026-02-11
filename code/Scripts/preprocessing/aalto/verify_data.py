import argparse
import json
from typing import Any, Dict, cast

import config
import h5py
import numpy as np
from h5py import Dataset

if __name__ == "__main__":
    """Loads and displays information from the H5 data file."""
    parser = argparse.ArgumentParser(description="Verify preprocessed data")
    parser.add_argument(
        "-t", "--test", action="store_true", help="Verify mock test data"
    )
    args = parser.parse_args()

    output_file = config.MOCK_OUTPUT_DATA_FILE if args.test else config.OUTPUT_FILE

    print(f"==== Verifying data in {output_file} ====")

    try:
        with h5py.File(output_file, "r") as h5f:
            X_dataset = cast(Dataset, h5f["x"])
            y_dataset = cast(Dataset, h5f["y"])
            X_loaded = cast(np.ndarray[Any, np.dtype[Any]], X_dataset[:])
            y_loaded = cast(np.ndarray[Any, np.dtype[Any]], y_dataset[:])

            user_map_loaded: Dict[str, int] = json.loads(
                cast(str, h5f.attrs["user_map"])
            )

            window_size = cast(int, h5f.attrs["window_size"])

        print(f"Loaded X shape: {X_loaded.shape}")
        print(f"Loaded y shape: {y_loaded.shape}")
        print(f"Window Size used: {window_size}")
        print("\nExample loaded window (first 3 features of sample 0):")
        print(X_loaded[0, :3, :])
        print(
            f"Label for sample 0: {y_loaded[0]} (User ID: {list(user_map_loaded.keys())[y_loaded[0]]})"
        )

    except FileNotFoundError:
        print(f"Error: Output file '{output_file}' not found.")
        if args.test:
            print("Please run 'run_preprocessing.py' with -t flag first.")
        else:
            print("Please run 'run_preprocessing.py' first.")
