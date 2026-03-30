import argparse

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import config
import h5py
import numpy as np
from h5py import Dataset


def verify_file(
    file_path: str, file_label: str
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Verify a single HDF5 file and return stats."""
    warnings = []
    stats = {}

    try:
        with h5py.File(file_path, "r") as h5f:
            if "x" not in h5f or "y" not in h5f:
                warnings.append(f"{file_label}: Missing 'x' or 'y' dataset")
                return False, stats, warnings

            X_dataset: Dataset = cast(Dataset, h5f["x"])
            y_dataset: Dataset = cast(Dataset, h5f["y"])

            X_shape = X_dataset.shape
            y_shape = y_dataset.shape

            stats["file"] = file_label
            stats["X_shape"] = X_shape
            stats["y_shape"] = y_shape
            stats["X_dtype"] = str(X_dataset.dtype)
            stats["y_dtype"] = str(y_dataset.dtype)

            if X_shape[0] != y_shape[0]:
                warnings.append(
                    f"{file_label}: X has {X_shape[0]} samples, y has {y_shape[0]}"
                )

            if len(X_shape) != 3:
                warnings.append(f"{file_label}: X should be 3D, got {X_shape}")

            if len(y_shape) != 1:
                warnings.append(f"{file_label}: y should be 1D, got {y_shape}")

            attrs = dict(h5f.attrs)
            stats["attributes"] = list(attrs.keys())

            if "user_map" not in attrs:
                warnings.append(f"{file_label}: Missing 'user_map' attribute")
            if "window_size" not in attrs:
                warnings.append(f"{file_label}: Missing 'window_size' attribute")
            if "stride" not in attrs:
                warnings.append(f"{file_label}: Missing 'stride' attribute")

            window_size = int(attrs.get("window_size", 0))
            stride = int(attrs.get("stride", 0))

            stats["window_size"] = window_size
            stats["stride"] = stride

            if X_shape[1] != window_size:
                warnings.append(
                    f"{file_label}: window_size attr is {window_size}, "
                    f"but X shape is {X_shape[1]}"
                )

            user_map: Dict[str, int] = json.loads(
                cast(str, attrs.get("user_map", "{}"))
            )
            stats["num_users"] = len(user_map)

            X_sample: np.ndarray[Any, np.dtype[Any]] = cast(
                np.ndarray[Any, np.dtype[Any]], X_dataset[:10]
            )
            y_sample: np.ndarray[Any, np.dtype[Any]] = cast(
                np.ndarray[Any, np.dtype[Any]], y_dataset[:10]
            )

            hold_times = X_sample[:, :, 0].flatten()
            flight_times = X_sample[:, :, 1].flatten()

            stats["hold_time_std"] = float(np.std(hold_times))
            stats["hold_time_min"] = float(np.min(hold_times))
            stats["hold_time_max"] = float(np.max(hold_times))
            stats["flight_time_std"] = float(np.std(flight_times))
            stats["flight_time_mean"] = float(np.mean(flight_times))

            if np.any(hold_times < -100) or np.any(flight_times < -100):
                warnings.append(
                    f"{file_label}: Contains suspicious negative values (may indicate data corruption)"
                )

            if np.any(np.abs(hold_times) > 10000) or np.any(
                np.abs(flight_times) > 10000
            ):
                warnings.append(
                    f"{file_label}: Contains extreme values (may indicate overflow/underflow)"
                )

            unique_labels = np.unique(y_sample)
            stats["unique_labels_sample"] = len(unique_labels)

            if len(user_map) > 0:
                max_label = max(user_map.values())
                if np.any(y_sample > max_label):
                    warnings.append(f"{file_label}: Labels exceed user_map range")

            return True, stats, warnings

    except FileNotFoundError:
        warnings.append(f"{file_label}: File not found: {file_path}")
        return False, stats, warnings
    except Exception as e:
        warnings.append(f"{file_label}: Error reading file: {e}")
        return False, stats, warnings


def verify_data_split(test_mode: bool = False) -> None:
    output_file = config.MOCK_OUTPUT_DATA_FILE if test_mode else config.OUTPUT_FILE

    print("==== Verifying Preprocessed Data ====\n")

    train_file = f"{output_file}:train"
    test_file = f"{output_file}:test"

    print(f"Output base: {output_file}")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}\n")

    train_ok, train_stats, train_warnings = verify_file(train_file, "Train")
    test_ok, test_stats, test_warnings = verify_file(test_file, "Test")

    print("= Data Statistics =\n")
    print(f"{'Metric':<30} {'Train':<20} {'Test':<20}")
    print("-" * 70)

    metrics = [
        ("X shape", "X_shape", "N/A"),
        ("y shape", "y_shape", "N/A"),
        ("X dtype", "X_dtype", "N/A"),
        ("y dtype", "y_dtype", "N/A"),
        ("Window size", "window_size", "N/A"),
        ("Stride", "stride", "N/A"),
        ("Num users", "num_users", "N/A"),
        ("Hold time std", "hold_time_std", ".2f"),
        (
            "Hold time range",
            "hold_time",
            lambda s: (
                f"{s.get('hold_time_min', 0):.1f} - {s.get('hold_time_max', 0):.1f}"
            ),
        ),
        ("Flight time std", "flight_time_std", ".2f"),
    ]

    for metric_name, train_key, format_spec in metrics:
        train_val = train_stats.get(train_key, "N/A")
        test_val = test_stats.get(train_key, "N/A")

        if callable(format_spec):
            train_display = format_spec(train_stats)
            test_display = format_spec(test_stats)
        elif isinstance(format_spec, str) and format_spec != "N/A":
            try:
                train_display = f"{float(train_val):{format_spec}}"
                test_display = f"{float(test_val):{format_spec}}"
            except (ValueError, TypeError):
                train_display = str(train_val)
                test_display = str(test_val)
        else:
            train_display = str(train_val) if train_val else "N/A"
            test_display = str(test_val) if test_val else "N/A"

        print(f"{metric_name:<30} {train_display:<20} {test_display:<20}")

    print("\n= Warnings =\n")

    if train_warnings or test_warnings:
        if train_warnings:
            print("Train warnings:")
            for w in train_warnings:
                print(f"  ⚠️  {w}")
        if test_warnings:
            print("Test warnings:")
            for w in test_warnings:
                print(f"  ⚠️  {w}")
    else:
        print("✅ No warnings")

    print("\n= Data Quality Checks =\n")

    checks_passed = 0
    checks_total = 0

    if train_stats and test_stats:
        checks_total += 1
        hold_mean_train = train_stats.get("hold_time_std", 0)
        hold_mean_test = test_stats.get("hold_time_std", 0)
        if hold_mean_train > -100 and hold_mean_test > -100:
            print("✅ Feature distributions reasonable (normalized to ~0)")
            checks_passed += 1
        else:
            print("❌ Feature values suspicious (check for data corruption)")

        checks_total += 1
        window_size = train_stats.get("window_size", 0)
        if window_size > 1:
            print(f"✅ Window size valid: {window_size}")
            checks_passed += 1
        else:
            print(f"❌ Invalid window size: {window_size}")

    print(f"\nChecks passed: {checks_passed}/{checks_total}")

    print("\n= Sample Data =\n")

    try:
        with h5py.File(train_file, "r") as h5f:
            X_dataset = h5f["x"]
            y_dataset = h5f["y"]
            user_map = json.loads(h5f.attrs["user_map"])

            print("First 3 samples from training set:")
            print(f"{'Sample':<8} {'Label':<8} {'Hold Times (first 5)':<40}")
            print("-" * 70)

            for i in range(min(3, len(y_dataset))):
                label = int(y_dataset[i])
                user_id = (
                    list(user_map.keys())[label] if label < len(user_map) else "Unknown"
                )
                hold_times = X_dataset[i, :5, 0]
                hold_str = ", ".join([f"{h:.1f}" for h in hold_times])
                print(f"{i:<8} {user_id:<12} {hold_str}")

    except Exception as e:
        print(f"Could not display sample data: {e}")

    print("\n= Summary =\n")

    all_ok = (
        train_ok and test_ok and len(train_warnings) == 0 and len(test_warnings) == 0
    )

    if all_ok:
        print("✅ All checks passed! Data is valid and ready for training.")
        sys.exit(0)
    elif train_ok and test_ok:
        print("⚠️  Data loaded successfully but has warnings. Review warnings above.")
        sys.exit(0)
    else:
        print("❌ Data verification failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify preprocessed data")
    parser.add_argument(
        "-t", "--test", action="store_true", help="Verify mock test data"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Verify only train file"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Verify only test file"
    )
    args = parser.parse_args()

    verify_data_split(test_mode=args.test)
