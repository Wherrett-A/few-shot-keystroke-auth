"""Main execution script for preprocessing Aalto data."""

import argparse

import config
import preprocessing
from generate_mock_data import create_mock_data_files

input_dir = config.INPUT_DATA_DIR
output_file = config.OUTPUT_FILE
window_size = config.WINDOW_SIZE
stride = config.STRIDE

if __name__ == "__main__":
    """Executes preprocessing functions based on the defined config."""
    parser = argparse.ArgumentParser(description="Preprocess Aalto data")
    parser.add_argument("-t", "--test", action="store_true", help="Generate mock data")
    args = parser.parse_args()

    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=None,
        help="Train/test split ratio (default: 0.8)",
    )
    args = parser.parse_args()

    split_ratio = args.ratio if args.ratio else config.SPLIT_RATIO

    if args.test:
        print("==== Test Mode Enabled ====")
        print("Generating mock data...")
        create_mock_data_files()
        print("Mock data generated successfully.")
        input_dir = config.MOCK_INPUT_DATA_DIR
        output_file = config.MOCK_OUTPUT_DATA_FILE

    preprocessing.process_and_save(
        input_dir=input_dir,
        output_file=output_file,
        window_size=window_size,
        stride=stride,
        split_ratio=split_ratio,
    )
