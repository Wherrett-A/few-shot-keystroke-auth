"""Main execution script for preprocessing Aalto data."""

import os
from typing import Optional
import logging

import argparse

import config
import preprocessing
from generate_mock_data import create_mock_data_files

import typing

input_dir = config.INPUT_DATA_DIR
output_file = config.OUTPUT_FILE
window_size = config.WINDOW_SIZE
stride = config.STRIDE

# Initialize basic logging (INFO by default)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main(
    input_dir: str,
    output_file: str,
    window_size: int,
    stride: int,
    split_ratio: float,
    chunk_size: int,
    num_workers: int,
) -> None:
    """Execute preprocessing functions based on the defined config."""
    preprocessing.process_and_save(
        input_dir=input_dir,
        output_file=output_file,
        window_size=window_size,
        stride=stride,
        split_ratio=split_ratio,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Aalto data")
    # Optional verbose flag to enable DEBUG logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # Define ALL arguments BEFORE parsing to avoid overwriting args
    parser.add_argument("-t", "--test", action="store_true", help="Generate mock data")
    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=None,
        help="Train/test split ratio (default: 0.8)",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=None,
        help="Sessions per chunk for memory efficiency (default: 100)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=str,
        default="1",
        help="Number of worker processes (default: 1, use 'max' for auto-detect)",
    )
    args = parser.parse_args()
    # Activate verbose logging if requested
    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled for preprocessing run")

    split_ratio = args.ratio if args.ratio is not None else config.SPLIT_RATIO
    chunk_size = args.chunk_size if args.chunk_size is not None else config.CHUNK_SIZE

    if args.workers == "max":
        num_workers = os.cpu_count() or 1
    else:
        num_workers = int(args.workers)

    if args.test:
        print("==== Test Mode Enabled ====")
        print("Generating mock data...")
        create_mock_data_files()
        print("Mock data generated successfully.")
        input_dir = config.MOCK_INPUT_DATA_DIR
        output_file = config.MOCK_OUTPUT_DATA_FILE

    main(
        input_dir=input_dir,
        output_file=output_file,
        window_size=window_size,
        stride=stride,
        split_ratio=split_ratio,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )
