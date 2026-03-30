"""Configuration file for pre-processing"""

INPUT_DATA_DIR: str = "data/datasets/aalto/files"
OUTPUT_FILE: str = "data/preprocessed/aalto.h5"

WINDOW_SIZE: int = 20
STRIDE: int = 5

MOCK_INPUT_DATA_DIR: str = "data/mock_data"
MOCK_OUTPUT_DATA_FILE: str = "data/mock_output/aalto_mock.h5"
NUM_MOCK_USERS: int = 50
SESSIONS_PER_MOCK_USER: int = 10

# Train/test split ratio (default 80:20)
SPLIT_RATIO: float = 0.8

# Number of worker processes for parallel file processing (default: 1 = single-threaded)
NUM_WORKERS: int = 1

# Chunk size for memory-efficient processing (sessions per chunk)
CHUNK_SIZE: int = 100

# Validation constraints
MIN_WINDOW_SIZE: int = 2
MIN_SPLIT_RATIO: float = 0.0
MAX_SPLIT_RATIO: float = 1.0
