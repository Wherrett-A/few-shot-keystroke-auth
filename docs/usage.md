# Few-Shot Keystroke Authentication - Usage Guide

This guide provides instructions for using the preprocessing and training scripts for keystroke authentication.

## Table of Contents

- [Overview](#overview)
- [Preprocessing](#preprocessing)
  - [Data Format](#data-format)
  - [Running Preprocessing](#running-preprocessing)
  - [Configuration](#configuration)
- [Training](#training)
  - [Requirements](#requirements)
  - [Running Training](#running-training)
  - [Configuration](#configuration-1)
  - [Output](#output)
- [Examples](#examples)

## Overview

This repository implements a few-shot keystroke authentication system using adaptive metric learning. The workflow consists of:

1. **Preprocessing**: Converting raw keystroke data into feature windows
2. **Training**: Building and training an LSTM model for user authentication

---

## Preprocessing

The preprocessing script converts raw keystroke timing data into feature windows suitable for LSTM training.

### Data Format

**Input**: Tab-separated keystroke files with columns:
- `PARTICIPANT_ID`: User identifier
- `TEST_SECTION_ID`: Session identifier
- `PRESS_TIME`: Timestamp when key was pressed
- `RELEASE_TIME`: Timestamp when key was released
- Plus other optional columns

**Output**: HDF5 file with:
- Dataset `x`: Shape `(num_samples, window_size, 2)` - normalized features (HOLD_TIME, FLIGHT_TIME)
- Dataset `y`: Shape `(num_samples,)` - user labels (integers)
- Attributes: `user_map`, `window_size`, `stride`, `split`

### Running Preprocessing

```bash
cd code/Scripts/preprocessing/aalto
python3 run_preprocessing.py
```

### Configuration

Edit `config.py` to customize:

```python
# Input/Output paths
INPUT_DATA_DIR = "data/datasets/aalto/files"
OUTPUT_FILE = "data/preprocessed/aalto.h5"

# Sliding window parameters
WINDOW_SIZE = 20  # Number of keystrokes in each window
STRIDE = 5        # Step size between windows

# Train/test split ratio
SPLIT_RATIO = 0.8  # 80% training, 20% testing

# Mock data generation (for testing)
MOCK_INPUT_DATA_DIR = "data/mock_data"
MOCK_OUTPUT_DATA_FILE = "data/mock_output/aalto_mock.h5"
NUM_MOCK_USERS = 50
SESSIONS_PER_MOCK_USER = 10
```

### Testing Mode

Generate mock data for testing:

```bash
python3 run_preprocessing.py -t
```

---

## Training

### Requirements

Install dependencies:

```bash
pip install -r code/requirements.txt
```

### Running Training

```bash
cd code/Scripts
python3 train.py --data-path data/mock_output/aalto_mock.h5
```

### Configuration

#### Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | `data/mock_output/aalto_mock.h5` | Path to preprocessed HDF5 data file |
| `--split` | `train` | Data split: `train` or `test` |

#### Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--embedding-dim` | `128` | Dimension of embedding space |
| `--lstm-units` | `128` | Number of LSTM units per layer |
| `--dropout-rate` | `0.5` | Dropout rate for regularization |

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `32` | Samples per gradient update |
| `--learning-rate` | `0.001` | Learning rate for Adam optimizer |

#### Output Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir` | `models` | Directory to save trained model |
| `--model-name` | `None` | Custom model filename |

#### GPU Argument

| Argument | Default | Description |
|----------|---------|-------------|
| `--gpu` | `False` | Force GPU usage |

### Output

Trained models are saved as:
- `models/{model_name}.keras` - Native Keras model file (recommended format)
- `models/{model_name}_metadata.json` - Training metadata
- `best_model.keras` - Best model during training

---

## Examples

### Quick Start

```bash
# Generate mock data for testing
python3 code/Scripts/preprocessing/aalto/run_preprocessing.py -t

# Train model on mock data
python3 code/Scripts/train.py --data-path data/mock_output/aalto_mock.h5
```

### Full Workflow

```bash
# 1. Preprocess real data
python3 code/Scripts/preprocessing/aalto/run_preprocessing.py

# 2. Train with custom parameters
python3 code/Scripts/train.py \
    --data-path data/preprocessed/aalto.h5 \
    --embedding-dim 256 \
    --lstm-units 256 \
    --epochs 200 \
    --batch-size 64 \
    --model-name aalto_model
```

### Training with Test Split

```bash
# Use test split for evaluation
python3 code/Scripts/train.py --split test
```

### Training with Early Stopping

```bash
# Train with early stopping (enabled by default after 10 epochs)
python3 code/Scripts/train.py --epochs 500
```

### Using GPU (if available)

```bash
# Force GPU usage
python3 code/Scripts/train.py --gpu
```

---

## Loading Trained Models

Models saved in `.keras` format can be loaded as follows:

```python
from tensorflow import keras

# Load model (requires safe_mode=False due to Lambda layer)
model = keras.models.load_model('models/my_model.keras', safe_mode=False)
```

Alternatively, enable unsafe deserialization globally:

```python
import keras
keras.config.enable_unsafe_deserialization()
model = keras.models.load_model('models/my_model.keras')
```

## Model Architecture

The trained model has the following architecture:

```
Input (window_size, 2) -> LSTM -> LSTM -> Dropout -> Dense -> Embedding
```

- **Input**: Sliding window of keystroke features
- **Two LSTM layers**: Captures temporal dependencies
- **Dropout**: Prevents overfitting
- **Dense layer**: Produces embedding vector

---

## Training Metadata

The saved metadata file contains (JSON format):

```json
{
    "timestamp": "2024-01-XX",  // Training timestamp
    "data_path": "...",          // Input data file
    "split": "train",            // Data split used
    "embedding_dim": 128,        // Embedding dimension
    "lstm_units": 128,           // LSTM units per layer
    "dropout_rate": 0.5,         // Dropout rate
    "epochs": 100,               // Number of epochs
    "batch_size": 32,            // Batch size
    "learning_rate": 0.001,      // Learning rate
    "window_size": 20,           // Window size from preprocessing
    "stride": 5,                 // Stride from preprocessing
    "num_users": 50              // Number of users in dataset
}
```
