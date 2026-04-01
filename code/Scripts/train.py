import argparse
import json
import os
import sys
from datetime import datetime
from typing import Callable

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import saving

# Configure TensorFlow GPU - automatic with memory growth enabled
if not tf.config.list_physical_devices("GPU"):
    print("No GPU detected, using CPU")
else:
    print(f"GPU detected: {tf.config.list_physical_devices('GPU')}")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Add parent directory to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "preprocessing", "aalto"))
import config


# ============================================================================
# DATA LOADING
# ============================================================================


def load_data(data_path: str, split: str = "train") -> tuple:
    # Convert to proper HDF5 path format
    if split == "train":
        hdf5_path = f"{data_path}:train"
    elif split == "test":
        hdf5_path = f"{data_path}:test"
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

    # Check if file exists
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Data file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        features = f["x"][:]
        labels = f["y"][:]
        user_map = json.loads(f.attrs["user_map"])
        window_size = int(f.attrs["window_size"])
        stride = int(f.attrs["stride"])

    print(f"Loaded {len(features)} samples from {hdf5_path}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of users: {len(user_map)}")

    return features, labels, user_map, window_size, stride


def create_triplet_dataset(features: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Generate triplets (anchor, positive, negative) for metric learning.

    For each anchor sample:
    - Positive: random sample from same user
    - Negative: random sample from different user

    Returns triplets array of shape (num_triplets * 3, *feature_shape)
    and dummy labels.
    """
    unique_labels = np.unique(labels)
    triplets_x = []
    triplets_y = []  # Dummy labels (not used in loss)

    # Group samples by user for efficient triplet mining
    user_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    for anchor_idx in range(len(features)):
        anchor_label = labels[anchor_idx]

        # Find positive candidates (same user, excluding anchor)
        same_user_indices = user_indices[anchor_label]
        if len(same_user_indices) < 2:
            continue  # Need at least 2 samples from this user
        positive_candidates = same_user_indices[same_user_indices != anchor_idx]
        if len(positive_candidates) == 0:
            continue

        # Find negative candidates (different user)
        different_user_indices = np.where(labels != anchor_label)[0]
        if len(different_user_indices) == 0:
            continue

        # Sample one positive and one negative
        positive_idx = np.random.choice(positive_candidates)
        negative_idx = np.random.choice(different_user_indices)

        triplets_x.append(
            [features[anchor_idx], features[positive_idx], features[negative_idx]]
        )
        triplets_y.append([anchor_label, anchor_label, anchor_label])  # Dummy

    triplets_x = np.array(triplets_x)
    triplets_y = np.array(triplets_y)

    # Reshape to (num_triplets * 3, *feature_shape)
    num_triplets = len(triplets_x)
    triplets_x = triplets_x.reshape((num_triplets * 3,) + features.shape[1:])
    triplets_y = triplets_y.reshape(num_triplets * 3)

    print(f"Generated {num_triplets} triplets from {len(features)} samples")

    return triplets_x, triplets_y


class TripletDataGenerator(keras.utils.Sequence):
    """
    Dynamic triplet generator with random negative sampling.

    Generates fresh triplets at the start of each epoch with random
    positive and negative sampling.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        margin: float = 1.0,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.margin = margin
        self.shuffle = shuffle
        self.rng = np.random.RandomState(random_state)

        # Group indices by user for efficient sampling
        self.unique_labels = np.unique(labels)
        self.user_indices = {
            label: np.where(labels == label)[0] for label in self.unique_labels
        }

        # Generate initial triplets
        self.on_epoch_end()

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return int(np.ceil(len(self.triplet_indices) / self.batch_size))

    def __getitem__(self, index: int) -> tuple:
        """Get one batch of triplets."""
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.triplet_indices))
        batch_indices = self.triplet_indices[start_idx:end_idx]

        # Build batch of triplets (anchor, positive, negative)
        batch_x = np.array(
            [
                [
                    self.features[anchor_idx],
                    self.features[positive_idx],
                    self.features[negative_idx],
                ]
                for anchor_idx, positive_idx, negative_idx in batch_indices
            ]
        )

        # Reshape to (batch_size * 3, *feature_shape)
        batch_size = len(batch_x)
        batch_x = batch_x.reshape((batch_size * 3,) + self.features.shape[1:])

        # Dummy labels (not used in triplet loss)
        batch_y = np.zeros(batch_size * 3)

        return batch_x, batch_y

    def on_epoch_end(self) -> None:
        """Generate new triplets at the end of each epoch."""
        self.triplet_indices = []
        total_samples = len(self.features)

        if self.shuffle:
            all_indices = self.rng.permutation(total_samples)
        else:
            all_indices = np.arange(total_samples)

        print(f"Generating triplets...", flush=True)

        report_interval = max(1, total_samples // 10)

        for i, anchor_idx in enumerate(all_indices):
            anchor_label = self.labels[anchor_idx]

            same_user = self.user_indices[anchor_label]
            same_user_excl = same_user[same_user != anchor_idx]

            if len(same_user_excl) < 1:
                continue

            different_user_mask = self.labels != anchor_label
            different_user_indices = np.where(different_user_mask)[0]

            if len(different_user_indices) < 1:
                continue

            positive_idx = self.rng.choice(same_user_excl)
            negative_idx = self.rng.choice(different_user_indices)

            self.triplet_indices.append((anchor_idx, positive_idx, negative_idx))

            if (i + 1) % report_interval == 0:
                progress = (i + 1) / total_samples * 100
                print(
                    f"  Progress: {progress:.1f}% ({i + 1}/{total_samples}) - {len(self.triplet_indices)} triplets",
                    flush=True,
                )

        if self.shuffle:
            self.rng.shuffle(self.triplet_indices)

        print(
            f"Generated {len(self.triplet_indices)} triplets from {total_samples} samples",
            flush=True,
        )


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================


def build_lstm_model(
    input_shape: tuple,
    num_classes: int,
    embedding_dim: int = 128,
    num_lstm_units: int = 128,
    dropout_rate: float = 0.5,
) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            # First LSTM layer with return_sequences=True for stacking
            layers.LSTM(
                num_lstm_units,
                return_sequences=True,
                activation="tanh",
            ),
            # Second LSTM layer
            layers.LSTM(num_lstm_units, return_sequences=False, activation="tanh"),
            # Dropout for regularization
            layers.Dropout(dropout_rate),
            # Output embedding layer
            layers.Dense(embedding_dim, activation="linear"),
            # Cast to float32 to ensure consistent dtype
            layers.Lambda(lambda x: tf.cast(x, tf.float32)),
        ]
    )

    return model


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


def create_triplet_loss(anchor_dim: int, margin: float = 1.0) -> Callable:
    """
    Creates a triplet loss function with online triplet mining.

    The model should output embeddings for (anchor, positive, negative) triplets.
    Input format: y_pred shape = (batch_size * 3, embedding_dim)
    where each group of 3 consecutive samples is (anchor, positive, negative).
    """

    def triplet_loss(y_true, y_pred):
        # Reshape predictions to (batch_size * 3, embedding_dim)
        y_pred = tf.cast(y_pred, tf.float32)
        batch_size = tf.shape(y_pred)[0] // 3

        # Reshape to (batch_size, 3, embedding_dim) to get triplets
        triplets = tf.reshape(y_pred, (batch_size, 3, anchor_dim))
        anchor = triplets[:, 0, :]  # (batch_size, embedding_dim)
        positive = triplets[:, 1, :]  # (batch_size, embedding_dim)
        negative = triplets[:, 2, :]  # (batch_size, embedding_dim)

        # Compute Euclidean distances
        # d(a, p) = ||a - p||_2
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        # d(a, n) = ||a - n||_2
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        # Triplet loss: max(d(a,p) - d(a,n) + margin, 0)
        # We want anchor-positive distance to be SMALLER than anchor-negative distance
        loss = tf.maximum(0.0, pos_dist - neg_dist + margin)

        return tf.reduce_mean(loss)

    return triplet_loss


def create_contrastive_loss(anchor_dim: int, margin: float = 1.0) -> Callable:
    def contrastive_loss(y_true, y_pred):
        # Reshape predictions to (batch, anchor_dim)
        y_pred = tf.reshape(y_pred, (-1, anchor_dim))

        # Compute pairwise distances
        distances = tf.reduce_sum(
            tf.square(y_pred[:, np.newaxis] - y_pred[np.newaxis, :]), axis=2
        )

        # Labels indicate same class (0) or different class (1)
        # Contrastive loss: y * d^2 + (1-y) * max(0, margin - d)^2
        margin_expanded = margin * tf.ones_like(distances)

        loss = y_true * distances + (1 - y_true) * tf.square(
            tf.maximum(0, margin_expanded - tf.sqrt(distances + 1e-6))
        )

        return tf.reduce_mean(loss)

    return contrastive_loss


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================


def save_model(model: keras.Model, path: str, metadata: dict = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".h5"):
        path = path.replace(".h5", ".keras")
    saving.save_model(model, path)
    print(f"Model saved to {path}")

    # Save metadata if provided
    if metadata:
        metadata_path = path.replace(".keras", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")


def load_model(model_path: str) -> keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM model for keystroke authentication"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/mock_output/aalto_mock.h5",
        help="Path to preprocessed HDF5 data file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Data split to use",
    )

    # Model arguments
    parser.add_argument(
        "--embedding-dim", type=int, default=128, help="Dimension of embedding space"
    )
    parser.add_argument(
        "--lstm-units", type=int, default=128, help="Number of LSTM units"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.5,
        help="Dropout rate for regularization",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )

    # Output arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom model name (default: timestamp-based)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Force use of GPU (TensorFlow auto-detects GPU by default)",
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data
    print("Loading data...")
    features, labels, user_map, window_size, stride = load_data(
        args.data_path, args.split
    )

    # Prepare data for training
    # Split into train/validation
    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"\nTrain split: {len(x_train)} samples")
    print(f"Validation split: {len(x_val)} samples")

    # Build model
    print("\nBuilding model...")
    model = build_lstm_model(
        input_shape=(window_size, 2),
        num_classes=len(user_map),
        embedding_dim=args.embedding_dim,
        num_lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=create_triplet_loss(anchor_dim=args.embedding_dim, margin=1.0),
    )

    print("\nCreating dynamic triplet generators...")
    train_generator = TripletDataGenerator(
        features=x_train,
        labels=y_train,
        batch_size=args.batch_size,
        margin=1.0,
        shuffle=True,
        random_state=42,
    )

    val_generator = TripletDataGenerator(
        features=x_val,
        labels=y_val,
        batch_size=args.batch_size,
        margin=1.0,
        shuffle=False,
        random_state=42,
    )

    # Create training pipeline
    loss_fn = create_triplet_loss(anchor_dim=args.embedding_dim)
    model, callbacks = create_training_pipeline(model, loss_fn, args.learning_rate)

    # Generate model filename
    if args.model_name:
        model_filename = f"{args.model_name}.keras"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"keystroke_lstm_{timestamp}.keras"

    model_path = os.path.join(args.model_dir, model_filename)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_path": args.data_path,
        "split": args.split,
        "embedding_dim": args.embedding_dim,
        "lstm_units": args.lstm_units,
        "dropout_rate": args.dropout_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "window_size": window_size,
        "stride": stride,
        "num_users": len(user_map),
    }

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the final model
    print("\nSaving model...")
    save_model(model, model_path, metadata)

    print("\nTraining complete!")
    print(f"Best model saved to: {model_path}")
    print(f"History saved to: best_model.keras")


if __name__ == "__main__":
    main()
