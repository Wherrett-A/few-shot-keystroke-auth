import argparse
import json
import os
import sys
from datetime import datetime

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
            # First LSTM layer with return_sequences=True for stacking
            layers.LSTM(
                num_lstm_units,
                return_sequences=True,
                input_shape=input_shape,
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

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanSquaredError(),
        metrics=["accuracy"],
    )

    return model


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


def create_triplet_loss(anchor_dim: int, margin: float = 1.0) -> callable:
    def triplet_loss(y_true, y_pred):
        # Reshape predictions to (batch, anchor_dim)
        y_pred = tf.cast(tf.reshape(y_pred, (-1, anchor_dim)), tf.float32)

        # Split into anchors and positives based on labels
        y_true_float = tf.cast(y_true, tf.float32)
        anchor_mask = tf.cast(y_true_float[:, np.newaxis] == y_true_float, tf.float32)
        positive_mask = tf.cast(y_true_float[:, np.newaxis] != y_true_float, tf.float32)

        # Compute pairwise distances using matrix multiplication
        # |a - b|^2 = |a|^2 + |b|^2 - 2*a.b
        y_pred_sq = tf.reduce_sum(tf.square(y_pred), axis=1, keepdims=True)
        dist_sq = (
            y_pred_sq
            + tf.transpose(y_pred_sq)
            - 2 * tf.matmul(y_pred, y_pred, transpose_b=True)
        )
        distances = tf.cast(dist_sq, tf.float32)

        # Apply margin
        margin_expanded = tf.cast(margin, tf.float32) * tf.ones(
            tf.shape(distances), dtype=tf.float32
        )

        # Triplet loss: d(anchor, positive) - d(anchor, negative) + margin
        # Cast distances to float32 to ensure type compatibility
        loss = tf.reduce_mean(
            tf.maximum(0.0, margin_expanded - tf.cast(distances, tf.float32))
        )

        return loss

    return triplet_loss


def create_contrastive_loss(anchor_dim: int, margin: float = 1.0) -> callable:
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
# TRAINING PIPELINE
# ============================================================================


def create_training_pipeline(
    model: keras.Model, loss_fn: callable, learning_rate: float = 1e-3
) -> tuple:
    # Re-compile with custom loss and learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # Create callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    return model, callbacks


def train(
    model: keras.Model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs: int,
    batch_size: int,
    callbacks: list,
) -> keras.callbacks.History:
    print(f"\nTraining for {epochs} epochs...")
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history


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

    # Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    history = train(
        model, x_train, y_train, x_val, y_val, args.epochs, args.batch_size, callbacks
    )

    # Save the final model
    print("\nSaving model...")
    save_model(model, model_path, metadata)

    print("\nTraining complete!")
    print(f"Best model saved to: {model_path}")
    print(f"History saved to: best_model.keras")


if __name__ == "__main__":
    main()
