"""
Evaluation Script for Keystroke Authentication Model.

This script evaluates a trained keystroke authentication model by computing:
- EER (Equal Error Rate): The point where FAR = FRR
- FAR (False Acceptance Rate): Rate of accepting impostors
- FRR (False Rejection Rate): Rate of rejecting legitimate users
- ROC curve and AUC
- DET curve (Detection Error Tradeoff)

Usage:
    python code/Scripts/evaluate.py --model-path models/keystroke_lstm_20260402_142640.keras
    python code/Scripts/evaluate.py --model-path models/best_model.keras
"""

import argparse
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from datetime import datetime
from typing import Tuple, Dict

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

print("Running evaluation on CPU (CUDA disabled for PTX compatibility)")


def create_triplet_loss(anchor_dim: int, margin: float = 1.0):
    """
    Creates a triplet loss function with online triplet mining.
    Must match the loss used during training.
    """

    def triplet_loss(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        batch_size = tf.shape(y_pred)[0] // 3

        triplets = tf.reshape(y_pred, (batch_size, 3, anchor_dim))
        anchor = triplets[:, 0, :]
        positive = triplets[:, 1, :]
        negative = triplets[:, 2, :]

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        loss = tf.maximum(0.0, pos_dist - neg_dist + margin)
        return tf.reduce_mean(loss)

    return triplet_loss


def build_lstm_model(
    input_shape: tuple,
    embedding_dim: int = 128,
    num_lstm_units: int = 256,
    dropout_rate: float = 0.2,
    use_l2_norm: bool = True,
) -> keras.Model:
    """
    Recreate the model architecture - must match training.
    Enhanced version with L2 normalization support.
    """
    layers_list = [
        layers.Input(shape=input_shape),
        layers.LSTM(num_lstm_units, return_sequences=True, activation="tanh"),
        layers.LayerNormalization(),
        layers.Dropout(dropout_rate),
        layers.LSTM(num_lstm_units, return_sequences=False, activation="tanh"),
        layers.LayerNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(embedding_dim, activation="linear"),
        layers.BatchNormalization(),
    ]

    if use_l2_norm:
        layers_list.append(layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1)))

    model = keras.Sequential(layers_list)
    return model


def load_data(data_path: str, split: str = "test") -> Tuple:
    """Load test data from HDF5 file. Auto-detects feature dimension."""
    if split == "train":
        hdf5_path = f"{data_path}:train"
    elif split == "test":
        hdf5_path = f"{data_path}:test"
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Data file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        features = f["x"][:]
        labels = f["y"][:]
        user_map = json.loads(f.attrs["user_map"])
        window_size = int(f.attrs["window_size"])
        stride = int(f.attrs["stride"])
        # Auto-detect feature dimension from data shape
        feature_dim = features.shape[2] if len(features.shape) == 3 else 2

    print(f"Loaded {len(features):,} samples from {hdf5_path}")
    print(f"Features shape: {features.shape} (feature dim: {feature_dim})")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of users: {len(user_map):,}")

    return features, labels, user_map, window_size, stride, feature_dim


def load_model(model_path: str, config_path: str = None) -> keras.Model:
    """Load model by rebuilding architecture and loading weights."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Loaded metadata from {config_path}")

    if not config:
        raise ValueError("Metadata file required to load model architecture")

    embedding_dim = config.get("embedding_dim", 128)
    lstm_units = config.get("lstm_units", 256)
    dropout_rate = config.get("dropout_rate", 0.2)
    window_size = config.get("window_size", 20)
    feature_dim = config.get("feature_dim", 2)
    use_l2_norm = config.get("use_l2_norm", True)

    print(
        f"Building model: embedding={embedding_dim}, lstm={lstm_units}, window={window_size}, features={feature_dim}, l2_norm={use_l2_norm}"
    )

    model = build_lstm_model(
        input_shape=(window_size, feature_dim),
        embedding_dim=embedding_dim,
        num_lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        use_l2_norm=use_l2_norm,
    )

    keras.config.enable_unsafe_deserialization()

    try:
        temp_model = keras.models.load_model(model_path, compile=False, safe_mode=False)
        model.set_weights(temp_model.get_weights())
        print(f"Weights loaded from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load weights from {model_path}: {e}")
        raise

    return model


def generate_embeddings(
    model: keras.Model, features: np.ndarray, batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings for all features using the trained model.

    Args:
        model: Trained LSTM model
        features: Input features of shape (num_samples, window_size, 2)
        batch_size: Batch size for prediction

    Returns:
        embeddings: Array of shape (num_samples, embedding_dim)
    """
    print(f"Generating embeddings for {len(features):,} samples...")
    embeddings = model.predict(features, batch_size=batch_size, verbose=1)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def compute_pairwise_distances(
    embeddings: np.ndarray, labels: np.ndarray, max_pairs_per_type: int = 500000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise distances with sampling for large datasets.

    For large datasets (>100k samples), uses random sampling to avoid
    memory issues while maintaining statistical validity.
    """
    print("Computing pairwise distances...")
    num_samples = len(embeddings)
    print(f"  Total samples: {num_samples:,}")

    if num_samples > 100000:
        print(
            f"  Large dataset - using sampling (max {max_pairs_per_type:,} pairs/type)"
        )
        return _compute_sampled_distances(embeddings, labels, max_pairs_per_type)

    # Small dataset - compute all pairwise distances
    embeddings_norm = np.sum(embeddings**2, axis=1, keepdims=True)
    distances_sq = (
        embeddings_norm + embeddings_norm.T - 2 * np.dot(embeddings, embeddings.T)
    )
    distances_sq = np.maximum(distances_sq, 0)
    distances = np.sqrt(distances_sq)

    labels_matrix = labels.reshape(-1, 1) == labels.reshape(1, -1)
    upper_tri_mask = np.triu(np.ones_like(labels_matrix, dtype=bool), k=1)

    genuine_distances = distances[labels_matrix & upper_tri_mask]
    impostor_distances = distances[(~labels_matrix) & upper_tri_mask]

    print(f"  Genuine pairs: {len(genuine_distances):,}")
    print(f"  Impostor pairs: {len(impostor_distances):,}")

    return genuine_distances, impostor_distances


def _compute_sampled_distances(
    embeddings: np.ndarray, labels: np.ndarray, max_pairs: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample pairs for large datasets."""
    rng = np.random.RandomState(42)
    num_samples = len(embeddings)
    unique_labels = np.unique(labels)
    print(f"  Unique users: {len(unique_labels):,}")

    # Group by user
    user_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    # Sample genuine pairs (same user, different samples)
    print("  Sampling genuine pairs...")
    genuine_dists = []
    for user_idx, indices in enumerate(user_indices.values()):
        if len(indices) < 2:
            continue
        n_pairs = min(max(1, max_pairs // len(unique_labels)), len(indices) // 2)
        idx1 = rng.choice(indices, size=n_pairs, replace=False)
        idx2 = rng.choice(indices, size=n_pairs, replace=False)
        mask = idx1 != idx2
        if mask.sum() > 0:
            dist = np.linalg.norm(
                embeddings[idx1[mask]] - embeddings[idx2[mask]], axis=1
            )
            genuine_dists.append(dist)

    # Sample impostor pairs (different users)
    print("  Sampling impostor pairs...")
    impostor_dists = []
    attempts = 0
    max_attempts = max_pairs * 3

    while len(impostor_dists) < max_pairs and attempts < max_attempts:
        i, j = rng.choice(num_samples, size=2, replace=False)
        if labels[i] != labels[j]:
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            impostor_dists.append(dist)
        attempts += 1

    genuine_result = np.concatenate(genuine_dists) if genuine_dists else np.array([])
    impostor_result = np.array(impostor_dists)

    print(f"  Sampled genuine pairs: {len(genuine_result):,}")
    print(f"  Sampled impostor pairs: {len(impostor_result):,}")

    return genuine_result, impostor_result


def compute_distance_statistics(genuine: np.ndarray, impostor: np.ndarray) -> Dict:
    """Compute statistics for genuine and impostor distance distributions."""
    return {
        "genuine": {
            "mean": float(np.mean(genuine)),
            "std": float(np.std(genuine)),
            "median": float(np.median(genuine)),
            "min": float(np.min(genuine)),
            "max": float(np.max(genuine)),
            "q1": float(np.percentile(genuine, 25)),
            "q3": float(np.percentile(genuine, 75)),
        },
        "impostor": {
            "mean": float(np.mean(impostor)),
            "std": float(np.std(impostor)),
            "median": float(np.median(impostor)),
            "min": float(np.min(impostor)),
            "max": float(np.max(impostor)),
            "q1": float(np.percentile(impostor, 25)),
            "q3": float(np.percentile(impostor, 75)),
        },
        "separability": {
            "d_prime": float(
                abs(np.mean(impostor) - np.mean(genuine))
                / np.sqrt((np.std(impostor) ** 2 + np.std(genuine) ** 2) / 2)
            ),
        },
    }


def compute_eer(
    genuine_distances: np.ndarray, impostor_distances: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) using interpolation.

    EER is the point where FAR = FRR.
    """
    genuine_sorted = np.sort(genuine_distances)
    impostor_sorted = np.sort(impostor_distances)

    min_thresh = min(genuine_sorted.min(), impostor_sorted.min())
    max_thresh = max(genuine_sorted.max(), impostor_sorted.max())
    thresholds = np.linspace(min_thresh, max_thresh, 1000)

    far_list = []
    frr_list = []

    for thresh in thresholds:
        # FAR: proportion of impostor distances BELOW threshold (false accept)
        far = np.sum(impostor_sorted < thresh) / len(impostor_sorted)
        # FRR: proportion of genuine distances ABOVE threshold (false reject)
        frr = np.sum(genuine_sorted > thresh) / len(genuine_sorted)
        far_list.append(far)
        frr_list.append(frr)

    far_array = np.array(far_list)
    frr_array = np.array(frr_list)

    diff = np.abs(far_array - frr_array)
    eer_idx = np.argmin(diff)
    eer = (far_array[eer_idx] + frr_array[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    return float(eer), float(eer_threshold)


def compute_far_frr_at_threshold(
    genuine_distances: np.ndarray, impostor_distances: np.ndarray, threshold: float
) -> Tuple[float, float]:
    """Compute FAR and FRR at a specific threshold."""
    far = np.sum(impostor_distances <= threshold) / len(impostor_distances)
    frr = np.sum(genuine_distances > threshold) / len(genuine_distances)

    return float(far), float(frr)


def compute_roc_metrics(
    genuine_distances: np.ndarray, impostor_distances: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve metrics.

    For biometric authentication, lower distance = accept, higher = reject.
    We negate distances to use sklearn's roc_curve (which assumes higher = positive).
    """
    # For biometric authentication: genuine (same user) = positive class
    y_true = np.concatenate(
        [np.ones(len(genuine_distances)), np.zeros(len(impostor_distances))]
    )
    # Negate distances: lower distance = higher score = more likely genuine
    y_scores = np.concatenate([-genuine_distances, -impostor_distances])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return float(roc_auc), fpr, tpr, thresholds


def plot_distance_distributions(
    genuine: np.ndarray,
    impostor: np.ndarray,
    eer_threshold: float,
    save_path: str = None,
):
    """Plot histogram of genuine and impostor distance distributions."""
    plt.figure(figsize=(12, 6))

    plt.hist(
        genuine,
        bins=100,
        alpha=0.6,
        label="Genuine (same user)",
        color="green",
        density=True,
    )
    plt.hist(
        impostor,
        bins=100,
        alpha=0.6,
        label="Impostor (different user)",
        color="red",
        density=True,
    )

    plt.axvline(
        x=eer_threshold,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"EER Threshold: {eer_threshold:.4f}",
    )

    plt.xlabel("Euclidean Distance", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Distance Distribution: Genuine vs Impostor", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Distance distribution plot saved to {save_path}")
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, save_path: str = None
):
    """Plot ROC curve."""
    plt.figure(figsize=(10, 8))

    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FAR)", fontsize=12)
    plt.ylabel("True Positive Rate (1 - FRR)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    plt.close()


def evaluate_model(
    model_path: str,
    data_path: str,
    split: str = "test",
    output_dir: str = "reports",
    batch_size: int = 32,
) -> Dict:
    """
    Complete evaluation pipeline.

    Args:
        model_path: Path to trained .keras model file
        data_path: Path to HDF5 data file
        split: Data split to use ('train' or 'test')
        output_dir: Directory to save reports and plots
        batch_size: Batch size for embedding generation

    Returns:
        results: Dictionary containing all evaluation metrics
    """
    print("=" * 70)
    print("KEYSTROKE AUTHENTICATION MODEL EVALUATION")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n[1/6] Loading model...")
    metadata_path = model_path.replace(".keras", "_metadata.json")
    model = load_model(
        model_path, metadata_path if os.path.exists(metadata_path) else None
    )

    print("\n[2/6] Loading data...")
    features, labels, user_map, window_size, stride, feature_dim = load_data(
        data_path, split
    )

    print("\n[3/6] Generating embeddings...")
    embeddings = generate_embeddings(model, features, batch_size)

    nan_mask = np.isnan(embeddings).any(axis=1)
    if nan_mask.any():
        print(f"  Removing {nan_mask.sum():,} embeddings with NaN values")
        embeddings = embeddings[~nan_mask]
        labels = labels[~nan_mask]
        print(f"  Clean embeddings: {len(embeddings):,} samples")

    inf_mask = np.isinf(embeddings).any(axis=1)
    if inf_mask.any():
        print(f"  Removing {inf_mask.sum():,} embeddings with Inf values")
        embeddings = embeddings[~inf_mask]
        labels = labels[~inf_mask]
        print(f"  Clean embeddings: {len(embeddings):,} samples")

    print("\n[4/6] Computing pairwise distances...")
    genuine_distances, impostor_distances = compute_pairwise_distances(
        embeddings, labels
    )

    print("\n[5/6] Computing evaluation metrics...")

    eer, eer_threshold = compute_eer(genuine_distances, impostor_distances)
    print(f"  EER: {eer:.4f} ({eer * 100:.2f}%)")
    print(f"  EER Threshold: {eer_threshold:.4f}")

    far_at_eer, frr_at_eer = compute_far_frr_at_threshold(
        genuine_distances, impostor_distances, eer_threshold
    )
    print(f"  FAR at EER: {far_at_eer:.4f} ({far_at_eer * 100:.2f}%)")
    print(f"  FRR at EER: {frr_at_eer:.4f} ({frr_at_eer * 100:.2f}%)")

    frr_target = 0.01
    genuine_sorted = np.sort(genuine_distances)
    thresh_frr_01 = genuine_sorted[int(np.ceil((1 - frr_target) * len(genuine_sorted)))]
    far_at_frr_01, _ = compute_far_frr_at_threshold(
        genuine_distances, impostor_distances, thresh_frr_01
    )
    print(f"  FAR at FRR=1%: {far_at_frr_01:.4f} ({far_at_frr_01 * 100:.2f}%)")

    roc_auc, fpr, tpr, thresholds = compute_roc_metrics(
        genuine_distances, impostor_distances
    )
    print(f"  ROC AUC: {roc_auc:.4f}")

    stats = compute_distance_statistics(genuine_distances, impostor_distances)
    print(f"\n  Distance Statistics:")
    print(
        f"    Genuine:  mean={stats['genuine']['mean']:.4f}, std={stats['genuine']['std']:.4f}"
    )
    print(
        f"    Impostor: mean={stats['impostor']['mean']:.4f}, std={stats['impostor']['std']:.4f}"
    )
    print(f"    d-prime (separability): {stats['separability']['d_prime']:.4f}")

    print("\n[6/6] Generating visualizations...")

    plot_distance_distributions(
        genuine_distances,
        impostor_distances,
        eer_threshold,
        save_path=os.path.join(output_dir, f"distance_distribution_{timestamp}.png"),
    )

    plot_roc_curve(
        fpr,
        tpr,
        roc_auc,
        save_path=os.path.join(output_dir, f"roc_curve_{timestamp}.png"),
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "data_path": data_path,
        "split": split,
        "num_samples": len(features),
        "num_users": len(user_map),
        "window_size": window_size,
        "feature_dim": feature_dim,
        "embedding_dim": embeddings.shape[1],
        "metrics": {
            "eer": eer,
            "eer_threshold": eer_threshold,
            "far_at_eer": far_at_eer,
            "frr_at_eer": frr_at_eer,
            "far_at_frr_01": far_at_frr_01,
            "roc_auc": roc_auc,
        },
        "distance_statistics": stats,
    }

    results_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {results_path}")

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model:              {os.path.basename(model_path)}")
    print(f"Data:               {os.path.basename(data_path)} ({split} split)")
    print(
        f"Features:           {feature_dim} dimensions ({'enhanced' if feature_dim > 2 else 'baseline'})"
    )
    print(f"Samples:            {len(features):,}")
    print(f"Users:              {len(user_map):,}")
    print("-" * 70)
    print(f"EER:                {eer:.4f} ({eer * 100:.2f}%)")
    print(f"ROC AUC:            {roc_auc:.4f}")
    print(f"FAR at EER:         {far_at_eer:.4f} ({far_at_eer * 100:.2f}%)")
    print(f"FRR at EER:         {frr_at_eer:.4f} ({frr_at_eer * 100:.2f}%)")
    print(f"FAR at FRR=1%:      {far_at_frr_01:.4f} ({far_at_frr_01 * 100:.2f}%)")
    print("=" * 70)

    print("\nRESEARCH QUESTION CHECK:")
    if eer < 0.03:
        print(f"✓ EER < 3% target ACHIEVED: {eer * 100:.2f}%")
    else:
        print(f"✗ EER < 3% target NOT MET: {eer * 100:.2f}% (target: <3%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate keystroke authentication model"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained .keras model file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/preprocessed/aalto.h5",
        help="Path to HDF5 data file (default: data/preprocessed/aalto.h5)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Data split to evaluate (default: test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save evaluation results (default: reports)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )

    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
