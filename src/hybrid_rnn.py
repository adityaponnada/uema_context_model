"""
Step 6: Hybrid GTCN model (Setup 2 - Within-user split).

Trains a Gated Temporal Convolutional Network using a within-user split:
first 10% of each user's data for training, remaining 90% for validation.
Uses TimeDistributed wrapper to process variable-length chunk sequences.
Evaluates on held-out data and computes permutation importance.

Architecture: TimeDistributed(GTCN_Core) with manual masking
"""

import os
import gc
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from src.helpers import (
    drop_zero_mi_columns,
    impute_group_median_then_ffill,
    impute_test_with_medians_and_ffill,
    z_normalize_columns,
    z_normalize_test_using_global_mean,
    configure_gpu,
    optimized_loss_fn,
    optimized_f1_class0,
    mask_generator_fn,
    get_custom_objects,
    prepare_within_user_tensors,
    preprocess_held_out_data,
    run_final_test,
    analyze_user_f1_distribution,
    find_optimal_threshold,
    calculate_permutation_importance,
    DEFAULT_COLS_TO_SCALE,
    save_figure,
    save_text_results,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Hybrid GTCN model (Setup 2)."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save figures, results, and intermediate files.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Full path to processed features CSV file.")
    parser.add_argument("--heldout_csv", type=str, required=True,
                        help="Full path to processed held-out features CSV file.")
    parser.add_argument("--train_frac", type=float, default=0.1,
                        help="Fraction of each user's data for training (default: 0.1).")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20).")
    parser.add_argument("--use_cpu", action="store_true", default=True,
                        help="Use CPU only (default: True).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override decision threshold (if None, find optimal).")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only evaluate using existing model.")
    return parser.parse_args()


def split_train_test_by_participant(
    df: pd.DataFrame, id_col: str = "participant_id", train_frac: float = 0.1
):
    """Split each participant's data temporally (first train_frac for train, rest for test).

    Args:
        df: Input DataFrame.
        id_col: Participant ID column.
        train_frac: Fraction of each user's data for training.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_list, test_list = [], []
    for pid, group in df.groupby(id_col):
        n = len(group)
        split_idx = int(np.floor(train_frac * n))
        group_sorted = group.sort_index()
        train_list.append(group_sorted.iloc[:split_idx])
        test_list.append(group_sorted.iloc[split_idx:])
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df


def build_hybrid_gtcn(l_chunk: int, n_features: int, conv_filters: int = 8, kernel_size: int = 32):
    """Build the Hybrid GTCN model (Setup 2) with TimeDistributed wrapper.

    Uses manual masking via Lambda layer to handle sentinel padding.

    Args:
        l_chunk: Input sequence length per chunk.
        n_features: Number of input features.
        conv_filters: Number of convolution filters.
        kernel_size: Kernel size for initial convolution.

    Returns:
        Compiled Keras Model.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv1D, multiply, Activation, Dropout, TimeDistributed, Dense, Lambda
    )

    def gated_block(x, d, f, k, m):
        a = Conv1D(f, k, padding="causal", dilation_rate=d)(x)
        b = Conv1D(f, k, padding="causal", dilation_rate=d, activation="sigmoid")(x)
        res = Activation("relu")(multiply([a, b]))
        return multiply([res, m])

    # Outer model: accepts (None, l_chunk, n_features) chunks
    inputs = Input(shape=(None, l_chunk, n_features), name="main_input")

    # Inner core model: processes single chunks
    inner_input = Input(shape=(l_chunk, n_features))

    # Manual masking
    mask = Lambda(mask_generator_fn, output_shape=(l_chunk, 1))(inner_input)
    x = multiply([inner_input, mask])

    x = Conv1D(conv_filters, kernel_size, padding="causal", activation="relu")(x)
    x = multiply([x, mask])

    x = gated_block(x, 2, conv_filters, 2, mask)
    x = gated_block(x, 4, conv_filters, 2, mask)
    x = gated_block(x, 8, conv_filters, 2, mask)
    x = Dropout(0.2)(x)

    inner_output = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    inner_output = multiply([inner_output, mask])

    chunk_processor = Model(inner_input, inner_output, name="GTCN_Core")

    # Wrap in TimeDistributed to process each chunk
    outputs = TimeDistributed(chunk_processor)(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_hybrid_model(
    model, X_train, Y_train, X_val, Y_val, epochs: int, models_dir: str
):
    """Train the Hybrid GTCN model.

    Args:
        model: Compiled Keras model.
        X_train: Training features (N, 1, L_chunk, D).
        Y_train: Training labels.
        X_val: Validation features (N, 4, L_chunk, D).
        Y_val: Validation labels.
        epochs: Number of epochs.
        models_dir: Directory to save best model.

    Returns:
        Training history dict.
    """
    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}
    best_val_f1 = -1.0

    print("\nStarting Within-User 10/90 Training...")

    for epoch in range(epochs):
        start = time.time()

        # Training phase
        t_loss_batch, t_f1_batch = [], []
        for u in range(X_train.shape[0]):
            res = model.train_on_batch(X_train[u:u + 1], Y_train[u:u + 1], return_dict=True)
            t_loss_batch.append(res["loss"])
            t_f1_batch.append(res["optimized_f1_class0"])

        # Validation phase
        v_loss_batch, v_f1_batch = [], []
        for u in range(X_val.shape[0]):
            res = model.test_on_batch(X_val[u:u + 1], Y_val[u:u + 1], return_dict=True)
            v_loss_batch.append(res["loss"])
            v_f1_batch.append(res["optimized_f1_class0"])

        avg_t_loss, avg_t_f1 = np.mean(t_loss_batch), np.mean(t_f1_batch)
        avg_v_loss, avg_v_f1 = np.mean(v_loss_batch), np.mean(v_f1_batch)

        history["train_loss"].append(float(avg_t_loss))
        history["train_f1"].append(float(avg_t_f1))
        history["val_loss"].append(float(avg_v_loss))
        history["val_f1"].append(float(avg_v_f1))

        print(f"Epoch {epoch + 1}/{epochs} | {time.time() - start:.1f}s")
        print(f"  TRAIN: Loss {avg_t_loss:.4f}, F1(C0) {avg_t_f1:.4f}")
        print(f"  VAL:   Loss {avg_v_loss:.4f}, F1(C0) {avg_v_f1:.4f}")

        if avg_v_f1 > best_val_f1:
            best_val_f1 = avg_v_f1
            model_path = os.path.join(models_dir, "best_within_user_gtcn.h5")
            print(f"  >>> New Best Val F1! Saving to {model_path}")
            model.save(model_path)

        gc.collect()

    return history


def main() -> None:
    """Main pipeline for Hybrid GTCN training and evaluation."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)

    configure_gpu(args.use_cpu)

    # Load data
    raw_feature_df_scaled = pd.read_csv(args.input_csv)
    print(f"Loaded input CSV: {args.input_csv}")
    print(f"Loaded data: {raw_feature_df_scaled.shape}")

    raw_feature_df_scaled = drop_zero_mi_columns(raw_feature_df_scaled, verbose=True)

    # Within-user split
    train_df, test_df = split_train_test_by_participant(
        raw_feature_df_scaled, train_frac=args.train_frac
    )
    n_users = train_df["participant_id"].nunique()
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Users: {n_users}")

    # Impute
    train_df, medians = impute_group_median_then_ffill(train_df, verbose=True)
    medians.to_csv(os.path.join(args.output_dir, "hybrid_rnn_medians.csv"), index=True)

    test_df = impute_test_with_medians_and_ffill(test_df, medians, verbose=True)

    # Z-normalize
    train_df, global_means = z_normalize_columns(train_df, DEFAULT_COLS_TO_SCALE, verbose=True)
    global_means.to_csv(os.path.join(args.output_dir, "global_means_hybrid_rnn.csv"), index=True)

    test_df = z_normalize_test_using_global_mean(test_df, global_means, verbose=True)

    # Feature dimensions
    feature_cols = [c for c in train_df.columns if c not in ["participant_id", "outcome"]]
    n_feature_cols = len(feature_cols)
    print(f"Number of features: {n_feature_cols}")

    # Tensor parameters
    L_CHUNK = 3967
    NUM_CHUNKS_TRAIN = 1
    NUM_CHUNKS_VAL = 4
    SENTINEL_VALUE = 999.0

    if not args.skip_training:
        # Build tensors
        X_tr, Y_tr, X_va, Y_va = prepare_within_user_tensors(
            train_df, test_df, L_CHUNK, NUM_CHUNKS_TRAIN, NUM_CHUNKS_VAL, SENTINEL_VALUE
        )
        print(f"Training tensor shape: {X_tr.shape}")
        print(f"Validation tensor shape: {X_va.shape}")

        # Build and compile model
        model = build_hybrid_gtcn(L_CHUNK, n_feature_cols)
        model.compile(optimizer="adam", loss=optimized_loss_fn, metrics=[optimized_f1_class0])

        # Save model summary
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        save_text_results(summary_text, args.output_dir, "hybrid_gtcn_model_summary.txt")

        # Train
        history = train_hybrid_model(model, X_tr, Y_tr, X_va, Y_va, args.epochs, models_dir)

        # Save training history
        history_path = os.path.join(args.output_dir, "within_user_training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f)

        # Find optimal threshold
        best_model = tf.keras.models.load_model(
            os.path.join(models_dir, "best_within_user_gtcn.h5"),
            custom_objects=get_custom_objects(), compile=False, safe_mode=False
        )
        opt_thresh, best_f1, fig_thresh = find_optimal_threshold(best_model, X_va, Y_va)
        threshold = opt_thresh
        save_figure(fig_thresh, args.output_dir, "hybrid_rnn_f1_vs_threshold.png")
        save_text_results(
            f"Optimal threshold: {opt_thresh}\nBest F1 (Class 0): {best_f1}",
            args.output_dir, "hybrid_rnn_optimal_threshold.txt"
        )
    else:
        best_model = tf.keras.models.load_model(
            os.path.join(models_dir, "best_within_user_gtcn.h5"),
            custom_objects=get_custom_objects(), compile=False, safe_mode=False
        )
        if args.threshold is None:
            raise ValueError("--threshold is required when using --skip_training (no tuning stage to derive it).")
        threshold = args.threshold

    # Evaluate on held-out data
    if os.path.exists(args.heldout_csv):
        print(f"\nLoading held-out data: {args.heldout_csv}")
        heldout_df = pd.read_csv(args.heldout_csv)

        heldout_df = impute_test_with_medians_and_ffill(
            heldout_df,
            medians if not args.skip_training else pd.read_csv(os.path.join(args.output_dir, "hybrid_rnn_medians.csv"), index_col=0),
            verbose=True
        )
        training_cols = list(train_df.columns) if not args.skip_training else None
        if training_cols:
            heldout_df = heldout_df[training_cols]
        heldout_df = z_normalize_test_using_global_mean(
            heldout_df,
            global_means if not args.skip_training else pd.read_csv(os.path.join(args.output_dir, "global_means_hybrid_rnn.csv"), index_col=0),
            verbose=True
        )

        heldout_df.to_csv(os.path.join(args.output_dir, "hybrid_heldout_scaled.csv"), index=False)

        NUM_CHUNKS = 4
        X_final, Y_final, heldout_pids = preprocess_held_out_data(heldout_df, L_CHUNK, NUM_CHUNKS)

        f1_val, metrics_text = run_final_test(best_model, X_final, Y_final, threshold=threshold, setup_name="Hybrid GTCN")
        plt.savefig(os.path.join(args.output_dir, "hybrid_rnn_confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()

        save_text_results(metrics_text, args.output_dir, "hybrid_rnn_heldout_metrics.txt")

        results_df = analyze_user_f1_distribution(
            best_model, X_final, Y_final, heldout_pids, threshold=threshold
        )
        plt.savefig(os.path.join(args.output_dir, "hybrid_rnn_f1_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close()

        results_df.to_csv(os.path.join(args.output_dir, "hybrid_f1_scores.csv"), index=False)
        print("Saved per-user F1 scores")

    print("\nHybrid RNN pipeline complete.")


if __name__ == "__main__":
    main()
