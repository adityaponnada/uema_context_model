"""
Step 5: General GTCN model (Setup 1 - Between-user split).

Trains a Gated Temporal Convolutional Network using a between-user split:
10 random users for training, remaining 90 for validation.
Evaluates on held-out data and computes permutation importance.

Architecture: Causal Conv1D -> 3 Gated Conv Blocks (dilation 2,4,8) -> Dense sigmoid
"""

import os
import gc
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from src.helpers import (
    set_global_seed,
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
    process_and_pad,
    reshape_to_chunks,
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
        description="Train and evaluate the General GTCN model (Setup 1)."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save figures, results, and intermediate files.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Full path to processed features CSV file.")
    parser.add_argument("--heldout_csv", type=str, required=True,
                        help="Full path to processed held-out features CSV file.")
    parser.add_argument("--n_train_users", type=int, default=10,
                        help="Number of users for training (default: 10).")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20).")
    parser.add_argument("--use_cpu", action="store_true", default=True,
                        help="Use CPU only (default: True).")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for train/test split.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override decision threshold (if None, find optimal).")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only evaluate using existing model.")
    return parser.parse_args()


def split_train_test_by_users_random(
    df: pd.DataFrame, id_col: str = "participant_id",
    n_train_users: int = 10, random_state: int = None
):
    """Split DataFrame into train/test by randomly selecting train users.

    Args:
        df: Input DataFrame.
        id_col: Participant ID column.
        n_train_users: Number of users for training set.
        random_state: Random seed.

    Returns:
        Tuple of (train_df, test_df).
    """
    unique_ids = pd.Index(df[id_col].dropna().unique())
    n_unique = len(unique_ids)
    if n_train_users <= 0 or n_train_users >= n_unique:
        raise ValueError(f"n_train_users must be >0 and < {n_unique}")

    rng = np.random.default_rng(random_state)
    train_ids = rng.choice(unique_ids, size=n_train_users, replace=False)
    train_df = df[df[id_col].isin(train_ids)].reset_index(drop=True)
    test_df = df[~df[id_col].isin(train_ids)].reset_index(drop=True)
    return train_df, test_df


def build_general_gtcn(l_chunk: int, num_features: int, conv_filters: int = 8, kernel_size: int = 32):
    """Build the General GTCN model (Setup 1).

    Architecture: Input -> Masking -> Conv1D -> 3 Gated Conv Blocks -> Dropout -> Dense sigmoid

    Args:
        l_chunk: Input sequence length per chunk.
        num_features: Number of input features.
        conv_filters: Number of convolution filters.
        kernel_size: Kernel size for initial convolution.

    Returns:
        Compiled Keras Model.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Masking, Conv1D, multiply, Activation, Dropout, TimeDistributed, Dense
    )

    def gated_conv_block(x, dilation_rate, filters, k_size):
        conv_a = Conv1D(filters=filters, kernel_size=k_size, padding="causal",
                        dilation_rate=dilation_rate, activation=None)(x)
        conv_b = Conv1D(filters=filters, kernel_size=k_size, padding="causal",
                        dilation_rate=dilation_rate, activation="sigmoid")(x)
        gated = multiply([conv_a, conv_b])
        return Activation("relu")(gated)

    inputs = Input(shape=(l_chunk, num_features))
    x = Masking(mask_value=999.0)(inputs)
    x = Conv1D(filters=conv_filters, kernel_size=kernel_size, padding="causal", activation="relu")(x)
    x = gated_conv_block(x, dilation_rate=2, filters=conv_filters, k_size=2)
    x = gated_conv_block(x, dilation_rate=4, filters=conv_filters, k_size=2)
    x = gated_conv_block(x, dilation_rate=8, filters=conv_filters, k_size=2)
    x = Dropout(0.2)(x)
    outputs = TimeDistributed(Dense(1, activation="sigmoid"))(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_model(
    model, X_train_chunked, Y_train_chunked, X_test_chunked, Y_test_chunked,
    n_train: int, n_val: int, epochs: int, models_dir: str
):
    """Train the GTCN model with user-by-user batch processing.

    Args:
        model: Compiled Keras model.
        X_train_chunked: 4D training features.
        Y_train_chunked: 4D training labels.
        X_test_chunked: 4D validation features.
        Y_test_chunked: 4D validation labels.
        n_train: Number of training users.
        n_val: Number of validation users.
        epochs: Number of training epochs.
        models_dir: Directory to save best model.

    Returns:
        Training history dict.
    """
    history_log = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}
    best_val_f1 = -1.0

    print(f"\nTraining on {n_train} users | Validating on {n_val} users")

    for epoch in range(epochs):
        start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        # Training phase
        t_loss, t_f1 = 0.0, 0.0
        for u in range(n_train):
            X_u = tf.convert_to_tensor(X_train_chunked[u], dtype=tf.float32)
            Y_u = tf.convert_to_tensor(Y_train_chunked[u], dtype=tf.float32)
            res = model.train_on_batch(x=X_u, y=Y_u, return_dict=True)
            t_loss += float(res["loss"])
            t_f1 += float(res["optimized_f1_class0"])
            print(".", end="", flush=True)
            del X_u, Y_u
            if u % 2 == 0:
                gc.collect()

        # Validation phase
        v_loss, v_f1 = 0.0, 0.0
        for u in range(n_val):
            X_u = tf.convert_to_tensor(X_test_chunked[u], dtype=tf.float32)
            Y_u = tf.convert_to_tensor(Y_test_chunked[u], dtype=tf.float32)
            res = model.test_on_batch(x=X_u, y=Y_u, return_dict=True)
            v_loss += float(res["loss"])
            v_f1 += float(res["optimized_f1_class0"])
            del X_u, Y_u
            if u % 10 == 0:
                gc.collect()

        avg_t_f = t_f1 / n_train
        avg_v_f = v_f1 / n_val
        avg_t_l = t_loss / n_train
        avg_v_l = v_loss / n_val
        duration = time.time() - start_time

        history_log["train_loss"].append(float(avg_t_l))
        history_log["train_f1"].append(float(avg_t_f))
        history_log["val_loss"].append(float(avg_v_l))
        history_log["val_f1"].append(float(avg_v_f))

        print(f"\nDONE | Time: {duration:.1f}s | Train F1: {avg_t_f:.4f} | Val F1: {avg_v_f:.4f}")

        if avg_v_f > best_val_f1:
            best_val_f1 = avg_v_f
            model_path = os.path.join(models_dir, "best_model_safe.h5")
            model.save(model_path)
            print(f"  >>> New Best Val F1! Saved to {model_path}")

        gc.collect()

    return history_log


def main() -> None:
    """Main pipeline for General GTCN training and evaluation."""
    set_global_seed()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)

    configure_gpu(args.use_cpu)

    # Load and preprocess data
    raw_feature_df_scaled = pd.read_csv(args.input_csv)
    print(f"Loaded input CSV: {args.input_csv}")
    print(f"Loaded data: {raw_feature_df_scaled.shape}")

    # Drop zero MI columns
    raw_feature_df_scaled = drop_zero_mi_columns(
        raw_feature_df_scaled, verbose=True
    )

    # Save column names for later use
    col_path = os.path.join(args.output_dir, "processed_feature_columns.txt")
    with open(col_path, "w") as f:
        for col in raw_feature_df_scaled.columns:
            f.write(f"{col}\n")

    # Split into train/test by users
    train_df, test_df = split_train_test_by_users_random(
        raw_feature_df_scaled, n_train_users=args.n_train_users,
        random_state=args.random_state
    )
    n_train = train_df["participant_id"].nunique()
    n_val = test_df["participant_id"].nunique()
    print(f"Train users: {n_train}, Test users: {n_val}")

    # Impute training data
    train_df, medians = impute_group_median_then_ffill(train_df, verbose=True)

    # Save medians for reuse
    medians.to_csv(os.path.join(args.output_dir, "general_rnn_medians.csv"), index=True)
    print("Saved training medians")

    # Impute test data using training medians
    test_df = impute_test_with_medians_and_ffill(test_df, medians, verbose=True)

    # Z-normalize training data
    train_df, global_means = z_normalize_columns(train_df, DEFAULT_COLS_TO_SCALE, verbose=True)

    # Save global means
    global_means.to_csv(os.path.join(args.output_dir, "global_means_general_rnn.csv"), index=True)
    print("Saved global means")

    # Z-normalize test data using global means
    test_df = z_normalize_test_using_global_mean(test_df, global_means, verbose=True)

    # Get feature dimensions
    feature_cols = [c for c in train_df.columns if c not in ["participant_id", "outcome"]]
    n_feature_cols = len(feature_cols)
    print(f"Number of features: {n_feature_cols}")

    # Tensor parameters
    L_CHUNK = 3967
    NUM_CHUNKS = 4
    MAX_TIME_SLOTS = L_CHUNK * NUM_CHUNKS
    SENTINEL_VALUE = 999.0

    if not args.skip_training:
        # Process and pad data
        X_train_padded, Y_train_padded, train_user_ids = process_and_pad(
            train_df, MAX_TIME_SLOTS, SENTINEL_VALUE
        )
        X_test_padded, Y_test_padded, test_user_ids = process_and_pad(
            test_df, MAX_TIME_SLOTS, SENTINEL_VALUE
        )

        X_train_chunked, Y_train_chunked = reshape_to_chunks(
            X_train_padded, Y_train_padded, n_train, NUM_CHUNKS, L_CHUNK, n_feature_cols
        )
        X_test_chunked, Y_test_chunked = reshape_to_chunks(
            X_test_padded, Y_test_padded, n_val, NUM_CHUNKS, L_CHUNK, n_feature_cols
        )
        print(f"Training shape: {X_train_chunked.shape}")
        print(f"Validation shape: {X_test_chunked.shape}")

        # Build and compile model
        model = build_general_gtcn(L_CHUNK, n_feature_cols)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=optimized_loss_fn,
            metrics=[optimized_f1_class0],
            jit_compile=False,
        )

        # Save model summary
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        summary_text = "\n".join(summary_lines)
        print(summary_text)
        save_text_results(summary_text, args.output_dir, "general_gtcn_model_summary.txt")

        # Train
        history_log = train_model(
            model, X_train_chunked, Y_train_chunked,
            X_test_chunked, Y_test_chunked,
            n_train, n_val, args.epochs, models_dir
        )

        # Save training history
        history_path = os.path.join(args.output_dir, "general_training_history.json")
        with open(history_path, "w") as f:
            json.dump(history_log, f)
        print(f"Training history saved to {history_path}")

        # Find optimal threshold
        best_model = tf.keras.models.load_model(
            os.path.join(models_dir, "best_model_safe.h5"), compile=False
        )
        opt_thresh, best_f1, fig_thresh = find_optimal_threshold(
            best_model, X_test_chunked, Y_test_chunked
        )
        threshold = opt_thresh
        save_figure(fig_thresh, args.output_dir, "general_rnn_f1_vs_threshold.png")
        save_text_results(
            f"Optimal threshold: {opt_thresh}\nBest F1 (Class 0): {best_f1}",
            args.output_dir, "general_rnn_optimal_threshold.txt"
        )
    else:
        best_model = tf.keras.models.load_model(
            os.path.join(models_dir, "best_model_safe.h5"),
            custom_objects=get_custom_objects(), compile=False
        )
        if args.threshold is None:
            raise ValueError("--threshold is required when using --skip_training (no tuning stage to derive it).")
        threshold = args.threshold

    # Evaluate on held-out data
    if os.path.exists(args.heldout_csv):
        print(f"\nLoading held-out data: {args.heldout_csv}")
        heldout_df = pd.read_csv(args.heldout_csv)

        # Impute and normalize held-out data
        heldout_df = impute_test_with_medians_and_ffill(heldout_df, medians if not args.skip_training else pd.read_csv(os.path.join(args.output_dir, "general_rnn_medians.csv"), index_col=0), verbose=True)
        training_cols = list(train_df.columns) if not args.skip_training else None
        if training_cols:
            heldout_df = heldout_df[training_cols]
        heldout_df = z_normalize_test_using_global_mean(
            heldout_df,
            global_means if not args.skip_training else pd.read_csv(os.path.join(args.output_dir, "global_means_general_rnn.csv"), index_col=0),
            verbose=True
        )

        # Scale held-out data
        heldout_df.to_csv(os.path.join(args.output_dir, "general_heldout_scaled.csv"), index=False)

        # Preprocess held-out data
        X_final, Y_final, heldout_pids = preprocess_held_out_data(heldout_df, L_CHUNK, NUM_CHUNKS)

        # Run final test
        f1_val, metrics_text = run_final_test(best_model, X_final, Y_final, threshold=threshold, setup_name="General GTCN")
        plt.savefig(os.path.join(args.output_dir, "general_rnn_confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()

        save_text_results(metrics_text, args.output_dir, "general_rnn_heldout_metrics.txt")

        # Per-user F1 distribution
        results_df = analyze_user_f1_distribution(
            best_model, X_final, Y_final, heldout_pids, threshold=threshold
        )
        plt.savefig(os.path.join(args.output_dir, "general_rnn_f1_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close()

        results_df.to_csv(os.path.join(args.output_dir, "general_f1_scores.csv"), index=False)
        print("Saved per-user F1 scores")

    print("\nGeneral RNN pipeline complete.")


if __name__ == "__main__":
    main()
