"""
Step 8: Evaluate General GTCN model (Setup 1) on withdrew participants.

Loads withdrew participant features, imputes and normalizes using training
statistics, converts to 4D tensors, calculates burden thresholds,
runs zero-shot simulation, and computes study extension projections.
"""

import os
import random
import argparse

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
    configure_gpu,
    impute_within_participant,
    z_normalize_within_participant,
    convert_to_4d_tensors,
    calculate_burden_thresholds,
    run_zero_shot_simulation,
    calculate_study_extension,
    plot_actual_vs_projected_density,
    save_figure,
    save_text_results,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate General GTCN (Setup 1) on withdrew participants."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation outputs.")
    parser.add_argument("--withdrew_csv", type=str, required=True,
                        help="Full path to processed withdrew features CSV file.")
    parser.add_argument("--medians_csv", type=str, required=True,
                        help="Full path to training medians CSV file.")
    parser.add_argument("--global_means_csv", type=str, required=True,
                        help="Full path to training global means CSV file.")
    parser.add_argument("--column_list", type=str, required=True,
                        help="Full path to feature column list .txt file.")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Full path to trained model .h5 file.")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Decision threshold from model tuning stage.")
    parser.add_argument("--use_cpu", action="store_true", default=True,
                        help="Use CPU only.")
    return parser.parse_args()


def main() -> None:
    """Main evaluation pipeline for General GTCN on withdrew data."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    configure_gpu(args.use_cpu)

    # Load data
    withdrew_features = pd.read_csv(args.withdrew_csv)
    print(f"Loaded withdrew CSV: {args.withdrew_csv}")
    print(f"Withdrew features shape: {withdrew_features.shape}")

    # Load training statistics
    global_median = pd.read_csv(args.medians_csv)
    print(f"Loaded medians: {args.medians_csv} {global_median.shape}")

    global_means = pd.read_csv(args.global_means_csv)
    print(f"Loaded global means: {args.global_means_csv} {global_means.shape}")

    # Get max days per user before imputation
    max_days_df = withdrew_features.groupby("participant_id")["days_in_study"].max().reset_index()
    max_days_df.rename(columns={"days_in_study": "max_days_in_study"}, inplace=True)

    # Impute and normalize
    withdrew_features = impute_within_participant(withdrew_features, global_median)
    withdrew_features = z_normalize_within_participant(withdrew_features, global_means)

    # Filter to training columns
    col_path = args.column_list
    with open(col_path, "r") as f:
        column_list = [line.strip() for line in f if line.strip()]
    withdrew_features = withdrew_features[column_list]
    print(f"Filtered to {len(column_list)} columns")
    print(f"Unique participants: {withdrew_features['participant_id'].nunique()}")

    # Convert to 4D tensors
    feature_cols = [c for c in withdrew_features.columns if c not in ["participant_id", "outcome"]]
    num_features = len(feature_cols)
    X_withdrawn, Y_withdrawn, p_ids = convert_to_4d_tensors(
        withdrew_features, num_features=num_features
    )

    # Calculate burden thresholds
    my_features = withdrew_features.drop(columns=["participant_id", "outcome"]).columns.tolist()
    df_breaking_points = calculate_burden_thresholds(
        X_withdrawn, Y_withdrawn, p_ids,
        days_col="days_in_study", feature_columns=my_features
    )
    df_breaking_points.to_csv(
        os.path.join(args.output_dir, "withdrawn_user_thresholds_s1.csv"), index=False
    )
    print("Saved burden thresholds")

    # Run zero-shot simulation
    models_dir = os.path.dirname(args.model_file)
    df_sim = run_zero_shot_simulation(
        args.model_file, X_withdrawn, Y_withdrawn, p_ids,
        threshold=args.threshold, models_dir=models_dir
    )
    df_sim.to_csv(os.path.join(args.output_dir, "withdrawn_user_simulation_setup1.csv"), index=False)

    # Print simulation stats
    results_text = (
        f"Setup 1 Simulation Results\n"
        f"{'=' * 40}\n"
        f"Mean reduction rate: {df_sim['reduction_rate'].mean() * 100:.2f}%\n"
        f"Std reduction rate: {df_sim['reduction_rate'].std() * 100:.2f}%\n"
        f"Mean F1 (Class 0): {df_sim['f1_score_c0'].mean():.4f}\n"
        f"Std F1 (Class 0): {df_sim['f1_score_c0'].std():.4f}\n"
    )
    print(results_text)
    save_text_results(results_text, args.output_dir, "general_rnn_simulation_results.txt")

    # Plot reduction rate distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_sim["reduction_rate"] * 100, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    ax.axvline(df_sim["reduction_rate"].mean() * 100, color="red", linestyle="dashed",
               linewidth=2, label="Mean Reduction Rate")
    ax.axvline(df_sim["reduction_rate"].median() * 100, color="green", linestyle="dashed",
               linewidth=2, label="Median Reduction Rate")
    ax.set_title("Distribution of Burden Reduction Rates (Setup 1 Model)", fontsize=16)
    ax.set_xlabel("Burden Reduction Rate (%)", fontsize=14)
    ax.set_ylabel("Number of Participants", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.75)
    save_figure(fig, args.output_dir, "general_rnn_burden_reduction_distribution.png")

    # Calculate study extension
    df_extension = calculate_study_extension(df_breaking_points, df_sim, model_name="Setup 1")
    df_extension.to_csv(
        os.path.join(args.output_dir, "withdrawn_user_study_extension_setup1.csv"), index=False
    )

    extension_text = (
        f"Setup 1 Study Extension Results\n"
        f"{'=' * 40}\n"
        f"Mean F1: {df_extension['f1'].mean():.4f}\n"
        f"Std F1: {df_extension['f1'].std():.4f}\n"
        f"Mean projected days: {df_extension['projected_days'].mean():.2f}\n"
        f"Participants surviving 365 days: {(df_extension['projected_days'] >= 365).sum()}\n"
    )
    print(extension_text)
    save_text_results(extension_text, args.output_dir, "general_rnn_study_extension_results.txt")

    # Plot study extension distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(df_extension["projected_days"], bins=20, color="lightcoral", edgecolor="black", alpha=0.7)
    ax2.axvline(df_extension["projected_days"].mean(), color="red", linestyle="dashed",
                linewidth=2, label="Mean Extension")
    ax2.axvline(df_extension["projected_days"].median(), color="green", linestyle="dashed",
                linewidth=2, label="Median Extension")
    ax2.axvline(365, color="blue", linestyle="dashed", linewidth=2, label="Full Study (365 days)")
    ax2.set_title("Distribution of Projected Study Extensions (Setup 1)", fontsize=16)
    ax2.set_xlabel("Projected Study Extension (days)", fontsize=14)
    ax2.set_ylabel("Number of Participants", fontsize=14)
    ax2.legend()
    save_figure(fig2, args.output_dir, "general_rnn_study_extension_distribution.png")

    # Actual vs projected density
    fig3 = plot_actual_vs_projected_density(
        df_extension, title="Actual vs Projected Days (Setup 1)"
    )
    save_figure(fig3, args.output_dir, "general_rnn_actual_vs_projected_density.png")

    print("\nSetup 1 evaluation complete.")


if __name__ == "__main__":
    main()
