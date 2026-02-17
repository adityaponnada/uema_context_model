"""
Step 3: Feature selection and normalization.

Reads raw features CSV, performs one-hot encoding on categorical variables,
applies fixed-max scaling on days_in_study, adds missingness indicators,
and saves processed features.

Supports three data modes: training, heldout, and withdrew.
"""

import os
import argparse

import numpy as np
import pandas as pd

from src.helpers import (
    one_hot_encode_features,
    fixed_max_scale_days_in_study,
    add_missingness_indicators,
    missing_value_table,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature selection and normalization for raw features."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing input data files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed output files.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Name of raw features CSV file (in data_dir).")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Name for output processed features CSV (saved in output_dir).")
    return parser.parse_args()


def main() -> None:
    """Main feature processing pipeline."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw features
    input_path = os.path.join(args.data_dir, args.input_csv)
    raw_feature_df = pd.read_csv(input_path)
    print(f"Raw feature DataFrame shape: {raw_feature_df.shape}")
    print(f"Unique participants: {raw_feature_df['participant_id'].nunique()}")

    # Remove unknown users
    raw_feature_df = raw_feature_df[
        raw_feature_df["participant_id"].astype(str).str.lower() != "unknown_user"
    ].reset_index(drop=True)

    # Display missing value summary
    mv_table = missing_value_table(raw_feature_df)
    print("\nMissing value summary:")
    print(mv_table)

    # One-hot encode categorical variables
    categorical_vars = ["time_of_day", "location_category", "wake_day_part"]
    raw_feature_df_encoded = one_hot_encode_features(raw_feature_df, categorical_vars)
    print(f"\nAfter one-hot encoding: {raw_feature_df_encoded.shape}")
    print(f"Columns: {raw_feature_df_encoded.columns.tolist()}")

    # Fixed-max scaling for days_in_study
    raw_feature_df_encoded = fixed_max_scale_days_in_study(
        raw_feature_df_encoded, days_col="days_in_study", fixed_max=365.0
    )
    print("Applied fixed-max scaling to days_in_study")

    # Add missingness indicators
    raw_feature_df_encoded = add_missingness_indicators(raw_feature_df_encoded)
    mi_cols = [c for c in raw_feature_df_encoded.columns if c.startswith("mi_")]
    print(f"Added {len(mi_cols)} missingness indicator columns")

    # Drop prompt_time_converted if present
    if "prompt_time_converted" in raw_feature_df_encoded.columns:
        raw_feature_df_encoded = raw_feature_df_encoded.drop(columns=["prompt_time_converted"])
        print("Dropped column 'prompt_time_converted'")

    print(f"\nFinal shape: {raw_feature_df_encoded.shape}")
    print(f"Unique participants: {raw_feature_df_encoded['participant_id'].nunique()}")

    # Save processed features
    output_path = os.path.join(args.output_dir, args.output_csv)
    raw_feature_df_encoded.to_csv(output_path, index=False)
    print(f"Processed features saved to {output_path}")


if __name__ == "__main__":
    main()
