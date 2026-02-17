"""
Step 1: Import and prepare dataset.

Imports participant status data from TIME study, filters completed participants,
splits into training and holdout sets, loads compliance matrix CSVs, and saves
intermediate files for downstream processing.
"""

import os
import sys
import random
import datetime
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.helpers import load_comp_matrix


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Import and prepare TIME study dataset.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing TIME study data files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save intermediate output files.")
    parser.add_argument("--n_train", type=int, default=100,
                        help="Number of participants for the training set (default: 100).")
    parser.add_argument("--status_file", type=str, default="participant_status_tracking_v2.csv",
                        help="Name of the participant status CSV file.")
    parser.add_argument("--compliance_dir", type=str, default="compliance_matrix",
                        help="Subdirectory name containing per-user compliance matrix folders.")
    return parser.parse_args()


def load_participant_status(data_dir: str, status_file: str) -> pd.DataFrame:
    """Load and filter completed participants from the status tracking file.

    Args:
        data_dir: Root data directory.
        status_file: Name of the status CSV file.

    Returns:
        DataFrame with participant_id and status columns for completed participants.
    """
    filepath = os.path.join(data_dir, status_file)
    status_df = pd.read_csv(filepath)
    print(f"Loaded status file: {filepath}")
    print(f"Total participants: {len(status_df)}")

    # Filter completed participants
    status_df = status_df[status_df["Participant Status "] == "Completed"][
        ["Visualizer ID", "Participant Status "]
    ]
    status_df.rename(
        columns={"Visualizer ID": "participant_id", "Participant Status ": "status"},
        inplace=True,
    )
    status_df.reset_index(drop=True, inplace=True)
    status_df["participant_id"] = status_df["participant_id"] + "@timestudy_com"

    print(f"Completed participants: {len(status_df)}")
    return status_df


def split_train_holdout(
    completed_participants: List[str], n_train: int
) -> Tuple[List[str], List[str]]:
    """Split completed participants into training and holdout lists.

    Uses non-deterministic random sampling.

    Args:
        completed_participants: Sorted list of all completed participant IDs.
        n_train: Number of participants for training.

    Returns:
        Tuple of (training_list, holdout_list).
    """
    n_train = min(n_train, len(completed_participants))
    if len(completed_participants) == 0:
        return [], []

    training_list = random.sample(completed_participants, k=n_train)
    holdout_set = set(training_list)
    holdout_list = [p for p in completed_participants if p not in holdout_set]

    print(f"Training list size: {len(training_list)}")
    print(f"Holdout list size: {len(holdout_list)}")
    return training_list, holdout_list


def main() -> None:
    """Main pipeline for importing and preparing the dataset."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load participant status
    status_df = load_participant_status(args.data_dir, args.status_file)
    completed_participants = status_df["participant_id"].sort_values().tolist()
    print(f"Total completed participants: {len(completed_participants)}")

    # Split into training and holdout
    training_list, holdout_list = split_train_holdout(completed_participants, args.n_train)

    # Normalize training list
    normalized = []
    seen = set()
    for p in training_list:
        key = str(p).strip()
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    training_list = [p for p in normalized if p.endswith("@timestudy_com")]

    # Load compliance matrix for training participants
    compliance_dir = os.path.join(args.data_dir, args.compliance_dir)
    compliance_matrix = load_comp_matrix(training_list, compliance_dir)

    print(f"Compliance matrix shape: {compliance_matrix.shape}")
    if not compliance_matrix.empty:
        num_participants = compliance_matrix["Participant_ID"].nunique()
        print(f"Unique participants in compliance matrix: {num_participants}")

    # Remove unknown users
    compliance_matrix = compliance_matrix[
        compliance_matrix["Participant_ID"] != "unknown_user"
    ]

    # Save outputs
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    compliance_path = os.path.join(args.output_dir, f"compliance_matrix_{current_time}.csv")
    compliance_matrix.to_csv(compliance_path, index=False)
    print(f"Compliance matrix saved to {compliance_path}")

    training_path = os.path.join(args.output_dir, f"training_list_{current_time}.txt")
    with open(training_path, "w") as f:
        for item in training_list:
            f.write(f"{item}\n")
    print(f"Training list saved to {training_path}")

    holdout_path = os.path.join(args.output_dir, f"holdout_list_{current_time}.txt")
    with open(holdout_path, "w") as f:
        for item in holdout_list:
            f.write(f"{item}\n")
    print(f"Holdout list saved to {holdout_path}")


if __name__ == "__main__":
    main()
