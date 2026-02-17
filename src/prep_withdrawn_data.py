"""
Step 7: Prepare withdrawn participant data.

Reads participant status file, filters withdrew participants,
loads their compliance matrix CSVs, and saves the assembled DataFrame.
"""

import os
import argparse

import pandas as pd

from src.helpers import load_comp_matrix


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare withdrew participant compliance matrix."
    )
    parser.add_argument("--status_csv", type=str, required=True,
                        help="Full path to participant status tracking CSV file.")
    parser.add_argument("--compliance_dir", type=str, required=True,
                        help="Full path to directory containing per-user compliance matrix folders.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Full path for the output withdrew compliance matrix CSV file.")
    return parser.parse_args()


def main() -> None:
    """Main pipeline for preparing withdrew participant data."""
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Load participant status
    status_df = pd.read_csv(args.status_csv)
    print(f"Loaded status file: {args.status_csv}")

    # Filter withdrew participants
    status_df = status_df[status_df["Participant Status "] == "Withdrew"][
        ["Visualizer ID", "Participant Status "]
    ]
    status_df.rename(
        columns={"Visualizer ID": "participant_id", "Participant Status ": "status"},
        inplace=True,
    )
    status_df.reset_index(drop=True, inplace=True)
    status_df["participant_id"] = status_df["participant_id"] + "@timestudy_com"
    print(f"Withdrew participants: {len(status_df)}")

    withdrew_ids = status_df["participant_id"].tolist()

    # Load compliance matrix for withdrew participants
    withdrew_df = load_comp_matrix(withdrew_ids, args.compliance_dir)
    print(f"Withdrew DataFrame shape: {withdrew_df.shape}")

    if not withdrew_df.empty:
        print(f"Unique participants: {withdrew_df['Participant_ID'].nunique()}")

        # Remove unknown users
        withdrew_df = withdrew_df[withdrew_df["Participant_ID"] != "unknown_user"]
        print(f"After removing unknown_user: {withdrew_df.shape}")
        print(f"Unique participants: {withdrew_df['Participant_ID'].nunique()}")

    # Save
    withdrew_df.to_csv(args.output_csv, index=False)
    print(f"Withdrew dataframe saved to {args.output_csv}")


if __name__ == "__main__":
    main()
