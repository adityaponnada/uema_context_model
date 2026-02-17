"""
Step 7: Prepare withdrawn participant data.

Reads participant status file, filters withdrew participants,
loads their compliance matrix CSVs, and saves the assembled DataFrame.
"""

import os
import argparse
import datetime

import pandas as pd

from src.helpers import load_comp_matrix


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare withdrew participant compliance matrix."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing TIME study data.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output files.")
    parser.add_argument("--status_file", type=str, default="participant_status_tracking_v2.csv",
                        help="Name of the participant status CSV file.")
    parser.add_argument("--compliance_dir", type=str, default="compliance_matrix",
                        help="Subdirectory name containing per-user compliance matrix folders.")
    return parser.parse_args()


def main() -> None:
    """Main pipeline for preparing withdrew participant data."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load participant status
    status_path = os.path.join(args.data_dir, args.status_file)
    status_df = pd.read_csv(status_path)
    print(f"Loaded status file: {status_path}")

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
    compliance_dir = os.path.join(args.data_dir, args.compliance_dir)
    withdrew_df = load_comp_matrix(withdrew_ids, compliance_dir)
    print(f"Withdrew DataFrame shape: {withdrew_df.shape}")

    if not withdrew_df.empty:
        print(f"Unique participants: {withdrew_df['Participant_ID'].nunique()}")

        # Remove unknown users
        withdrew_df = withdrew_df[withdrew_df["Participant_ID"] != "unknown_user"]
        print(f"After removing unknown_user: {withdrew_df.shape}")
        print(f"Unique participants: {withdrew_df['Participant_ID'].nunique()}")

    # Save
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"withdrew_comp_mx_{current_time}.csv")
    withdrew_df.to_csv(out_path, index=False)
    print(f"Withdrew dataframe saved to {out_path}")


if __name__ == "__main__":
    main()
