"""
Step 4: Held-out data preparation.

Reads the holdout participant list, loads their compliance matrix CSVs
from per-user folders, and saves the assembled holdout DataFrame.
"""

import os
import argparse
import datetime

import pandas as pd

from src.helpers import load_comp_matrix


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare held-out dataset from compliance matrix CSVs."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing TIME study data.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output files.")
    parser.add_argument("--holdout_list", type=str, required=True,
                        help="Path to holdout_list .txt file (one participant_id per line).")
    parser.add_argument("--compliance_dir", type=str, default="compliance_matrix",
                        help="Subdirectory name containing per-user compliance matrix folders.")
    return parser.parse_args()


def main() -> None:
    """Main held-out data preparation pipeline."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Read holdout list
    with open(args.holdout_list, "r", encoding="utf-8") as f:
        holdout_list = [line.strip() for line in f if line.strip()]
    print(f"Read {len(holdout_list)} entries from holdout file")

    # Load compliance matrix for holdout participants
    compliance_dir = os.path.join(args.data_dir, args.compliance_dir)
    heldout_df = load_comp_matrix(holdout_list, compliance_dir)
    print(f"Held-out DataFrame shape: {heldout_df.shape}")

    # Save assembled holdout dataframe
    today = datetime.date.today().isoformat()
    out_fname = f"heldout_comp_mx_{today}.csv"
    out_path = os.path.join(args.output_dir, out_fname)

    if not heldout_df.empty:
        heldout_df.to_csv(out_path, index=False)
        print(f"Saved heldout dataframe to: {out_path} (rows={len(heldout_df)})")
    else:
        print("heldout_df is empty; nothing was saved.")


if __name__ == "__main__":
    main()
