"""
Step 4: Held-out data preparation.

Reads the holdout participant list, loads their compliance matrix CSVs
from per-user folders, and saves the assembled holdout DataFrame.
"""

import os
import argparse

import pandas as pd

from src.helpers import load_comp_matrix


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare held-out dataset from compliance matrix CSVs."
    )
    parser.add_argument("--holdout_list", type=str, required=True,
                        help="Full path to holdout_list .txt file (one participant_id per line).")
    parser.add_argument("--compliance_dir", type=str, required=True,
                        help="Full path to directory containing per-user compliance matrix folders.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Full path for the output held-out CSV file.")
    return parser.parse_args()


def main() -> None:
    """Main held-out data preparation pipeline."""
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Read holdout list
    with open(args.holdout_list, "r", encoding="utf-8") as f:
        holdout_list = [line.strip() for line in f if line.strip()]
    print(f"Read {len(holdout_list)} entries from holdout file: {args.holdout_list}")

    # Load compliance matrix for holdout participants
    heldout_df = load_comp_matrix(holdout_list, args.compliance_dir)
    print(f"Held-out DataFrame shape: {heldout_df.shape}")

    # Save assembled holdout dataframe
    if not heldout_df.empty:
        heldout_df.to_csv(args.output_csv, index=False)
        print(f"Saved heldout dataframe to: {args.output_csv} (rows={len(heldout_df)})")
    else:
        print("heldout_df is empty; nothing was saved.")


if __name__ == "__main__":
    main()
