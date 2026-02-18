"""
CLI entry point for the UEMA Context Model pipeline.

Exposes each pipeline step as a subcommand. Run with:
    python -m cli.main <step> [options]
"""

import sys
import argparse


def main() -> None:
    """Parse subcommand and dispatch to the appropriate module."""
    parser = argparse.ArgumentParser(
        description="UEMA Context Model Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Pipeline Steps (run in order):
  1. import-prep         Import and prepare dataset
  2. compute-features    Compute raw features from compliance matrix
  3. feature-norm        Feature selection and normalization
  4. heldout-prep        Prepare held-out dataset
  5. general-rnn         Train/evaluate General GTCN model (Setup 1)
  6. hybrid-rnn          Train/evaluate Hybrid GTCN model (Setup 2)
  7. prep-withdrawn      Prepare withdrawn participant data
  8. withdrew-general     Evaluate General GTCN on withdrew participants
  9. withdrew-hybrid      Evaluate Hybrid GTCN on withdrew participants + random baseline
  10. survival            Survival analysis and statistical comparisons
  11. combine-results     Merge all result .txt files into full_analysis.txt

Example:
  python -m cli.main import-prep --data_dir /path/to/data --output_dir /path/to/output
  python -m cli.main general-rnn --data_dir /path/to/data --output_dir /path/to/output
""",
    )

    parser.add_argument(
        "step",
        choices=[
            "import-prep",
            "compute-features",
            "feature-norm",
            "heldout-prep",
            "general-rnn",
            "hybrid-rnn",
            "prep-withdrawn",
            "withdrew-general",
            "withdrew-hybrid",
            "survival",
            "combine-results",
        ],
        help="Pipeline step to execute.",
    )

    # Parse only the first argument to determine the step
    args, remaining = parser.parse_known_args()

    # Override sys.argv so the sub-module's argparse picks up remaining args
    sys.argv = [f"cli.main {args.step}"] + remaining

    if args.step == "import-prep":
        from src.import_prep_dataset import main as step_main
    elif args.step == "compute-features":
        from src.compute_raw_features import main as step_main
    elif args.step == "feature-norm":
        from src.feature_selection_normalization import main as step_main
    elif args.step == "heldout-prep":
        from src.held_out_data_prep import main as step_main
    elif args.step == "general-rnn":
        from src.general_rnn import main as step_main
    elif args.step == "hybrid-rnn":
        from src.hybrid_rnn import main as step_main
    elif args.step == "prep-withdrawn":
        from src.prep_withdrawn_data import main as step_main
    elif args.step == "withdrew-general":
        from src.withdrew_general_eval import main as step_main
    elif args.step == "withdrew-hybrid":
        from src.withdrew_hybrid_eval import main as step_main
    elif args.step == "survival":
        from src.survival_analysis import main as step_main
    elif args.step == "combine-results":
        from src.combine_results_txt import main as step_main
    else:
        parser.print_help()
        sys.exit(1)

    step_main()


if __name__ == "__main__":
    main()
