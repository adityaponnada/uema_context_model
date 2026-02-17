# UEMA Context Model

Gated Temporal Convolutional Network (GTCN) pipeline for predicting EMA non-response and reducing survey burden in longitudinal studies.

## Setup

### Prerequisites
- Python 3.11.14

### Create virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** TensorFlow 2.15+ requires Python 3.11. On macOS with Apple Silicon, TensorFlow uses the Metal plugin for GPU acceleration (disabled by default in this pipeline for stability).

## Usage

All pipeline steps are run through the CLI entry point. All input files are specified as explicit full paths to CSV or TXT files.

```bash
python -m cli.main <step> [options]
```

### Pipeline Steps

Run these in order:

```bash
# 1. Import and prepare dataset
python -m cli.main import-prep \
  --data_dir /path/to/time_study_data \
  --output_dir /path/to/output

# 2. Compute raw features
python -m cli.main compute-features \
  --input_csv /path/to/output/sampled_compliance_matrix.csv \
  --output_csv /path/to/output/raw_features.csv

# 3. Feature selection and normalization (run 3x: training, heldout, withdrew)
python -m cli.main feature-norm \
  --data_dir /path/to/output \
  --output_dir /path/to/output \
  --input_csv raw_features.csv \
  --output_csv processed_features_rnn.csv

# 4. Prepare held-out dataset
python -m cli.main heldout-prep \
  --holdout_list /path/to/output/holdout_list.txt \
  --compliance_dir /path/to/time_study_data/compliance_matrix \
  --output_csv /path/to/output/heldout_comp_mx.csv

# 5. Train General GTCN model (Setup 1)
python -m cli.main general-rnn \
  --input_csv /path/to/output/processed_features_rnn.csv \
  --heldout_csv /path/to/output/processed_features_heldout.csv \
  --output_dir /path/to/output

# 6. Train Hybrid GTCN model (Setup 2)
python -m cli.main hybrid-rnn \
  --input_csv /path/to/output/processed_features_rnn.csv \
  --heldout_csv /path/to/output/processed_features_heldout.csv \
  --output_dir /path/to/output

# 7. Prepare withdrawn participant data
python -m cli.main prep-withdrawn \
  --status_csv /path/to/time_study_data/participant_status_tracking_v2.csv \
  --compliance_dir /path/to/time_study_data/compliance_matrix \
  --output_csv /path/to/output/withdrew_comp_mx.csv

# 8. Evaluate General GTCN on withdrew participants
python -m cli.main withdrew-general \
  --withdrew_csv /path/to/output/processed_features_withdrew.csv \
  --medians_csv /path/to/output/general_rnn_medians.csv \
  --global_means_csv /path/to/output/global_means_general_rnn.csv \
  --column_list /path/to/output/processed_feature_columns.txt \
  --threshold 0.43 \
  --output_dir /path/to/output

# 9. Evaluate Hybrid GTCN on withdrew participants + random baseline
python -m cli.main withdrew-hybrid \
  --withdrew_csv /path/to/output/processed_features_withdrew.csv \
  --medians_csv /path/to/output/hybrid_rnn_medians.csv \
  --global_means_csv /path/to/output/global_means_hybrid_rnn.csv \
  --column_list /path/to/output/processed_feature_columns.txt \
  --threshold 0.47 \
  --output_dir /path/to/output

# 10. Survival analysis and statistical comparisons
python -m cli.main survival \
  --random_csv /path/to/output/withdrawn_user_random_baseline_simulation.csv \
  --s1_extension_csv /path/to/output/withdrawn_user_study_extension_setup1.csv \
  --s2_extension_csv /path/to/output/withdrawn_user_study_extension_setup2.csv \
  --general_f1_csv /path/to/output/general_f1_scores.csv \
  --hybrid_f1_csv /path/to/output/hybrid_f1_scores.csv \
  --output_dir /path/to/output
```

### Step-specific help

```bash
python -m cli.main <step> --help
```

## Project Structure

```
uema_context_model/
├── src/
│   ├── __init__.py
│   ├── helpers.py                        # Shared helper functions
│   ├── import_prep_dataset.py            # Step 1: Dataset import/prep
│   ├── compute_raw_features.py           # Step 2: Raw feature computation
│   ├── feature_selection_normalization.py # Step 3: Feature selection/normalization
│   ├── held_out_data_prep.py             # Step 4: Held-out data prep
│   ├── general_rnn.py                    # Step 5: General GTCN (Setup 1)
│   ├── hybrid_rnn.py                     # Step 6: Hybrid GTCN (Setup 2)
│   ├── prep_withdrawn_data.py            # Step 7: Withdrawn participant prep
│   ├── withdrew_general_eval.py          # Step 8: Withdrew eval (Setup 1)
│   ├── withdrew_hybrid_eval.py           # Step 9: Withdrew eval (Setup 2) + random baseline
│   └── survival_analysis.py             # Step 10: Survival analysis
├── cli/
│   └── main.py                          # CLI entry point
├── config/
│   └── config.yaml                      # Default configuration values
├── models/                              # Saved .h5 model files
├── tests/
│   └── test_helpers.py                  # Unit tests
├── requirements.txt
├── README.md
└── .gitignore
```

## Outputs

All figures (.png), result summaries (.txt), and intermediate CSV files are saved to the user-specified `--output_dir`. Model `.h5` files are saved to the `models/` directory in the project root.
