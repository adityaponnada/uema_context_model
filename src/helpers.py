"""
Shared helper functions used across multiple pipeline modules.

Consolidates duplicate functions from:
- general_rnn / hybrid_rnn (imputation, normalization, model utils, evaluation)
- withdrew_general_eval / withdrew_hybrid_eval (simulation, burden analysis)
- held_out_data_prep / prep_withdrawn_data (data loading)
- feature_selection_normalization (encoding, scaling)
"""

import os
import re
import gc
import glob
import json
import time
import random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# Configuration
# =============================================================================

def _load_config() -> dict:
    """Load config.yaml from the project config/ directory."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int = None) -> int:
    """Set reproducibility seed for all relevant libraries.

    If seed is None, reads the default from config/config.yaml.

    Args:
        seed: Random seed value. Defaults to config value.

    Returns:
        The seed value that was set.
    """
    import tensorflow as tf

    if seed is None:
        cfg = _load_config()
        seed = cfg.get("seed", 42)

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Global seed set to {seed}")
    return seed


# =============================================================================
# Data Loading
# =============================================================================

def load_comp_matrix(
    participant_ids: List[str],
    base_dir: str,
    file_pattern: str = "uema_feature_mx_*.csv",
    chunk_size: int = 10000
) -> pd.DataFrame:
    """Load and concatenate compliance matrix CSVs for a list of participant IDs.

    For each participant folder in base_dir, reads all CSVs matching file_pattern
    and concatenates them into a single DataFrame.

    Args:
        participant_ids: List of participant ID strings (folder names).
        base_dir: Root directory containing participant folders.
        file_pattern: Glob pattern for CSV files within each folder.
        chunk_size: Number of rows per chunk when reading large CSVs.

    Returns:
        Concatenated DataFrame of all matched participant data.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    pid_set = set(str(x).strip() for x in participant_ids)
    all_entries = sorted(os.listdir(base_dir))
    matched_folders = [
        d for d in all_entries
        if os.path.isdir(os.path.join(base_dir, d)) and d in pid_set
    ]
    matched_folders.sort()

    out_frames = []
    for folder in matched_folders:
        folder_path = os.path.join(base_dir, folder)
        files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
        if not files:
            continue

        print(f"Reading participant: {folder} | files: {len(files)}")
        user_frames = []
        for fp in files:
            try:
                reader = pd.read_csv(fp, chunksize=chunk_size, low_memory=True)
                for chunk in reader:
                    user_frames.append(chunk)
            except pd.errors.EmptyDataError:
                continue
            except Exception as e:
                print(f"Failed reading {fp}: {e}")

        if user_frames:
            try:
                user_df = pd.concat(user_frames, ignore_index=True)
                out_frames.append(user_df)
            except ValueError:
                continue
            del user_frames
            gc.collect()

    if out_frames:
        result = pd.concat(out_frames, ignore_index=True)
    else:
        result = pd.DataFrame()

    print(f"Loaded compliance matrix: {result.shape}")
    return result


# =============================================================================
# Feature Processing
# =============================================================================

def drop_zero_mi_columns(
    df: pd.DataFrame,
    mi_prefix: str = "mi_",
    inplace: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """Drop missingness-indicator columns whose non-null values are all zero.

    Leaves columns that are entirely NaN.

    Args:
        df: Input DataFrame.
        mi_prefix: Prefix identifying missingness indicator columns.
        inplace: If True, modify df in-place.
        verbose: If True, print dropped columns.

    Returns:
        DataFrame with zero-value mi_ columns removed.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    if not inplace:
        df = df.copy()

    mi_cols = [c for c in df.columns if str(c).startswith(mi_prefix)]
    to_drop = []
    for c in mi_cols:
        non_null = df[c].dropna()
        if len(non_null) > 0 and (non_null == 0).all():
            to_drop.append(c)

    if to_drop:
        if verbose:
            print(f"Dropping {len(to_drop)} columns: {to_drop}")
        df.drop(columns=to_drop, inplace=True)

    return df


def one_hot_encode_features(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """One-hot encode specified categorical columns.

    Args:
        df: Input DataFrame.
        columns: List of column names to one-hot encode.

    Returns:
        DataFrame with one-hot encoded columns as 0/1 integers.
    """
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=columns, prefix=columns, drop_first=False)
    for col in df_encoded.columns:
        if any(col.startswith(f"{c}_") for c in columns):
            df_encoded[col] = df_encoded[col].astype(int)
    return df_encoded


def fixed_max_scale_days_in_study(
    df: pd.DataFrame,
    days_col: str = "days_in_study",
    fixed_max: float = 365.0,
    inplace: bool = False
) -> pd.DataFrame:
    """Scale days_in_study to [0,1] using a fixed maximum value.

    Args:
        df: Input DataFrame.
        days_col: Column name for days in study.
        fixed_max: Maximum value for scaling (default 365).
        inplace: If True, modify df in-place.

    Returns:
        DataFrame with days_col scaled to [0, 1].
    """
    if days_col not in df.columns:
        raise ValueError(f"days_col '{days_col}' not found in DataFrame")

    if not inplace:
        df = df.copy()

    coerced = pd.to_numeric(df[days_col], errors="coerce")
    df[days_col] = coerced.clip(lower=0, upper=float(fixed_max)) / float(fixed_max)
    df[days_col] = df[days_col].astype(float)
    return df


def add_missingness_indicators(
    df: pd.DataFrame,
    skip_cols: Optional[List[str]] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """Add binary missingness indicator columns (mi_*) for each feature column.

    Args:
        df: Input DataFrame.
        skip_cols: Columns to skip (default: participant_id, outcome, prompt_time_converted).
        inplace: If True, modify df in-place.

    Returns:
        DataFrame with added mi_ columns (1 if NaN, 0 otherwise).
    """
    if skip_cols is None:
        skip_cols = ["participant_id", "outcome", "prompt_time_converted"]
    if not inplace:
        df = df.copy()

    cols_to_process = [
        c for c in df.columns
        if c not in skip_cols and not str(c).startswith("mi_")
    ]
    for c in cols_to_process:
        df[f"mi_{c}"] = df[c].isna().astype(int)
    return df


def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage of missing values per feature column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing_% per column, sorted descending.
    """
    skip_cols = ["participant_id", "prompt_time_converted", "outcome"]
    cols = [col for col in df.columns if col.lower() not in skip_cols]
    missing_percent = df[cols].isnull().mean() * 100
    empty_percent = (df[cols] == "").mean() * 100
    total_missing_percent = missing_percent + empty_percent
    result = pd.DataFrame({"missing_%": total_missing_percent.round(2)})
    return result.sort_values("missing_%", ascending=False)


# =============================================================================
# Imputation
# =============================================================================

def impute_group_median_then_ffill(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    outcome_col: str = "outcome",
    mi_prefix: str = "mi_",
    inplace: bool = False,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing values per participant using group medians and forward-fill.

    Steps:
    1. Group by participant id.
    2. For numeric columns (excluding id, outcome, mi_*): compute group median.
    3. If first value is NaN, fill with group median (fallback to global median).
    4. Forward-fill remaining NaNs within each group.
    5. Return imputed DataFrame and global medians.

    Args:
        df: Input DataFrame.
        id_col: Participant ID column name.
        outcome_col: Outcome column name to skip.
        mi_prefix: Prefix for missingness indicator columns to skip.
        inplace: If True, modify df in-place.
        verbose: If True, print progress.

    Returns:
        Tuple of (imputed DataFrame, medians DataFrame).
    """
    if df is None:
        raise ValueError("df must be a pandas DataFrame")
    if not inplace:
        df = df.copy()

    exclude = {id_col, outcome_col}
    cols_to_process = [
        c for c in df.columns
        if c not in exclude and not str(c).startswith(mi_prefix)
    ]
    if verbose:
        print(f"Processing {len(cols_to_process)} columns (excluding {exclude} and prefix)")

    numeric_cols = df[cols_to_process].select_dtypes(include=[np.number]).columns.tolist()
    global_medians = df[numeric_cols].median() if numeric_cols else pd.Series(dtype=float)

    if id_col in df.columns and numeric_cols:
        grouped = df.groupby(id_col, sort=False)
        for pid, idx in grouped.groups.items():
            for col in numeric_cols:
                s = df.loc[idx, col]
                try:
                    gm = s.median()
                except Exception:
                    gm = np.nan

                if pd.isna(gm):
                    gm = global_medians.get(col, np.nan)

                # Fill first NaN with group/global median
                first_idx = idx[0] if hasattr(idx, '__getitem__') else idx.tolist()[0]
                if pd.isna(df.loc[first_idx, col]):
                    df.loc[first_idx, col] = gm

                # Forward-fill remaining NaNs
                df.loc[idx, col] = df.loc[idx, col].ffill()

                # Fill any remaining NaNs with global median
                df.loc[idx, col] = df.loc[idx, col].fillna(global_medians.get(col, 0))

    medians_df = pd.DataFrame(global_medians).T
    medians_df.index = ["global_median"]
    return df, medians_df


def impute_test_with_medians_and_ffill(
    df: pd.DataFrame,
    medians_df: pd.DataFrame,
    id_col: str = "participant_id",
    outcome_col: str = "outcome",
    mi_prefix: str = "mi_",
    inplace: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """Impute missing values in test data using provided medians and forward-fill.

    For numeric columns in medians_df, fills NaN with the median value.
    Then forward-fills remaining NaNs within each participant group.

    Args:
        df: Test DataFrame.
        medians_df: DataFrame with column medians (from training imputation).
        id_col: Participant ID column.
        outcome_col: Outcome column to skip.
        mi_prefix: Prefix for missingness indicator columns to skip.
        inplace: If True, modify df in-place.
        verbose: If True, print progress.

    Returns:
        Imputed DataFrame.
    """
    if df is None:
        raise ValueError("df must be a pandas DataFrame")
    if medians_df is None or medians_df.empty:
        raise ValueError("medians_df must be a non-empty DataFrame")

    if not inplace:
        df = df.copy()

    exclude = {id_col, outcome_col}
    cols_to_process = [
        c for c in df.columns
        if c not in exclude and not str(c).startswith(mi_prefix)
    ]
    if verbose:
        print(f"Imputing {len(cols_to_process)} columns using provided medians + ffill")

    # Extract median values from medians_df
    if medians_df.shape[0] == 1:
        median_map = medians_df.iloc[0].to_dict()
    elif "global_median" in medians_df.index:
        median_map = medians_df.loc["global_median"].to_dict()
    else:
        median_map = medians_df.iloc[0].to_dict()

    numeric_cols = df[cols_to_process].select_dtypes(include=[np.number]).columns.tolist()

    # Fill NaN with median for numeric columns
    for col in numeric_cols:
        if col in median_map and pd.notna(median_map[col]):
            # Fill first value per group if NaN
            if id_col in df.columns:
                grouped = df.groupby(id_col, sort=False)
                for pid, idx in grouped.groups.items():
                    first_idx = idx[0] if hasattr(idx, '__getitem__') else idx.tolist()[0]
                    if pd.isna(df.loc[first_idx, col]):
                        df.loc[first_idx, col] = median_map[col]

    # Forward-fill within participant groups
    if id_col in df.columns:
        df[numeric_cols] = df.groupby(id_col, sort=False)[numeric_cols].ffill()

    # Fill any remaining NaNs with median
    for col in numeric_cols:
        if col in median_map and pd.notna(median_map[col]):
            df[col] = df[col].fillna(median_map[col])

    return df


def impute_within_participant(
    df: pd.DataFrame,
    global_median: pd.DataFrame,
    id_col: str = "participant_id"
) -> pd.DataFrame:
    """Impute missing values by forward-filling within each participant.

    Skips columns starting with 'mi_' and excludes id_col and outcome columns.
    If the first observation for a participant is NaN, fills it using global_median.

    Args:
        df: Input features DataFrame.
        global_median: DataFrame/Series mapping column to median value.
        id_col: Participant ID column name.

    Returns:
        Imputed copy of df.
    """
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in df columns")

    df = df.copy()
    exclude = {id_col.lower(), "outcome", "outcomes"}
    cols_to_impute = [
        c for c in df.columns
        if c.lower() not in exclude and not c.lower().startswith("mi_")
    ]

    # Forward-fill within each participant
    try:
        df[cols_to_impute] = df.groupby(id_col, sort=False)[cols_to_impute].ffill()
    except Exception:
        df[cols_to_impute] = df.groupby(id_col, sort=False)[cols_to_impute].apply(
            lambda g: g.ffill()
        )

    def _get_global_median(col: str) -> float:
        """Extract median value for a column from global_median."""
        try:
            if isinstance(global_median, pd.Series):
                return global_median.get(col, np.nan)
            if isinstance(global_median, pd.DataFrame):
                if col in global_median.columns:
                    vals = global_median[col].dropna().values
                    if len(vals) > 0:
                        return float(vals[0])
            return np.nan
        except Exception:
            return np.nan

    # Fill remaining NaNs (first rows per participant) with global median
    for col in cols_to_impute:
        if df[col].isna().any():
            gm = _get_global_median(col)
            if pd.notna(gm):
                df[col] = df[col].fillna(gm)

    return df


# =============================================================================
# Normalization
# =============================================================================

# Default columns to z-normalize
DEFAULT_COLS_TO_SCALE = [
    "dist_from_home", "last_phone_usage", "closeness_to_sleep_time",
    "closeness_to_wake_time", "mims_5min", "completion_24h", "completion_1h",
    "time_between_prompts", "time_since_last_answered",
    "completion_since_wake", "completion_since_start",
]


def z_normalize_columns(
    df: pd.DataFrame,
    cols_to_scale: List[str],
    id_col: str = "participant_id",
    inplace: bool = False,
    ddof: int = 0,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Z-normalize specified columns per-participant.

    For each participant, subtract participant mean and divide by participant std.
    Groups with zero or undefined std use 1.0 to avoid division by zero.

    Args:
        df: Input DataFrame.
        cols_to_scale: Column names to normalize.
        id_col: Participant ID column.
        inplace: If True, modify df in-place.
        ddof: Degrees of freedom for std calculation.
        verbose: If True, print progress.

    Returns:
        Tuple of (normalized DataFrame, global means DataFrame).
    """
    if df is None:
        raise ValueError("df must be a pandas DataFrame")
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in DataFrame")

    if not inplace:
        df = df.copy()

    cols = [c for c in cols_to_scale if c in df.columns]
    missing = [c for c in cols_to_scale if c not in df.columns]
    if missing and verbose:
        print(f"Warning: columns not found and will be skipped: {missing}")

    if not cols:
        return df, pd.DataFrame()

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute global means BEFORE normalization
    global_means_series = df[cols].mean()
    means_df = pd.DataFrame(global_means_series).T
    means_df.index = ["global_mean"]

    # Group-wise z-normalization
    def _z_norm_group(group: pd.DataFrame) -> pd.DataFrame:
        for c in cols:
            mean_val = group[c].mean()
            std_val = group[c].std(ddof=ddof)
            if pd.isna(std_val) or std_val == 0:
                std_val = 1.0
            group[c] = (group[c] - mean_val) / std_val
        return group

    df = df.groupby(id_col, sort=False, group_keys=False).apply(_z_norm_group)
    return df, means_df


def z_normalize_test_using_global_mean(
    df: pd.DataFrame,
    global_means_df: pd.DataFrame,
    cols_to_scale: Optional[List[str]] = None,
    id_col: str = "participant_id",
    ddof: int = 0,
    inplace: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """Z-normalize test data using training global means and per-participant std.

    Args:
        df: Test DataFrame.
        global_means_df: DataFrame with training global means.
        cols_to_scale: Columns to normalize (default: DEFAULT_COLS_TO_SCALE).
        id_col: Participant ID column.
        ddof: Degrees of freedom for std calculation.
        inplace: If True, modify df in-place.
        verbose: If True, print progress.

    Returns:
        Normalized DataFrame.
    """
    if df is None:
        raise ValueError("df must be a pandas DataFrame")
    if global_means_df is None or global_means_df.empty:
        raise ValueError("global_means_df must be a non-empty DataFrame")
    if not inplace:
        df = df.copy()

    allowed_cols = DEFAULT_COLS_TO_SCALE
    if cols_to_scale is None:
        cols_to_scale = allowed_cols

    # Resolve global means mapping
    if global_means_df.shape[0] == 1:
        gm_map = global_means_df.iloc[0].to_dict()
    elif "global_mean" in global_means_df.index:
        gm_map = global_means_df.loc["global_mean"].to_dict()
    else:
        gm_map = global_means_df.iloc[0].to_dict()

    cols = [c for c in cols_to_scale if c in df.columns and c in allowed_cols]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _z_norm_test_group(group: pd.DataFrame) -> pd.DataFrame:
        for c in cols:
            gm = gm_map.get(c, 0.0)
            std_val = group[c].std(ddof=ddof)
            if pd.isna(std_val) or std_val == 0:
                std_val = 1.0
            group[c] = (group[c] - gm) / std_val
        return group

    df = df.groupby(id_col, sort=False, group_keys=False).apply(_z_norm_test_group)
    if verbose:
        print(f"Z-normalized {len(cols)} columns using global means")
    return df


def z_normalize_within_participant(
    df: pd.DataFrame,
    global_means: pd.DataFrame,
    id_col: str = "participant_id",
    cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Z-normalize features for withdrew/eval data using provided global means.

    Args:
        df: Input DataFrame.
        global_means: DataFrame with global mean values.
        id_col: Participant ID column.
        cols: Columns to normalize (default: DEFAULT_COLS_TO_SCALE).

    Returns:
        Normalized copy of df.
    """
    if cols is None:
        cols = DEFAULT_COLS_TO_SCALE

    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in df columns")

    df = df.copy()
    exclude = {id_col.lower(), "outcome", "outcomes"}
    cols_to_scale = [c for c in cols if c in df.columns and c.lower() not in exclude]

    def _get_mean(col: str) -> float:
        try:
            if isinstance(global_means, pd.Series):
                return global_means.get(col, np.nan)
            if isinstance(global_means, pd.DataFrame):
                if col in global_means.columns:
                    vals = global_means[col].dropna().values
                    if len(vals) > 0:
                        return float(vals[0])
            return np.nan
        except Exception:
            return np.nan

    for c in cols_to_scale:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _z_norm_group(group: pd.DataFrame) -> pd.DataFrame:
        for c in cols_to_scale:
            gm = _get_mean(c)
            if pd.isna(gm):
                gm = group[c].mean()
            std_val = group[c].std()
            if pd.isna(std_val) or std_val == 0:
                std_val = 1.0
            group[c] = (group[c] - gm) / std_val
        return group

    df = df.groupby(id_col, sort=False, group_keys=False).apply(_z_norm_group)
    return df


# =============================================================================
# TensorFlow Model Utilities
# =============================================================================

def configure_gpu(use_cpu_only: bool = True) -> None:
    """Configure TensorFlow GPU/CPU usage.

    Args:
        use_cpu_only: If True, disable GPU and use CPU only.
    """
    import tensorflow as tf

    if use_cpu_only:
        print("\n[STABILITY MODE] Disabling GPU to prevent Metal compilation hangs...")
        tf.config.set_visible_devices([], "GPU")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("[GPU MODE] Memory growth enabled.")
            except RuntimeError as e:
                print(f"Memory growth setting failed: {e}")


def get_sentinel_value() -> float:
    """Return the sentinel value used for padding."""
    return 999.0


def get_class_weights():
    """Return class weights tensor for weighted loss."""
    import tensorflow as tf
    return tf.constant([0.8, 0.2], dtype=tf.float32)


def mask_generator_fn(x):
    """Generate binary mask from input tensor (1 for data, 0 for padding)."""
    import tensorflow as tf
    return tf.cast(tf.not_equal(x[:, :, :1], 999.0), tf.float32)


def optimized_loss_fn(y_true, y_pred):
    """Weighted binary cross-entropy loss with sentinel masking."""
    import tensorflow as tf
    sentinel = 999.0
    class_weights = tf.constant([0.8, 0.2], dtype=tf.float32)

    mask = tf.cast(tf.not_equal(y_true, sentinel), tf.float32)
    y_p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    bce = -(y_true * tf.math.log(y_p) + (1.0 - y_true) * tf.math.log(1.0 - y_p))
    y_true_int = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_true_clipped = tf.clip_by_value(y_true_int, 0, 1)
    weights = tf.gather(class_weights, y_true_clipped)
    weights = tf.expand_dims(weights, axis=-1)
    return tf.reduce_sum(bce * weights * mask) / (tf.reduce_sum(mask) + 1e-7)


def optimized_f1_class0(y_true, y_pred):
    """F1 score for class 0 (non-response) with sentinel masking."""
    import tensorflow as tf
    sentinel = 999.0

    mask = tf.cast(tf.not_equal(y_true, sentinel), tf.float32)
    y_t = (1.0 - y_true) * mask
    y_p = (1.0 - tf.math.round(y_pred)) * mask
    tp = tf.reduce_sum(y_t * y_p)
    fp = tf.reduce_sum((1.0 - y_t) * y_p * mask)
    fn = tf.reduce_sum(y_t * (1.0 - y_p) * mask)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return 2 * ((precision * recall) / (precision + recall + 1e-7))


def get_custom_objects():
    """Return dict of custom objects needed to load saved models."""
    import tensorflow as tf
    return {
        "optimized_loss_fn": optimized_loss_fn,
        "optimized_f1_class0": optimized_f1_class0,
        "mask_generator_fn": mask_generator_fn,
        "tf": tf,
    }


# =============================================================================
# Tensor Preparation
# =============================================================================

def process_and_pad(
    df: pd.DataFrame,
    max_len: int,
    pad_val: float = 999.0,
    n_feature_cols: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert DataFrame to padded 3D arrays (N, max_len, features).

    Args:
        df: Input DataFrame with participant_id and outcome columns.
        max_len: Maximum sequence length for padding.
        pad_val: Value to use for padding.
        n_feature_cols: Expected number of feature columns (for validation).

    Returns:
        Tuple of (X_padded, Y_padded, participant_ids).
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    grouped = df.groupby("participant_id")
    X_list, Y_list, participant_ids = [], [], []

    for participant_id, group in grouped:
        participant_ids.append(participant_id)
        X_seq = group.drop(columns=["participant_id", "outcome"]).values
        X_list.append(X_seq)
        Y_seq = group["outcome"].values.astype("float32").reshape(-1, 1)
        Y_list.append(Y_seq)

    X_padded = pad_sequences(
        X_list, maxlen=max_len, padding="post", value=pad_val, dtype="float32"
    )
    Y_padded = pad_sequences(
        Y_list, maxlen=max_len, padding="post", value=pad_val, dtype="float32"
    )

    return X_padded, Y_padded, participant_ids


def reshape_to_chunks(
    X_padded: np.ndarray,
    Y_padded: np.ndarray,
    n_users: int,
    n_chunks: int,
    l_chunk: int,
    n_features: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape 3D padded tensors to 4D chunked tensors.

    Args:
        X_padded: Padded features array (N, L_max, D).
        Y_padded: Padded labels array (N, L_max, 1).
        n_users: Number of users.
        n_chunks: Number of chunks.
        l_chunk: Length of each chunk.
        n_features: Number of features.

    Returns:
        Tuple of (X_chunked, Y_chunked) with shape (N, n_chunks, l_chunk, D).
    """
    X_chunked = X_padded.reshape(n_users, n_chunks, l_chunk, n_features)
    Y_chunked = Y_padded.reshape(n_users, n_chunks, l_chunk, 1)
    return X_chunked, Y_chunked


def convert_to_4d_tensors(
    df: pd.DataFrame,
    l_chunk: int = 3967,
    num_chunks: int = 4,
    num_features: int = 40,
    sentinel_value: float = 999.0
):
    """Convert DataFrame into 4D TensorFlow tensors for withdrew user evaluation.

    Args:
        df: Input DataFrame with participant_id and outcome columns.
        l_chunk: Length of each chunk.
        num_chunks: Number of chunks.
        num_features: Number of feature columns.
        sentinel_value: Padding value.

    Returns:
        Tuple of (X_tensor, Y_tensor, participant_ids).
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    max_total_steps = l_chunk * num_chunks
    print(f"Processing {len(df['participant_id'].unique())} participants...")

    X_list, Y_list, participant_ids = [], [], []
    grouped = df.groupby("participant_id")

    for p_id, group in grouped:
        participant_ids.append(p_id)
        x_features = group.drop(columns=["participant_id", "outcome"]).values
        y_labels = group["outcome"].values.astype("float32").reshape(-1, 1)
        X_list.append(x_features)
        Y_list.append(y_labels)

    print(f"Padding sequences to {max_total_steps} steps...")
    X_padded = pad_sequences(
        X_list, maxlen=max_total_steps, padding="post",
        dtype="float32", value=sentinel_value
    )
    Y_padded = pad_sequences(
        Y_list, maxlen=max_total_steps, padding="post",
        dtype="float32", value=sentinel_value
    )

    num_participants = len(participant_ids)
    print("Reshaping into 4D tensors...")
    X_4d = X_padded.reshape(num_participants, num_chunks, l_chunk, num_features)
    Y_4d = Y_padded.reshape(num_participants, num_chunks, l_chunk, 1)

    X_tensor = tf.cast(X_4d, tf.float32)
    Y_tensor = tf.cast(Y_4d, tf.float32)

    print(f"Final X Shape: {X_tensor.shape}")
    print(f"Final Y Shape: {Y_tensor.shape}")
    return X_tensor, Y_tensor, participant_ids


def prepare_within_user_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    l_chunk: int,
    n_chunks_train: int,
    n_chunks_val: int,
    sentinel_value: float = 999.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare 4D tensors for within-user (Setup 2) training.

    Args:
        train_df: Training DataFrame (10% snippets per user).
        val_df: Validation DataFrame (90% per user).
        l_chunk: Chunk length.
        n_chunks_train: Number of chunks for training.
        n_chunks_val: Number of chunks for validation.
        sentinel_value: Padding value.

    Returns:
        Tuple of (X_train, Y_train, X_val, Y_val).
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    def _process_set(df: pd.DataFrame, n_chunks: int):
        grouped = df.groupby("participant_id")
        X_list, Y_list = [], []
        for _, group in grouped:
            X_seq = group.drop(columns=["participant_id", "outcome"]).values
            Y_seq = group["outcome"].values.astype("float32").reshape(-1, 1)
            X_list.append(X_seq)
            Y_list.append(Y_seq)

        max_pad_len = n_chunks * l_chunk
        X_padded = pad_sequences(
            X_list, maxlen=max_pad_len, padding="post",
            value=sentinel_value, dtype="float32"
        )
        Y_padded = pad_sequences(
            Y_list, maxlen=max_pad_len, padding="post",
            value=sentinel_value, dtype="float32"
        )
        X_4d = X_padded.reshape(len(X_list), n_chunks, l_chunk, X_padded.shape[-1])
        Y_4d = Y_padded.reshape(len(Y_list), n_chunks, l_chunk, 1)
        return X_4d, Y_4d

    print("Building Training Tensors...")
    X_train, Y_train = _process_set(train_df, n_chunks_train)
    print("Building Validation Tensors...")
    X_val, Y_val = _process_set(val_df, n_chunks_val)
    return X_train, Y_train, X_val, Y_val


def preprocess_held_out_data(
    df: pd.DataFrame,
    l_chunk: int = 3967,
    num_chunks: int = 4,
    sentinel_value: float = 999.0
):
    """Convert held-out DataFrame into 4D chunked tensors.

    Args:
        df: Scaled/imputed held-out DataFrame.
        l_chunk: Chunk length.
        num_chunks: Number of chunks.
        sentinel_value: Padding value.

    Returns:
        Tuple of (X_4d, Y_4d, participant_ids).
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    required_len = num_chunks * l_chunk
    print(f"Preparing held-out tensors for {df['participant_id'].nunique()} users...")

    grouped = df.groupby("participant_id")
    X_list, Y_list, p_ids = [], [], []

    for p_id, group in grouped:
        p_ids.append(p_id)
        X_seq = group.drop(columns=["participant_id", "outcome"]).values
        Y_seq = group["outcome"].values.astype("float32").reshape(-1, 1)
        X_list.append(X_seq)
        Y_list.append(Y_seq)

    X_padded = pad_sequences(
        X_list, maxlen=required_len, padding="post",
        value=sentinel_value, dtype="float32"
    )
    Y_padded = pad_sequences(
        Y_list, maxlen=required_len, padding="post",
        value=sentinel_value, dtype="float32"
    )

    n_users = len(p_ids)
    X_4d = X_padded.reshape(n_users, num_chunks, l_chunk, X_padded.shape[-1])
    Y_4d = Y_padded.reshape(n_users, num_chunks, l_chunk, 1)

    print(f"Padding/Chunking complete. Shape: {X_4d.shape}")
    return X_4d, Y_4d, p_ids


# =============================================================================
# Model Evaluation
# =============================================================================

def _predict_user(model, X_user: np.ndarray, sentinel_value: float = 999.0) -> np.ndarray:
    """Predict for a single user, handling both 3D and 4D model input shapes.

    The General GTCN (Setup 1) expects 3D input (batch, L_chunk, features),
    while the Hybrid GTCN (Setup 2) expects 4D input (batch, chunks, L_chunk, features).
    This function detects the model's expected input rank and iterates over
    chunks when feeding a 3D model with 4D data.

    Args:
        model: Trained Keras model.
        X_user: Single user's data with shape (chunks, L_chunk, features).
        sentinel_value: Padding value.

    Returns:
        Flattened prediction array across all chunks.
    """
    model_input_rank = len(model.input_shape)
    data_rank = len(X_user.shape)

    if model_input_rank == 3 and data_rank == 3:
        # 3D model, 4D data (user slice is 3D: chunks, L_chunk, features)
        # Feed each chunk separately and concatenate
        all_probs = []
        for c in range(X_user.shape[0]):
            chunk = X_user[c:c + 1]  # (1, L_chunk, features)
            probs = model.predict(chunk, verbose=0).flatten()
            all_probs.append(probs)
        return np.concatenate(all_probs)
    elif model_input_rank == 4 and data_rank == 3:
        # 4D model, 4D data (user slice is 3D: chunks, L_chunk, features)
        # Add batch dimension
        return model.predict(X_user[np.newaxis], verbose=0).flatten()
    else:
        # Fallback: add batch dim and predict
        return model.predict(X_user[np.newaxis], verbose=0).flatten()


def run_final_test(
    model,
    X_held_out: np.ndarray,
    Y_held_out: np.ndarray,
    threshold: float = 0.43,
    sentinel_value: float = 999.0,
    setup_name: str = "Model"
) -> float:
    """Run final evaluation on held-out data and print classification report.

    Args:
        model: Trained Keras model.
        X_held_out: 4D array (N, Chunks, L_chunk, Features).
        Y_held_out: 4D array (N, Chunks, L_chunk, 1).
        threshold: Decision threshold.
        sentinel_value: Padding value to mask.
        setup_name: Label for the model setup.

    Returns:
        F1 score for class 0.
    """
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_true, all_pred = [], []

    print(f"\nEvaluating Final Test Set ({setup_name}) with Threshold: {threshold}")

    for u in range(len(X_held_out)):
        y_prob = _predict_user(model, X_held_out[u], sentinel_value)
        y_true = Y_held_out[u].flatten()
        mask = y_true != sentinel_value
        preds = (y_prob[mask] > threshold).astype(int)
        all_true.extend(y_true[mask])
        all_pred.extend(preds)

    report_text = classification_report(
        all_true, all_pred,
        target_names=["Non-Response (C0)", "Response (C1)"]
    )

    header = (
        "\n" + "=" * 45 + "\n"
        f"      FINAL HELD-OUT TEST RESULTS ({setup_name})\n"
        + "=" * 45 + "\n"
    )
    print(header)
    print(report_text)

    final_f1_c0 = f1_score(all_true, all_pred, pos_label=0)
    print(f"Final F1-Score (Class 0): {final_f1_c0:.4f}")

    # Build full metrics text for saving
    report_dict = classification_report(
        all_true, all_pred,
        target_names=["Non-Response (C0)", "Response (C1)"],
        output_dict=True,
    )
    metrics_text = (
        f"HELD-OUT TEST RESULTS ({setup_name})\n"
        f"{'=' * 50}\n"
        f"Threshold: {threshold}\n\n"
        f"--- Class 0 (Non-Response) ---\n"
        f"Precision: {report_dict['Non-Response (C0)']['precision']:.4f}\n"
        f"Recall:    {report_dict['Non-Response (C0)']['recall']:.4f}\n"
        f"F1-Score:  {report_dict['Non-Response (C0)']['f1-score']:.4f}\n"
        f"Support:   {report_dict['Non-Response (C0)']['support']}\n\n"
        f"--- Class 1 (Response) ---\n"
        f"Precision: {report_dict['Response (C1)']['precision']:.4f}\n"
        f"Recall:    {report_dict['Response (C1)']['recall']:.4f}\n"
        f"F1-Score:  {report_dict['Response (C1)']['f1-score']:.4f}\n"
        f"Support:   {report_dict['Response (C1)']['support']}\n\n"
        f"--- Overall ---\n"
        f"Accuracy:       {report_dict['accuracy']:.4f}\n"
        f"Macro Avg F1:   {report_dict['macro avg']['f1-score']:.4f}\n"
        f"Weighted Avg F1:{report_dict['weighted avg']['f1-score']:.4f}\n"
    )

    cm = confusion_matrix(all_true, all_pred, normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2%", cmap="Greens",
        xticklabels=["Pred C0", "Pred C1"],
        yticklabels=["Actual C0", "Actual C1"]
    )
    plt.title(f"Confusion Matrix - {setup_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")

    return final_f1_c0, metrics_text


def analyze_user_f1_distribution(
    model,
    X_held_out: np.ndarray,
    Y_held_out: np.ndarray,
    participant_ids: List[str],
    threshold: float = 0.43,
    sentinel_value: float = 999.0
) -> pd.DataFrame:
    """Calculate per-user F1 score for class 0 and return as DataFrame.

    Args:
        model: Trained Keras model.
        X_held_out: 4D array.
        Y_held_out: 4D array.
        participant_ids: List of participant IDs.
        threshold: Decision threshold.
        sentinel_value: Padding value.

    Returns:
        DataFrame with participant_id and f1_score_c0 columns.
    """
    from sklearn.metrics import f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    user_scores = []
    print(f"Starting per-user evaluation for {len(participant_ids)} participants...")

    for u in range(len(X_held_out)):
        y_prob = _predict_user(model, X_held_out[u], sentinel_value)
        y_true = Y_held_out[u].flatten()
        mask = y_true != sentinel_value
        if not np.any(mask):
            continue
        y_true_real = y_true[mask]
        y_pred_real = (y_prob[mask] > threshold).astype(int)
        score = f1_score(y_true_real, y_pred_real, pos_label=0, zero_division=0)
        user_scores.append({
            "participant_id": participant_ids[u],
            "f1_score_c0": score,
        })

    df_scores = pd.DataFrame(user_scores)

    # Build descriptive stats text
    desc = df_scores["f1_score_c0"].describe()
    median_val = df_scores["f1_score_c0"].median()
    mean_val = df_scores["f1_score_c0"].mean()

    stats_text = (
        f"Count:   {desc['count']:.0f}\n"
        f"Mean:    {desc['mean']:.4f}\n"
        f"Median:  {median_val:.4f}\n"
        f"Std:     {desc['std']:.4f}\n"
        f"Min:     {desc['min']:.4f}\n"
        f"25%:     {desc['25%']:.4f}\n"
        f"50%:     {desc['50%']:.4f}\n"
        f"75%:     {desc['75%']:.4f}\n"
        f"Max:     {desc['max']:.4f}\n"
    )

    print("\n" + "=" * 30)
    print(" PER-USER F1 STATS (CLASS 0)")
    print("=" * 30)
    print(stats_text)

    # Plot distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df_scores["f1_score_c0"], bins=15, kde=True, color="teal", alpha=0.6)
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.3f}")
    plt.axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Median: {median_val:.3f}")
    plt.title("Per-User F1-Score Distribution (Class 0)")
    plt.xlabel("F1 Score")
    plt.ylabel("Count")
    plt.legend()

    return df_scores, stats_text


def find_optimal_threshold(
    model,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    sentinel_value: float = 999.0
) -> Tuple[float, float, Any]:
    """Find optimal threshold that maximizes F1 for class 0 on validation set.

    Args:
        model: Trained Keras model.
        X_val: Validation features.
        Y_val: Validation labels.
        sentinel_value: Padding value.

    Returns:
        Tuple of (optimal_threshold, best_f1_score, fig).
    """
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    all_probs, all_true = [], []
    print("Generating predictions for validation set...")

    for u in range(len(X_val)):
        y_prob = _predict_user(model, X_val[u], sentinel_value)
        y_true = Y_val[u].flatten()
        mask = y_true != sentinel_value
        all_true.extend(y_true[mask])
        all_probs.extend(y_prob[mask])

    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    # Precision-Recall curve targeting Class 0
    precision, recall, thresholds = precision_recall_curve(
        1.0 - all_true, 1.0 - all_probs
    )
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    best_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[best_idx]

    # Convert threshold back to class 1 probability space
    best_thresh_c0 = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    optimal_threshold = 1.0 - best_thresh_c0

    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Best F1 (Class 0): {max_f1:.4f}")

    # Plot F1 vs threshold
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores[:-1], color="darkorange", lw=2, label="F1 Score (Class 0)")
    ax.axvline(best_thresh_c0, color="red", linestyle="--", alpha=0.6)
    ax.scatter([best_thresh_c0], [max_f1], color="red", s=100, zorder=5, label="Optimal Point")
    ax.annotate(
        f"Max F1: {max_f1:.4f}\nOptimal Thresh (Busy): {best_thresh_c0:.4f}\nOptimal Thresh (Resp): {optimal_threshold:.4f}",
        xy=(best_thresh_c0, max_f1),
        xytext=(best_thresh_c0 + 0.05, max_f1 - 0.1),
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )
    ax.set_title("Class 0 (Non-Response) F1-Score vs. Threshold", fontsize=14)
    ax.set_xlabel("Threshold for Non-Response Probability", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    return optimal_threshold, max_f1, fig


def calculate_permutation_importance(
    model,
    X_test_4d: np.ndarray,
    Y_test_4d: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.31,
    sentinel_value: float = 999.0
) -> pd.DataFrame:
    """Calculate permutation importance for GTCN features.

    Args:
        model: Trained GTCN model.
        X_test_4d: 4D array (N_users, Chunks, L_chunk, Features).
        Y_test_4d: 4D array (N_users, Chunks, L_chunk, 1).
        feature_names: List of feature column names.
        threshold: Decision threshold for class 0.
        sentinel_value: Padding value.

    Returns:
        DataFrame with feature importance scores sorted by importance.
    """
    from sklearn.metrics import f1_score

    print("\n" + "=" * 50)
    print(f"  FEATURE IMPORTANCE ANALYSIS (Threshold: {threshold})")
    print("=" * 50)

    # Baseline F1
    all_true, all_base_probs = [], []
    for u in range(len(X_test_4d)):
        X_u = X_test_4d[u]
        if hasattr(X_u, "numpy"):
            X_u = X_u.numpy()
        probs = _predict_user(model, X_u, sentinel_value)
        y_true = Y_test_4d[u]
        if hasattr(y_true, "numpy"):
            y_true = y_true.numpy()
        y_true = y_true.flatten()
        mask = y_true != sentinel_value
        all_true.extend(y_true[mask])
        all_base_probs.extend(probs[mask])

    baseline_f1 = f1_score(
        all_true,
        (np.array(all_base_probs) > threshold).astype(int),
        pos_label=0, zero_division=0
    )
    print(f"Baseline F1 (Class 0): {baseline_f1:.4f}")

    importance_results = []

    for f_idx in range(len(feature_names)):
        all_perm_probs = []
        for u in range(len(X_test_4d)):
            X_orig = X_test_4d[u]
            if hasattr(X_orig, "numpy"):
                X_orig = X_orig.numpy()
            X_perm = X_orig.copy()
            for c in range(X_perm.shape[0]):
                np.random.shuffle(X_perm[c, :, f_idx])

            probs = _predict_user(model, X_perm, sentinel_value)
            y_true = Y_test_4d[u]
            if hasattr(y_true, "numpy"):
                y_true = y_true.numpy()
            y_true = y_true.flatten()
            mask = y_true != sentinel_value
            all_perm_probs.extend(probs[mask])

        perm_f1 = f1_score(
            all_true,
            (np.array(all_perm_probs) > threshold).astype(int),
            pos_label=0, zero_division=0
        )
        importance = baseline_f1 - perm_f1
        importance_results.append({
            "feature": feature_names[f_idx],
            "importance": importance,
            "baseline_f1": baseline_f1,
            "permuted_f1": perm_f1,
        })
        print(f"  [{f_idx + 1}/{len(feature_names)}] {feature_names[f_idx]}: {importance:.4f}")

    df_imp = pd.DataFrame(importance_results).sort_values("importance", ascending=False)
    return df_imp


# =============================================================================
# Simulation & Burden Analysis
# =============================================================================

def calculate_burden_thresholds(
    X_tensor,
    Y_tensor,
    participant_ids: List[str],
    days_col: str = "days_in_study",
    feature_columns: Optional[List[str]] = None,
    sentinel: float = 999.0
) -> pd.DataFrame:
    """Calculate burden tolerance threshold and velocity for withdrew participants.

    Args:
        X_tensor: 4D Tensor (N, chunks, L_chunk, D).
        Y_tensor: 4D Tensor (N, chunks, L_chunk, 1).
        participant_ids: List of participant IDs.
        days_col: Column name for days in study.
        feature_columns: List of feature column names.
        sentinel: Padding value.

    Returns:
        DataFrame with burden thresholds per participant.
    """
    if isinstance(days_col, str):
        if feature_columns is None:
            raise ValueError("Must provide feature_columns to use a column name string.")
        try:
            days_idx = list(feature_columns).index(days_col)
            print(f"Resolved '{days_col}' to index {days_idx}")
        except ValueError:
            raise ValueError(f"Column '{days_col}' not found in feature_columns.")
    else:
        days_idx = days_col

    X_np = X_tensor.numpy() if hasattr(X_tensor, "numpy") else X_tensor
    Y_np = Y_tensor.numpy() if hasattr(Y_tensor, "numpy") else Y_tensor

    threshold_results = []
    print(f"Calculating thresholds for {len(participant_ids)} users...")

    for i, p_id in enumerate(participant_ids):
        x_user = X_np[i].reshape(-1, X_np.shape[-1])
        y_user = Y_np[i].flatten()
        valid_mask = np.abs(y_user - sentinel) > 0.1

        if not np.any(valid_mask):
            threshold_results.append({
                "participant_id": p_id,
                "burden_threshold": 0,
                "actual_days_in_study": 0,
                "burden_velocity": 0,
            })
            continue

        clean_x = x_user[valid_mask]
        clean_y = y_user[valid_mask]

        # Count intrusive pings (outcome == 0)
        burden_count = int(np.sum(clean_y == 0.0))

        # Get actual days in study from the days_in_study column
        days_vals = clean_x[:, days_idx]
        actual_data = days_vals[np.abs(days_vals - sentinel) > 0.1]
        if len(actual_data) > 0:
            # days_in_study is already scaled to [0,1] with max=365
            actual_days = float(np.max(actual_data)) * 365.0
        else:
            actual_days = 0.0

        velocity = burden_count / actual_days if actual_days > 0 else 0.0

        threshold_results.append({
            "participant_id": p_id,
            "burden_threshold": burden_count,
            "actual_days_in_study": actual_days,
            "burden_velocity": velocity,
        })

    return pd.DataFrame(threshold_results)


def run_zero_shot_simulation(
    model_path: str,
    X_tensor,
    Y_tensor,
    participant_ids: List[str],
    threshold: float = 0.43,
    sentinel_value: float = 999.0,
    models_dir: str = "models"
) -> pd.DataFrame:
    """Simulate burden reduction on withdrew users using a pre-trained model.

    Args:
        model_path: Path to the .h5 model file.
        X_tensor: 4D Tensor (N, chunks, L_chunk, Features).
        Y_tensor: 4D Tensor (N, chunks, L_chunk, 1).
        participant_ids: List of participant IDs.
        threshold: Decision threshold.
        sentinel_value: Padding value.
        models_dir: Directory containing model files.

    Returns:
        Tuple of (DataFrame with simulation results, overall classification metrics text).
    """
    from tensorflow.keras.models import load_model
    from sklearn.metrics import f1_score, classification_report

    full_model_path = os.path.join(models_dir, model_path) if not os.path.isabs(model_path) else model_path
    print(f"\nLoading model: {full_model_path}")
    model = load_model(full_model_path, custom_objects=get_custom_objects(), compile=False)

    simulation_results = []
    all_true_agg, all_pred_agg = [], []

    print(f"Simulating pings for {len(participant_ids)} participants...")

    for i, p_id in enumerate(participant_ids):
        X_user = X_tensor[i]
        if hasattr(X_user, "numpy"):
            X_user = X_user.numpy()
        probs = _predict_user(model, X_user, sentinel_value)
        y_true = Y_tensor[i].numpy().flatten() if hasattr(Y_tensor[i], "numpy") else Y_tensor[i].flatten()
        valid_mask = y_true != sentinel_value
        y_true_real = y_true[valid_mask]
        y_prob_real = probs[valid_mask]

        pings_sent_mask = y_prob_real > threshold
        y_pred_real = pings_sent_mask.astype(int)

        user_f1 = f1_score(y_true_real, y_pred_real, pos_label=0, zero_division=0)

        model_intrusive_pings = np.sum((y_true_real == 0.0) & pings_sent_mask)
        successful_blocks = np.sum((y_true_real == 0.0) & ~pings_sent_mask)
        total_busy_moments = np.sum(y_true_real == 0.0)

        # Recall for class 1: TP(C1) / (TP(C1) + FN(C1))
        total_response = np.sum(y_true_real == 1.0)
        correct_response = np.sum((y_true_real == 1.0) & (y_pred_real == 1))
        user_recall_c1 = correct_response / total_response if total_response > 0 else 0.0

        all_true_agg.extend(y_true_real)
        all_pred_agg.extend(y_pred_real)

        reduction_rate = successful_blocks / total_busy_moments if total_busy_moments > 0 else 0

        simulation_results.append({
            "participant_id": p_id,
            "total_busy_moments": int(total_busy_moments),
            "model_intrusive_pings": int(model_intrusive_pings),
            "successful_blocks": int(successful_blocks),
            "reduction_rate": float(reduction_rate),
            "f1_score_c0": float(user_f1),
            "recall_class_1": float(user_recall_c1),
        })

    df_sim = pd.DataFrame(simulation_results)
    overall_f1 = f1_score(all_true_agg, all_pred_agg, pos_label=0, zero_division=0)
    print(f"Overall F1 (Class 0): {overall_f1:.4f}")
    print(f"Mean Reduction Rate: {df_sim['reduction_rate'].mean() * 100:.2f}%")

    # Build overall classification metrics
    report_dict = classification_report(
        all_true_agg, all_pred_agg,
        target_names=["Non-Response (C0)", "Response (C1)"],
        output_dict=True,
    )
    overall_metrics_text = (
        f"--- Class 0 (Non-Response) ---\n"
        f"Precision: {report_dict['Non-Response (C0)']['precision']:.4f}\n"
        f"Recall:    {report_dict['Non-Response (C0)']['recall']:.4f}\n"
        f"F1-Score:  {report_dict['Non-Response (C0)']['f1-score']:.4f}\n"
        f"Support:   {report_dict['Non-Response (C0)']['support']}\n\n"
        f"--- Class 1 (Response) ---\n"
        f"Precision: {report_dict['Response (C1)']['precision']:.4f}\n"
        f"Recall:    {report_dict['Response (C1)']['recall']:.4f}\n"
        f"F1-Score:  {report_dict['Response (C1)']['f1-score']:.4f}\n"
        f"Support:   {report_dict['Response (C1)']['support']}\n\n"
        f"--- Overall ---\n"
        f"Accuracy:        {report_dict['accuracy']:.4f}\n"
        f"Macro Avg F1:    {report_dict['macro avg']['f1-score']:.4f}\n"
        f"Weighted Avg F1: {report_dict['weighted avg']['f1-score']:.4f}\n"
    )
    print(overall_metrics_text)

    return df_sim, overall_metrics_text


def calculate_study_extension(
    df_thresholds: pd.DataFrame,
    df_simulation: pd.DataFrame,
    model_name: str = "Model"
) -> pd.DataFrame:
    """Calculate projected study life for withdrew participants.

    Args:
        df_thresholds: DataFrame with burden_threshold and actual_days_in_study.
        df_simulation: DataFrame with model_intrusive_pings.
        model_name: Label for the model.

    Returns:
        DataFrame with projected study extension per participant.
    """
    df_combined = pd.merge(df_thresholds, df_simulation, on="participant_id")
    results = []
    print(f"Calculating study extension for {model_name}...")

    for _, row in df_combined.iterrows():
        p_id = row["participant_id"]
        threshold = row["burden_threshold"]
        actual_days = row["actual_days_in_study"]
        model_pings = row["model_intrusive_pings"]
        f1 = row["f1_score_c0"]
        recall_c1 = row.get("recall_class_1", np.nan)

        v_lazy = threshold / actual_days if actual_days > 0 else 0
        v_model = model_pings / actual_days if actual_days > 0 else 0

        if v_model == 0:
            projected_days = 365.0
        else:
            projected_days = threshold / v_model

        projected_days_capped = min(365.0, projected_days)
        stayed_full_study = 1 if projected_days_capped >= 365.0 else 0

        results.append({
            "participant_id": p_id,
            "threshold": threshold,
            "f1": f1,
            "recall_class_1": recall_c1,
            "actual_days": actual_days,
            "v_lazy": v_lazy,
            "v_model": v_model,
            "projected_days": projected_days_capped,
            "stayed_full_study": stayed_full_study,
        })

    return pd.DataFrame(results)


def simulate_random_baseline(
    df_thresholds: pd.DataFrame,
    Y_tensor,
    participant_ids: List[str],
    block_rate: float = 0.20,
    iterations: int = 50,
    sentinel: float = 999.0
) -> pd.DataFrame:
    """Simulate a random silencing baseline for withdrew users.

    Args:
        df_thresholds: DataFrame with burden thresholds.
        Y_tensor: 4D Tensor with ground truth labels.
        participant_ids: List of participant IDs.
        block_rate: Fraction of pings to randomly block.
        iterations: Number of Monte Carlo iterations.
        sentinel: Padding value.

    Returns:
        DataFrame with random baseline simulation results.
    """
    random_results = []
    Y_np = Y_tensor.numpy() if hasattr(Y_tensor, "numpy") else Y_tensor

    print(f"Running Monte Carlo Random Simulation ({iterations} iterations per user)...")

    for i, p_id in enumerate(participant_ids):
        y_user_flat = Y_np[i].flatten()
        valid_mask = y_user_flat != sentinel
        y_actual = y_user_flat[valid_mask]

        user_info = df_thresholds[df_thresholds["participant_id"] == p_id].iloc[0]
        threshold = user_info["burden_threshold"]
        actual_days = user_info["actual_days_in_study"]

        simulated_intrusive_counts = []
        for _ in range(iterations):
            send_mask = np.random.choice(
                [1, 0], size=len(y_actual), p=[1 - block_rate, block_rate]
            )
            intrusive_pings = np.sum((y_actual == 0.0) & (send_mask == 1))
            simulated_intrusive_counts.append(intrusive_pings)

        avg_random_intrusive = np.mean(simulated_intrusive_counts)
        v_random = avg_random_intrusive / actual_days if actual_days > 0 else 0

        if v_random == 0:
            projected_days = 365.0
        else:
            projected_days = threshold / v_random

        projected_days = min(365.0, projected_days)

        random_results.append({
            "participant_id": p_id,
            "actual_days": actual_days,
            "avg_random_intrusive_pings": avg_random_intrusive,
            "projected_days": projected_days,
            "stayed_full_study": 1 if projected_days >= 365.0 else 0,
        })

    return pd.DataFrame(random_results)


# =============================================================================
# Plotting Utilities
# =============================================================================

def plot_actual_vs_projected_density(
    df: pd.DataFrame,
    actual_col: str = "actual_days",
    projected_col: str = "projected_days",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4)
):
    """Create density plot comparing actual vs projected days.

    Args:
        df: DataFrame with actual and projected columns.
        actual_col: Column name for actual days.
        projected_col: Column name for projected days.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    actual = df[actual_col].dropna()
    projected = df[projected_col].dropna()

    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(actual, ax=ax, label="Actual", fill=True, alpha=0.4)
    sns.kdeplot(projected, ax=ax, label="Projected", fill=True, alpha=0.4)

    actual_mean = actual.mean()
    projected_mean = projected.mean()
    if np.isfinite(actual_mean):
        ax.axvline(actual_mean, color="C0", linestyle="--", linewidth=1.5)
    if np.isfinite(projected_mean):
        ax.axvline(projected_mean, color="C1", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Days")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    ax.legend()
    return fig


def save_figure(fig, output_dir: str, filename: str) -> None:
    """Save a matplotlib figure to the output directory.

    Args:
        fig: Matplotlib figure.
        output_dir: Directory to save figures.
        filename: File name for the figure.
    """
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {filepath}")
    plt.close(fig)


def plot_gtcn_tsne(
    model_path: str,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    model_name: str = "GTCN"
) -> Tuple[Any, str]:
    """Visualize GTCN latent space using t-SNE on held-out data.

    Loads the model, identifies the bottleneck layer (handling both Setup 1 flat
    and Setup 2 nested/TimeDistributed architectures), extracts embeddings, and
    produces a 2D t-SNE scatter plot colored by class label.

    Args:
        model_path: Full path to the saved .h5 model file.
        X_test: 4D array (N_users, Chunks, L_chunk, Features).
        Y_test: 4D array (N_users, Chunks, L_chunk, 1).
        model_name: Display name for the model in plot title.

    Returns:
        Tuple of (matplotlib Figure, summary text string).
    """
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    sentinel = get_sentinel_value()
    print(f"\n--- Processing Latent Space for: {model_name} ---")

    # Load the model using shared custom objects
    full_model = load_model(
        model_path, custom_objects=get_custom_objects(), compile=False
    )

    # Detect architecture: Setup 2 (Nested/TimeDistributed) vs Setup 1 (Flat)
    is_setup_2 = any(
        'TimeDistributed' in layer.__class__.__name__
        and isinstance(getattr(layer, 'layer', None), tf.keras.Model)
        for layer in full_model.layers
    )

    if is_setup_2:
        structure_label = "Setup 2 (Nested Hybrid)"
        print(f"Structure Detected: {structure_label}")
        td_layer = next(
            l for l in full_model.layers
            if 'TimeDistributed' in l.__class__.__name__
            and isinstance(l.layer, tf.keras.Model)
        )
        inner_model = td_layer.layer

        # Find inner bottleneck layer
        inner_bottleneck = None
        for layer in reversed(inner_model.layers):
            if 'activation' in layer.name or 'multiply' in layer.name:
                inner_bottleneck = layer.name
                break

        inner_embedding_model = Model(
            inputs=inner_model.input,
            outputs=inner_model.get_layer(inner_bottleneck).output
        )
        embedding_model = Model(
            inputs=full_model.input,
            outputs=tf.keras.layers.TimeDistributed(inner_embedding_model)(full_model.input)
        )
        bottleneck_name = inner_bottleneck
        print(f"Targeting Inner Bottleneck: {inner_bottleneck}")
    else:
        structure_label = "Setup 1 (Flat Depth)"
        print(f"Structure Detected: {structure_label}")
        layer_name = None
        for layer in reversed(full_model.layers):
            if ('activation' in layer.name or 'multiply' in layer.name) \
                    and 'time_distributed' not in layer.name:
                layer_name = layer.name
                break

        embedding_model = Model(
            inputs=full_model.input,
            outputs=full_model.get_layer(layer_name).output
        )
        bottleneck_name = layer_name
        print(f"Targeting Bottleneck Layer: {layer_name}")

    # Extract embeddings per user
    all_embeddings, all_labels = [], []
    print("Extracting embeddings (flattening temporal dimension)...")

    for u in range(len(X_test)):
        emb = embedding_model.predict(X_test[u:u + 1], verbose=0)
        y_true = Y_test[u]
        if hasattr(y_true, "numpy"):
            y_true = y_true.numpy()
        y_true = y_true.flatten()
        mask = y_true != sentinel

        emb_flat = emb.reshape(-1, emb.shape[-1])
        all_embeddings.append(emb_flat[mask])
        all_labels.append(y_true[mask])

    X_emb = np.vstack(all_embeddings)
    y_emb = np.concatenate(all_labels)

    # Downsample and run t-SNE
    sample_size = min(5000, len(X_emb))
    print(f"Running t-SNE on {sample_size} valid samples...")
    idx = np.random.choice(len(X_emb), sample_size, replace=False)

    tsne = TSNE(
        n_components=2, perplexity=40, n_iter=1000,
        random_state=42, init='pca', learning_rate='auto'
    )
    X_2d = tsne.fit_transform(X_emb[idx])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    df_plot = pd.DataFrame({
        'Dim 1': X_2d[:, 0],
        'Dim 2': X_2d[:, 1],
        'State': [
            'Available (C1)' if l == 1 else 'Busy (C0)' for l in y_emb[idx]
        ]
    })

    sns.scatterplot(
        data=df_plot, x='Dim 1', y='Dim 2', hue='State',
        palette={'Available (C1)': 'royalblue', 'Busy (C0)': 'darkorange'},
        alpha=0.6, s=30, edgecolor=None, ax=ax
    )
    ax.set_title(f"Latent Space: {model_name}\n(Unseen Users)", fontsize=14)
    ax.grid(alpha=0.2)

    # Build summary text
    n_c0 = int(np.sum(y_emb[idx] == 0))
    n_c1 = int(np.sum(y_emb[idx] == 1))
    summary_text = (
        f"t-SNE Latent Space Summary for {model_name}\n"
        f"{'=' * 60}\n\n"
        f"Structure Detected: {structure_label}\n"
        f"Bottleneck Layer:   {bottleneck_name}\n"
        f"Total Valid Samples: {len(X_emb)}\n"
        f"t-SNE Sample Size:  {sample_size}\n\n"
        f"Class Distribution (sampled):\n"
        f"  Busy (C0):      {n_c0}\n"
        f"  Available (C1): {n_c1}\n"
    )
    print(summary_text)

    return fig, summary_text


def save_text_results(content: str, output_dir: str, filename: str) -> None:
    """Save text results to a .txt file in the output directory.

    Args:
        content: Text content to save.
        output_dir: Directory to save the file.
        filename: File name.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Results saved to: {filepath}")
