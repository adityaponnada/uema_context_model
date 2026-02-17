"""Unit tests for src.helpers module."""

import numpy as np
import pandas as pd
import pytest

from src.helpers import (
    drop_zero_mi_columns,
    one_hot_encode_features,
    fixed_max_scale_days_in_study,
    add_missingness_indicators,
    missing_value_table,
    impute_group_median_then_ffill,
    impute_within_participant,
    z_normalize_columns,
    DEFAULT_COLS_TO_SCALE,
)


# ---------------------------------------------------------------------------
# drop_zero_mi_columns
# ---------------------------------------------------------------------------

class TestDropZeroMiColumns:
    def test_drops_all_zero_mi_columns(self):
        df = pd.DataFrame({
            "feature_a": [1, 2, 3],
            "mi_feature_a": [0, 0, 0],
            "mi_feature_b": [1, 0, 1],
        })
        result = drop_zero_mi_columns(df, verbose=False)
        assert "mi_feature_a" not in result.columns
        assert "mi_feature_b" in result.columns

    def test_keeps_all_nan_mi_columns(self):
        df = pd.DataFrame({
            "feature_a": [1, 2, 3],
            "mi_feature_a": [np.nan, np.nan, np.nan],
        })
        result = drop_zero_mi_columns(df, verbose=False)
        assert "mi_feature_a" in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"mi_x": [0, 0, 0]})
        result = drop_zero_mi_columns(df, inplace=False)
        assert "mi_x" in df.columns
        assert "mi_x" not in result.columns


# ---------------------------------------------------------------------------
# one_hot_encode_features
# ---------------------------------------------------------------------------

class TestOneHotEncodeFeatures:
    def test_encodes_categorical_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "red"], "val": [1, 2, 3]})
        result = one_hot_encode_features(df, columns=["color"])
        assert "color_red" in result.columns
        assert "color_blue" in result.columns
        assert result["color_red"].dtype == int


# ---------------------------------------------------------------------------
# fixed_max_scale_days_in_study
# ---------------------------------------------------------------------------

class TestFixedMaxScaleDays:
    def test_scales_to_0_1(self):
        df = pd.DataFrame({"days_in_study": [0, 182.5, 365]})
        result = fixed_max_scale_days_in_study(df)
        np.testing.assert_allclose(result["days_in_study"].values, [0.0, 0.5, 1.0])

    def test_clips_above_max(self):
        df = pd.DataFrame({"days_in_study": [400]})
        result = fixed_max_scale_days_in_study(df, fixed_max=365.0)
        assert result["days_in_study"].iloc[0] == 1.0

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"other": [1, 2]})
        with pytest.raises(ValueError):
            fixed_max_scale_days_in_study(df)


# ---------------------------------------------------------------------------
# add_missingness_indicators
# ---------------------------------------------------------------------------

class TestAddMissingnessIndicators:
    def test_adds_mi_columns(self):
        df = pd.DataFrame({
            "participant_id": ["a", "a"],
            "outcome": [0, 1],
            "feature_x": [1.0, np.nan],
        })
        result = add_missingness_indicators(df)
        assert "mi_feature_x" in result.columns
        assert result["mi_feature_x"].iloc[0] == 0
        assert result["mi_feature_x"].iloc[1] == 1

    def test_skips_participant_and_outcome(self):
        df = pd.DataFrame({
            "participant_id": ["a"],
            "outcome": [0],
            "x": [1.0],
        })
        result = add_missingness_indicators(df)
        assert "mi_participant_id" not in result.columns
        assert "mi_outcome" not in result.columns


# ---------------------------------------------------------------------------
# missing_value_table
# ---------------------------------------------------------------------------

class TestMissingValueTable:
    def test_reports_missing_percentage(self):
        df = pd.DataFrame({
            "participant_id": ["a", "b", "c"],
            "feature_x": [1.0, np.nan, np.nan],
        })
        result = missing_value_table(df)
        assert "feature_x" in result.index
        assert result.loc["feature_x", "missing_%"] == pytest.approx(66.67, abs=0.01)


# ---------------------------------------------------------------------------
# impute_group_median_then_ffill
# ---------------------------------------------------------------------------

class TestImputeGroupMedian:
    def test_fills_nans(self):
        df = pd.DataFrame({
            "participant_id": ["a", "a", "a"],
            "outcome": [0, 1, 0],
            "feature": [np.nan, 2.0, np.nan],
        })
        result, medians = impute_group_median_then_ffill(df)
        assert not result["feature"].isna().any()

    def test_returns_medians_dataframe(self):
        df = pd.DataFrame({
            "participant_id": ["a", "a"],
            "outcome": [0, 1],
            "feature": [1.0, 3.0],
        })
        _, medians = impute_group_median_then_ffill(df)
        assert "feature" in medians.columns


# ---------------------------------------------------------------------------
# impute_within_participant
# ---------------------------------------------------------------------------

class TestImputeWithinParticipant:
    def test_forward_fills_within_group(self):
        df = pd.DataFrame({
            "participant_id": ["a", "a", "a"],
            "outcome": [0, 1, 0],
            "feature": [1.0, np.nan, np.nan],
        })
        global_median = pd.DataFrame({"feature": [5.0]})
        result = impute_within_participant(df, global_median)
        assert result["feature"].iloc[1] == 1.0
        assert result["feature"].iloc[2] == 1.0


# ---------------------------------------------------------------------------
# z_normalize_columns
# ---------------------------------------------------------------------------

class TestZNormalizeColumns:
    def test_normalizes_per_participant(self):
        df = pd.DataFrame({
            "participant_id": ["a", "a", "b", "b"],
            "outcome": [0, 1, 0, 1],
            "val": [10.0, 20.0, 100.0, 200.0],
        })
        result, means = z_normalize_columns(df, cols_to_scale=["val"])
        # Each group should have mean ~0
        group_means = result.groupby("participant_id")["val"].mean()
        np.testing.assert_allclose(group_means.values, [0.0, 0.0], atol=1e-10)

    def test_returns_global_means(self):
        df = pd.DataFrame({
            "participant_id": ["a", "b"],
            "outcome": [0, 1],
            "val": [10.0, 20.0],
        })
        _, means = z_normalize_columns(df, cols_to_scale=["val"])
        assert "val" in means.columns
        assert means["val"].iloc[0] == pytest.approx(15.0)
