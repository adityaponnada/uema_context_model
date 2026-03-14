"""
Step 10: Survival analysis comparing model setups and random baseline.

Reads simulation results from Setup 1, Setup 2, and Random Baseline,
performs Kaplan-Meier survival analysis, log-rank tests, paired t-tests
and permutation tests on F1 scores, Cox proportional hazard ratio
analysis, and generates publication-quality plots.
"""

import os
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.helpers import set_global_seed, save_figure, save_text_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Survival analysis comparing model setups and random baseline."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing intermediate files and where outputs are saved.")
    parser.add_argument("--random_csv", type=str, required=True,
                        help="Filename of random baseline simulation CSV (in output_dir).")
    parser.add_argument("--s1_extension_csv", type=str, required=True,
                        help="Filename of Setup 1 study extension CSV (in output_dir).")
    parser.add_argument("--s2_extension_csv", type=str, required=True,
                        help="Filename of Setup 2 study extension CSV (in output_dir).")
    parser.add_argument("--general_f1_csv", type=str, default=None,
                        help="Filename of generalized model per-user F1 scores CSV (in output_dir, optional).")
    parser.add_argument("--hybrid_f1_csv", type=str, default=None,
                        help="Filename of hybrid model per-user F1 scores CSV (in output_dir, optional).")
    return parser.parse_args()


def prepare_survival_dataframe(
    df_random: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame
) -> pd.DataFrame:
    """Combine simulation results into long-format DataFrame for survival analysis.

    Creates four groups: Actual (Lazy), Random (20% Block), Setup 1, Setup 2.
    The 'Actual (Lazy)' group uses ground truth actual_days.

    Args:
        df_random: Random baseline simulation results.
        df_s1: Setup 1 study extension results.
        df_s2: Setup 2 study extension results.

    Returns:
        Long-format DataFrame with duration, group, and event columns.
    """
    data_list = []

    # Actual (Lazy) group: ground truth withdrawal times
    actual_df = pd.DataFrame({
        "duration": df_s1["actual_days"],
        "group": "Actual (Lazy)",
        "event": (df_s1["actual_days"] < 365.0).astype(int),
    })
    data_list.append(actual_df)

    # Simulated groups
    mapping = {
        "Random (20% Block)": df_random,
        "Setup 1": df_s1,
        "Setup 2": df_s2,
    }

    for label, df in mapping.items():
        temp_df = pd.DataFrame({
            "duration": df["projected_days"],
            "group": label,
            "event": (df["projected_days"] < 365.0).astype(int),
        })
        data_list.append(temp_df)

    return pd.concat(data_list, ignore_index=True)


def run_statistical_tests(df_long: pd.DataFrame) -> str:
    """Perform omnibus log-rank test and pairwise post-hoc comparisons.

    Args:
        df_long: Long-format survival DataFrame.

    Returns:
        Formatted string with test results.
    """
    from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test

    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append("      SURVIVAL STATISTICS: OMNIBUS & POST-HOC")
    results_lines.append("=" * 60)

    results_omnibus = multivariate_logrank_test(
        df_long["duration"], df_long["group"], df_long["event"]
    )

    results_lines.append("OMNIBUS LOG-RANK TEST")
    results_lines.append(f"p-value: {results_omnibus.p_value:.2e}")
    results_lines.append(f"Test Statistic: {results_omnibus.test_statistic:.4f}")

    if results_omnibus.p_value < 0.05:
        results_lines.append("\nRESULT: Significant difference detected between groups.")
        results_lines.append("Proceeding to Pairwise Post-Hoc comparisons...")

        results_pairwise = pairwise_logrank_test(
            df_long["duration"], df_long["group"], df_long["event"]
        )

        results_lines.append("\nPAIRWISE LOG-RANK SUMMARY:")
        results_lines.append(
            results_pairwise.summary[["test_statistic", "p", "-log2(p)"]].to_string()
        )
    else:
        results_lines.append("\nRESULT: No significant difference detected across groups.")

    results_lines.append("=" * 60)
    text = "\n".join(results_lines)
    print(text)
    return text


def plot_survival_curves(df_long: pd.DataFrame):
    """Generate Kaplan-Meier survival curves for all four conditions.

    Args:
        df_long: Long-format survival DataFrame.

    Returns:
        Matplotlib Figure object.
    """

    from lifelines import KaplanMeierFitter

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    kmf = KaplanMeierFitter()

    styles = {
        "Actual (Lazy)": {"color": "black", "ls": "--", "alpha": 0.9},
        "Random (20% Block)": {"color": "red", "ls": ":", "alpha": 0.8},
        "Setup 1": {"color": "royalblue", "ls": "-", "alpha": 1.0},
        "Setup 2": {"color": "forestgreen", "ls": "-", "alpha": 1.0},
    }

    for group_name in ["Actual (Lazy)", "Random (20% Block)", "Setup 1", "Setup 2"]:
        if group_name not in df_long["group"].unique():
            continue
        mask = df_long["group"] == group_name
        kmf.fit(
            durations=df_long.loc[mask, "duration"],
            event_observed=df_long.loc[mask, "event"],
            label=group_name,
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=styles[group_name]["color"],
            linestyle=styles[group_name]["ls"],
            alpha=styles[group_name]["alpha"],
            lw=3.0 if "Setup" in group_name else 2.0,
        )

    ax.set_title(
        "Counterfactual Survival Analysis: Impact of Intelligent Burden Reduction",
        fontsize=16, fontweight="bold", pad=18,
    )
    ax.set_xlabel("Days in Study (Observation Period)", fontsize=13)
    ax.set_ylabel("Probability of Participant Retention", fontsize=13)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 370)
    ax.axvline(365, color="red", linestyle="--", alpha=0.35, label="Study End (Day 365)")

    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(which="major", linestyle="-", alpha=0.15)

    leg = ax.legend(loc="lower left", fontsize=11, frameon=True)
    if leg is not None:
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_alpha(1.0)
        for txt in leg.get_texts():
            txt.set_color("black")
        try:
            leg.get_title().set_color("black")
        except Exception:
            pass

    sns.despine(ax=ax, offset=8, trim=False)
    plt.tight_layout()
    return fig


def plot_retention_distributions(df_long: pd.DataFrame):
    """Generate 2x2 density plots for each condition's retention distribution.

    Args:
        df_long: Long-format survival DataFrame.

    Returns:
        Matplotlib Figure object.
    """


    order = ["Actual (Lazy)", "Random (20% Block)", "Setup 1", "Setup 2"]
    palette = {
        "Actual (Lazy)": "black",
        "Random (20% Block)": "red",
        "Setup 1": "royalblue",
        "Setup 2": "forestgreen",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, group in enumerate(order):
        ax = axes[i]
        if group not in df_long["group"].unique():
            continue

        data = df_long[df_long["group"] == group]["duration"]
        mean_val = data.mean()

        sns.kdeplot(
            data=data, ax=ax, color=palette[group],
            fill=True, alpha=0.3, linewidth=2, bw_adjust=0.5,
        )

        ax.axvline(x=mean_val, color="black", linestyle="--", linewidth=1.5, alpha=0.8)

        y_limit = ax.get_ylim()[1]
        ax.text(
            mean_val + 2, y_limit * 0.7,
            f"Mean: {mean_val:.1f}d", color="black", fontweight="bold",
            rotation=0,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

        ax.axvline(365, color="red", linestyle="-", alpha=0.2)
        ax.set_title(f"Condition: {group}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Density" if i % 2 == 0 else "")
        ax.set_xlabel("Days in Study" if i >= 2 else "")
        ax.set_xlim(0, 400)
        ax.grid(True, linestyle=":", alpha=0.4)

    plt.suptitle("Retention Density Distribution Analysis", fontsize=17, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def compare_model_performance(df_s1: pd.DataFrame, df_s2: pd.DataFrame) -> str:
    """Perform paired t-test on F1 scores between Setup 1 and Setup 2.

    Args:
        df_s1: Setup 1 extension DataFrame (must contain participant_id, f1).
        df_s2: Setup 2 extension DataFrame (must contain participant_id, f1).

    Returns:
        Formatted string with comparison results.
    """
    from scipy import stats

    df_comp = pd.merge(
        df_s1[["participant_id", "f1"]],
        df_s2[["participant_id", "f1"]],
        on="participant_id",
        suffixes=("_s1", "_s2"),
    )

    if len(df_comp) < 2:
        msg = "Error: Not enough overlapping participants to perform a paired t-test."
        print(msg)
        return msg

    s1_scores = df_comp["f1_s1"]
    s2_scores = df_comp["f1_s2"]

    t_stat, p_val = stats.ttest_rel(s2_scores, s1_scores)

    diffs = s2_scores - s1_scores
    cohen_d = np.mean(diffs) / np.std(diffs, ddof=1)

    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append("      STATISTICAL COMPARISON: SETUP 1 vs SETUP 2 (F1)")
    results_lines.append("=" * 60)
    results_lines.append(f"Number of Participants (N):  {len(df_comp)}")
    results_lines.append(f"Setup 1 Mean F1 (Class 0):   {s1_scores.mean():.4f}")
    results_lines.append(f"Setup 2 Mean F1 (Class 0):   {s2_scores.mean():.4f}")
    results_lines.append(f"Mean Difference:             {np.mean(diffs):+.4f}")
    results_lines.append("-" * 60)
    results_lines.append(f"T-statistic:                 {t_stat:.4f}")
    results_lines.append(f"P-value:                     {p_val:.2e}")
    results_lines.append(f"Cohen's d (Effect Size):      {cohen_d:.4f}")
    results_lines.append("-" * 60)

    if p_val < 0.05:
        verdict = "Statistically Significant"
        direction = "Setup 2 improved performance." if np.mean(diffs) > 0 else "Setup 1 performed better."
    else:
        verdict = "Not Statistically Significant"
        direction = "The performance difference is likely due to chance."

    results_lines.append(f"VERDICT: {verdict}")
    results_lines.append(f"INTERPRETATION: {direction}")

    if abs(cohen_d) < 0.2:
        effect = "Negligible"
    elif abs(cohen_d) < 0.5:
        effect = "Small"
    elif abs(cohen_d) < 0.8:
        effect = "Medium"
    else:
        effect = "Large"

    results_lines.append(f"EFFECT MAGNITUDE: {effect}")
    results_lines.append("=" * 60)

    text = "\n".join(results_lines)
    print(text)
    return text


def plot_f1_boxplot(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    fcol: str = "f1",
    labels: Tuple[str, str] = ("Depth Model", "Breadth Model"),
    baseline: float = 0.20,
):
    """Create box-and-whiskers plot comparing F1 distributions for two model setups.

    Args:
        df_s1: Setup 1 DataFrame containing fcol.
        df_s2: Setup 2 DataFrame containing fcol.
        fcol: Column name holding F1 scores.
        labels: Tuple of x-axis labels for the two models.
        baseline: Horizontal baseline value to draw.

    Returns:
        Tuple of (fig, ax).
    """
    palette = {"Depth Model": "royalblue", "Breadth Model": "forestgreen"}

    s1 = df_s1[fcol].dropna().astype(float)
    s2 = df_s2[fcol].dropna().astype(float)

    data = pd.concat([
        pd.DataFrame({"model": labels[0], "f1": s1.values}),
        pd.DataFrame({"model": labels[1], "f1": s2.values}),
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        x="model", y="f1", data=data, ax=ax,
        palette=palette, width=0.5, fliersize=4,
    )

    # Set fill alpha on box patches
    for patch in ax.patches:
        fc = patch.get_facecolor()
        patch.set_facecolor((*fc[:3], 0.5))

    ax.axhline(baseline, color="red", linestyle="--", linewidth=1.25)
    ax.set_ylim(0, 1)

    y_text = baseline + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.annotate(
        f"Random baseline = {baseline:.2f}",
        xy=(0.5, baseline),
        xytext=(0.90, y_text),
        ha="center", va="bottom", fontsize=10, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1, connectionstyle="arc3,rad=.2"),
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 distribution: Depth Model vs Breadth Model")
    return fig, ax


def plot_f1_boxplot_heldout(
    df_generalized: pd.DataFrame,
    df_hybrid: pd.DataFrame,
    fcol: str = "f1_score",
    labels: Tuple[str, str] = ("Depth Model", "Breadth Model"),
    baseline: float = 0.20,
):
    """Create box-and-whiskers plot comparing held-out F1 distributions.

    Args:
        df_generalized: Generalized model F1 scores.
        df_hybrid: Hybrid model F1 scores.
        fcol: Column name holding F1 scores.
        labels: Tuple of x-axis labels.
        baseline: Horizontal baseline value to draw.

    Returns:
        Tuple of (fig, ax).
    """
    palette = {"Depth Model": "royalblue", "Breadth Model": "forestgreen"}

    s1 = df_generalized[fcol].dropna().astype(float)
    s2 = df_hybrid[fcol].dropna().astype(float)

    data = pd.concat([
        pd.DataFrame({"model": labels[0], "f1": s1.values}),
        pd.DataFrame({"model": labels[1], "f1": s2.values}),
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        x="model", y="f1", data=data, ax=ax,
        palette=palette, width=0.5, fliersize=4,
    )

    # Set fill alpha on box patches
    for patch in ax.patches:
        fc = patch.get_facecolor()
        patch.set_facecolor((*fc[:3], 0.5))

    ax.axhline(baseline, color="red", linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 1)

    y_text = baseline + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.annotate(
        f"Random baseline = {baseline:.2f}",
        xy=(0.5, baseline),
        xytext=(0.90, y_text),
        ha="center", va="bottom", fontsize=10, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1, connectionstyle="arc3,rad=.2"),
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 distribution: Generalized vs Hybrid")
    return fig, ax


def paired_ttest_f1(
    df_generalized: pd.DataFrame,
    df_hybrid: pd.DataFrame,
    fcol: str = "f1_score",
    alpha: float = 0.05,
) -> str:
    """Perform paired-samples t-test on held-out F1 scores.

    Args:
        df_generalized: Generalized model per-user F1 scores.
        df_hybrid: Hybrid model per-user F1 scores.
        fcol: Column name for F1 scores.
        alpha: Significance level.

    Returns:
        Formatted string with t-test results.
    """
    from scipy import stats

    s1 = df_generalized[fcol].astype(float)
    s2 = df_hybrid[fcol].astype(float)

    # Pair by common index
    common_idx = s1.index.intersection(s2.index)
    if len(common_idx) > 0:
        a = s1.loc[common_idx]
        b = s2.loc[common_idx]
        mask = a.notna() & b.notna()
        a = a[mask]
        b = b[mask]
    else:
        a = s1.dropna().reset_index(drop=True)
        b = s2.dropna().reset_index(drop=True)
        nmin = min(len(a), len(b))
        a = a.iloc[:nmin]
        b = b.iloc[:nmin]

    n = len(a)
    if n < 2:
        msg = "Not enough paired observations for t-test (need >=2)"
        print(msg)
        return msg

    diff = (a.values - b.values).astype(float)
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    se = sd_diff / np.sqrt(n)
    dfree = n - 1

    t_stat = mean_diff / se
    p_value = stats.t.sf(abs(t_stat), dfree) * 2

    t_crit = stats.t.ppf(1 - alpha / 2, dfree)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    cohens_d = mean_diff / sd_diff if sd_diff != 0 else np.nan

    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append("   PAIRED T-TEST: GENERALIZED vs HYBRID (Held-out F1)")
    results_lines.append("=" * 60)
    results_lines.append(f"N paired observations:  {n}")
    results_lines.append(f"Mean difference:        {mean_diff:+.4f}")
    results_lines.append(f"95% CI:                 ({ci_low:+.4f}, {ci_high:+.4f})")
    results_lines.append(f"T-statistic:            {t_stat:.4f}")
    results_lines.append(f"P-value:                {p_value:.2e}")
    results_lines.append(f"Cohen's d:              {cohens_d:.4f}")
    results_lines.append("=" * 60)

    text = "\n".join(results_lines)
    print(text)
    return text


def permutation_test_f1(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    fcol_a: str = "f1_score",
    fcol_b: str = "f1_score",
    label_a: str = "Model A",
    label_b: str = "Model B",
    title: str = "PERMUTATION TEST: F1 COMPARISON",
    id_col: Optional[str] = None,
    n_perms: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> str:
    """Perform a paired permutation test on F1 scores between two models.

    Uses sign-flipping permutation under the null hypothesis that there is no
    systematic difference between the two models. Applicable to both:
    - Held-out data: Generalized vs Hybrid (pair by positional index, id_col=None)
    - Withdrew data: Setup 1 vs Setup 2 (pair by participant ID, id_col="participant_id")

    Args:
        df_a: DataFrame containing F1 scores for model A.
        df_b: DataFrame containing F1 scores for model B.
        fcol_a: Column name for F1 scores in df_a.
        fcol_b: Column name for F1 scores in df_b.
        label_a: Display label for model A.
        label_b: Display label for model B.
        title: Section header for the results summary.
        id_col: If provided, merge on this column to pair participants;
                otherwise pair by positional index (common index first,
                then truncate to min length).
        n_perms: Number of permutations.
        alpha: Significance level for null distribution CI and verdict.
        seed: Random seed for reproducibility.

    Returns:
        Formatted string with permutation test summary including observed
        mean difference, Cohen's d effect size with magnitude label,
        directionality, null distribution CI, and two-sided p-value.
    """
    rng = np.random.default_rng(seed)

    # --- Align paired observations ---
    if id_col is not None:
        df_merged = pd.merge(
            df_a[[id_col, fcol_a]],
            df_b[[id_col, fcol_b]],
            on=id_col,
            suffixes=("_a", "_b"),
        )
        # After merge: if fcol_a == fcol_b pandas appends _a/_b; otherwise no suffix
        col_a_merged = f"{fcol_a}_a" if fcol_a == fcol_b else fcol_a
        col_b_merged = f"{fcol_b}_b" if fcol_a == fcol_b else fcol_b
        valid = df_merged[col_a_merged].notna() & df_merged[col_b_merged].notna()
        a = df_merged.loc[valid, col_a_merged].astype(float).values
        b = df_merged.loc[valid, col_b_merged].astype(float).values
    else:
        s_a = df_a[fcol_a].astype(float)
        s_b = df_b[fcol_b].astype(float)
        common_idx = s_a.index.intersection(s_b.index)
        if len(common_idx) > 0:
            mask = s_a.loc[common_idx].notna() & s_b.loc[common_idx].notna()
            a = s_a.loc[common_idx][mask].values
            b = s_b.loc[common_idx][mask].values
        else:
            a = s_a.dropna().reset_index(drop=True)
            b = s_b.dropna().reset_index(drop=True)
            nmin = min(len(a), len(b))
            a = a.iloc[:nmin].values
            b = b.iloc[:nmin].values

    n = len(a)
    if n < 2:
        msg = "Not enough paired observations for permutation test (need >= 2)."
        print(msg)
        return msg

    # --- Observed test statistic: mean paired difference (B - A) ---
    differences = b - a
    t_obs = np.mean(differences)

    # --- Permutation distribution via sign-flipping ---
    t_perms = np.empty(n_perms)
    for i in range(n_perms):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        t_perms[i] = np.mean(differences * signs)

    # --- Two-sided p-value ---
    p_value = float(np.mean(np.abs(t_perms) >= np.abs(t_obs)))

    # --- Null distribution CI ---
    ci_lo = float(np.percentile(t_perms, 100 * alpha / 2))
    ci_hi = float(np.percentile(t_perms, 100 * (1 - alpha / 2)))

    # --- Effect size: Cohen's d on paired differences ---
    sd_diff = float(np.std(differences, ddof=1))
    cohens_d = t_obs / sd_diff if sd_diff > 0 else np.nan

    if np.isnan(cohens_d) or abs(cohens_d) < 0.2:
        effect_magnitude = "Negligible"
    elif abs(cohens_d) < 0.5:
        effect_magnitude = "Small"
    elif abs(cohens_d) < 0.8:
        effect_magnitude = "Medium"
    else:
        effect_magnitude = "Large"

    # --- Verdict and directionality ---
    if p_value < alpha:
        verdict = "Statistically Significant"
        if t_obs > 0:
            direction = f"{label_b} significantly outperforms {label_a}."
        else:
            direction = f"{label_a} significantly outperforms {label_b}."
    else:
        verdict = "Not Statistically Significant"
        direction = "No significant difference detected; result likely due to chance."

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append(f"   {title}")
    results_lines.append("=" * 60)
    results_lines.append(f"N paired observations:      {n}")
    results_lines.append(f"N permutations:             {n_perms:,}")
    results_lines.append(f"Mean F1 {label_a}:           {mean_a:.4f}")
    results_lines.append(f"Mean F1 {label_b}:           {mean_b:.4f}")
    results_lines.append(f"Observed mean diff")
    results_lines.append(f"  ({label_b} - {label_a}):  {t_obs:+.4f}")
    results_lines.append("-" * 60)
    results_lines.append(f"P-value (two-sided):        {p_value:.4f}")
    results_lines.append(
        f"Null dist. {int((1 - alpha) * 100)}% CI:     [{ci_lo:+.4f}, {ci_hi:+.4f}]"
    )
    results_lines.append(f"Cohen's d (effect size):    {cohens_d:.4f}")
    results_lines.append(f"Effect magnitude:           {effect_magnitude}")
    results_lines.append("-" * 60)
    results_lines.append(f"VERDICT:        {verdict}")
    results_lines.append(f"DIRECTIONALITY: {direction}")
    results_lines.append("=" * 60)

    text = "\n".join(results_lines)
    print(text)
    return text


def compute_hazard_ratios(df_long: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Compute hazard ratios using Cox Proportional Hazards regression.

    Args:
        df_long: Long-format survival DataFrame.

    Returns:
        Tuple of (hazard ratio DataFrame, formatted results string).
    """
    from lifelines import CoxPHFitter

    results_lines = []
    results_lines.append("=" * 60)
    results_lines.append("      HAZARD RATIO ANALYSIS (Effect Size)")
    results_lines.append("=" * 60)

    comparisons = [
        ("Random (20% Block)", "Actual (Lazy)"),
        ("Setup 1", "Actual (Lazy)"),
        ("Setup 2", "Actual (Lazy)"),
        ("Setup 2", "Setup 1"),
        ("Setup 1", "Random (20% Block)"),
        ("Setup 2", "Random (20% Block)"),
    ]

    hr_results = []

    for target, reference in comparisons:
        pair_df = df_long[df_long["group"].isin([reference, target])].copy()
        pair_df["is_target"] = (pair_df["group"] == target).astype(int)

        cph = CoxPHFitter()
        cph.fit(
            pair_df[["duration", "event", "is_target"]],
            duration_col="duration",
            event_col="event",
        )

        summary = cph.summary
        hr = summary.loc["is_target", "exp(coef)"]
        lower_ci = summary.loc["is_target", "exp(coef) lower 95%"]
        upper_ci = summary.loc["is_target", "exp(coef) upper 95%"]
        p_val = summary.loc["is_target", "p"]

        hr_results.append({
            "Comparison": f"{target} vs {reference}",
            "Hazard Ratio": hr,
            "95% CI Lower": lower_ci,
            "95% CI Upper": upper_ci,
            "p-value": p_val,
        })

    df_hr = pd.DataFrame(hr_results)

    results_lines.append(df_hr.to_string(index=False, formatters={
        "Hazard Ratio": "{:,.3f}".format,
        "95% CI Lower": "{:,.3f}".format,
        "95% CI Upper": "{:,.3f}".format,
        "p-value": "{:,.2e}".format,
    }))
    results_lines.append("-" * 60)
    results_lines.append("INTERPRETATION GUIDE:")
    results_lines.append("HR < 1.0: Target group has a lower risk (hazard) of withdrawal than Reference.")
    results_lines.append("HR = 0.30: 70% reduction in the hazard of quitting compared to reference.")
    results_lines.append("Example: Setup 2 vs Setup 1 HR = 0.75 means 25% lower risk for Setup 2.")
    results_lines.append("=" * 60)

    text = "\n".join(results_lines)
    print(text)
    return df_hr, text


def plot_effective_days_density(df_s1: pd.DataFrame, df_s2: pd.DataFrame):
    """Create stacked density plots of effective_days for Setup 1 and Setup 2.

    Both subplots share the same x-axis for direct comparison.
    Mean values are highlighted with vertical dotted lines.

    Args:
        df_s1: Setup 1 extension DataFrame with effective_days column.
        df_s2: Setup 2 extension DataFrame with effective_days column.

    Returns:
        Matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Setup 1
    s1_data = df_s1["effective_days"].dropna()
    s1_mean = s1_data.mean()
    sns.kdeplot(data=s1_data, ax=ax1, color="royalblue", fill=True, alpha=0.3, linewidth=2)
    ax1.axvline(s1_mean, color="royalblue", linestyle=":", linewidth=2, label=f"Mean: {s1_mean:.1f} days")
    ax1.set_title("Setup 1 (Depth Model)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Setup 2
    s2_data = df_s2["effective_days"].dropna()
    s2_mean = s2_data.mean()
    sns.kdeplot(data=s2_data, ax=ax2, color="forestgreen", fill=True, alpha=0.3, linewidth=2)
    ax2.axvline(s2_mean, color="forestgreen", linestyle=":", linewidth=2, label=f"Mean: {s2_mean:.1f} days")
    ax2.set_title("Setup 2 (Breadth Model)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Effective Days (Recall Class 1 × Projected Days)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.4)

    plt.suptitle("Effective Days Distribution Comparison", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> None:
    """Main survival analysis pipeline."""
    set_global_seed()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve filenames against output_dir
    random_csv_path = os.path.join(args.output_dir, args.random_csv)
    s1_extension_path = os.path.join(args.output_dir, args.s1_extension_csv)
    s2_extension_path = os.path.join(args.output_dir, args.s2_extension_csv)

    # Load simulation results
    df_sim_random = pd.read_csv(random_csv_path)
    print(f"Random baseline: {random_csv_path} {df_sim_random.shape}")

    df_s1 = pd.read_csv(s1_extension_path)
    print(f"Setup 1 extension: {s1_extension_path} {df_s1.shape}")

    df_s2 = pd.read_csv(s2_extension_path)
    print(f"Setup 2 extension: {s2_extension_path} {df_s2.shape}")

    # Compute effective_days = recall_class_1 * projected_days
    df_s1["effective_days"] = df_s1["recall_class_1"] * df_s1["projected_days"]
    df_s2["effective_days"] = df_s2["recall_class_1"] * df_s2["projected_days"]
    print(f"Setup 1 mean effective_days: {df_s1['effective_days'].mean():.2f}")
    print(f"Setup 2 mean effective_days: {df_s2['effective_days'].mean():.2f}")

    # Effective days density plot
    fig_eff = plot_effective_days_density(df_s1, df_s2)
    save_figure(fig_eff, args.output_dir, "effective_days_density_comparison.png")

    # Prepare survival DataFrame
    df_survival = prepare_survival_dataframe(df_sim_random, df_s1, df_s2)

    # Omnibus and pairwise log-rank tests
    logrank_text = run_statistical_tests(df_survival)
    save_text_results(logrank_text, args.output_dir, "survival_logrank_results.txt")

    # Kaplan-Meier survival curves
    fig_km = plot_survival_curves(df_survival)
    save_figure(fig_km, args.output_dir, "kaplan_meier_survival_curves.png")

    # Retention density distributions
    fig_density = plot_retention_distributions(df_survival)
    save_figure(fig_density, args.output_dir, "retention_density_distributions.png")

    # Paired t-test: Setup 1 vs Setup 2 F1 scores
    ttest_text = compare_model_performance(df_s1, df_s2)
    save_text_results(ttest_text, args.output_dir, "setup1_vs_setup2_ttest_results.txt")

    # Permutation test: Setup 1 vs Setup 2 F1 scores (withdrew data)
    perm_withdrew_text = permutation_test_f1(
        df_s1, df_s2,
        fcol_a="f1", fcol_b="f1",
        label_a="Setup 1", label_b="Setup 2",
        title="PERMUTATION TEST: SETUP 1 vs SETUP 2 (Withdrew F1)",
        id_col="participant_id",
    )
    save_text_results(perm_withdrew_text, args.output_dir, "setup1_vs_setup2_permutation_results.txt")

    # Violin plot: Setup 1 vs Setup 2 F1
    fig_box, _ = plot_f1_boxplot(df_s1, df_s2, fcol="f1")
    save_figure(fig_box, args.output_dir, "f1_boxplot_setup1_vs_setup2.png")

    # Hazard ratio analysis
    df_hr, hr_text = compute_hazard_ratios(df_survival)
    df_hr.to_csv(os.path.join(args.output_dir, "hazard_ratio_summary.csv"), index=False)
    save_text_results(hr_text, args.output_dir, "hazard_ratio_results.txt")

    # Held-out F1 comparison (if CSVs provided)
    general_f1_path = os.path.join(args.output_dir, args.general_f1_csv) if args.general_f1_csv else None
    hybrid_f1_path = os.path.join(args.output_dir, args.hybrid_f1_csv) if args.hybrid_f1_csv else None

    if general_f1_path and hybrid_f1_path and os.path.exists(general_f1_path) and os.path.exists(hybrid_f1_path):
        df_general_f1 = pd.read_csv(general_f1_path)
        df_hybrid_f1 = pd.read_csv(hybrid_f1_path)

        # Rename columns for consistency if needed
        if "f1_score_c0" in df_general_f1.columns:
            df_general_f1.rename(columns={"f1_score_c0": "f1_score"}, inplace=True)
        
        if "f1_score_c0" in df_hybrid_f1.columns:
            df_hybrid_f1.rename(columns={"f1_score_c0": "f1_score"}, inplace=True)

        print(f"\nGeneralized model F1 min: {df_general_f1['f1_score'].min():.4f}")
        print(f"Hybrid model F1 min: {df_hybrid_f1['f1_score'].min():.4f}")

        # Violin plot: Generalized vs Hybrid held-out F1
        fig_box_ho, _ = plot_f1_boxplot_heldout(df_general_f1, df_hybrid_f1, fcol="f1_score")
        save_figure(fig_box_ho, args.output_dir, "f1_boxplot_generalized_vs_hybrid.png")

        # Paired t-test on held-out F1
        heldout_ttest_text = paired_ttest_f1(df_general_f1, df_hybrid_f1, fcol="f1_score")
        save_text_results(heldout_ttest_text, args.output_dir, "heldout_paired_ttest_results.txt")

        # Permutation test on held-out F1
        perm_heldout_text = permutation_test_f1(
            df_general_f1, df_hybrid_f1,
            fcol_a="f1_score", fcol_b="f1_score",
            label_a="Generalized", label_b="Hybrid",
            title="PERMUTATION TEST: GENERALIZED vs HYBRID (Held-out F1)",
            id_col=None,
        )
        save_text_results(perm_heldout_text, args.output_dir, "heldout_permutation_results.txt")
    else:
        print("\nSkipping held-out F1 comparison (CSVs not found).")

    print("\nSurvival analysis complete.")


if __name__ == "__main__":
    main()
