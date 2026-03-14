"""
H003: Mixed-Regime Experiment Analysis
TENCON 2026 — Coherence vs Entrainment in Human-AI Agentic Systems

Design: 3 AI proportions (0, 0.2, 0.5) × 5 human-regime-bias levels (0, 0.25, 0.5, 0.75, 1)
        × 2 stress types (single, periodic) × 30 reps = 900 runs

Key question: In mixed-regime populations where agents vary continuously along the
coherence-entrainment spectrum, what human-side regime-bias is sufficient to preserve
adaptive capacity when AI agents are structurally biased toward entrainment (bias=0.8)?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import csv
import os

# ── Config ──────────────────────────────────────────────────────────────────
EXPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports')
DATA_FILE = os.path.join(EXPORT_DIR,
    'coherence_model_tencon H003_batch4_mixed_regime-spreadsheet.csv')
FIG_DIR = EXPORT_DIR

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# ── Parser ──────────────────────────────────────────────────────────────────

def load_h003(filepath):
    """Parse BehaviorSpace spreadsheet v2.0 (wide format) for H003."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    param_rows = {}
    run_number_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        reader = csv.reader([stripped])
        values = next(reader)
        first = values[0].strip('"')

        if first == '[run number]':
            run_number_idx = i
            param_rows['run_number'] = values
        elif first == '[total steps]':
            param_rows['total_steps'] = values
        elif first == '[final value]':
            param_rows['final_header'] = values
            for j in range(i+1, len(lines)):
                if lines[j].strip():
                    data_reader = csv.reader([lines[j].strip()])
                    param_rows['final_data'] = next(data_reader)
                    break
        elif run_number_idx is not None and first not in ('[run number]', '[total steps]', '[final value]'):
            param_rows[first] = values

    # Determine columns per run
    header = param_rows['final_header']
    first_step = None
    cols_per_run = 0
    for idx in range(1, len(header)):
        if header[idx] == '[step]':
            if first_step is None:
                first_step = idx
            else:
                cols_per_run = idx - first_step
                break

    data = param_rows['final_data']
    total_cols = len(data)
    num_runs = (total_cols - 1) // cols_per_run

    # Metric names
    metric_names = [header[1 + i] for i in range(cols_per_run)]

    # Build records
    records = []
    for run in range(num_runs):
        start = 1 + run * cols_per_run
        record = {}

        for param_name, row_vals in param_rows.items():
            if param_name in ('final_header', 'final_data', 'total_steps', 'run_number'):
                continue
            try:
                val = row_vals[start].strip('"')
                try:
                    val = float(val)
                    if val == int(val):
                        val = int(val)
                except (ValueError, TypeError):
                    if val.lower() == 'true':
                        val = True
                    elif val.lower() == 'false':
                        val = False
                record[param_name] = val
            except (IndexError, KeyError):
                pass

        try:
            record['run_number'] = int(param_rows['run_number'][start].strip('"'))
        except (IndexError, KeyError, ValueError):
            record['run_number'] = run + 1

        for m_idx, m_name in enumerate(metric_names):
            col = start + m_idx
            try:
                val = data[col].strip('"')
                record[m_name.replace('-', '_')] = float(val) if val else np.nan
            except (IndexError, ValueError):
                record[m_name.replace('-', '_')] = np.nan

        records.append(record)

    df = pd.DataFrame(records)

    # Normalize column names
    df.columns = [c.replace('-', '_').rstrip('?') for c in df.columns]

    # Ensure numeric types
    for col in ['ai_proportion', 'human_regime_bias_mean', 'ai_regime_bias_mean']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Cascade indicator: recovery_time == -1
    if 'recovery_time' in df.columns:
        df['cascade'] = (df['recovery_time'] == -1).astype(int)
        # For stats on recovering runs, create conditional recovery
        df['recovery_time_cond'] = df['recovery_time'].where(df['recovery_time'] >= 0)

    return df


# ── Analysis ────────────────────────────────────────────────────────────────

def summary_table(df):
    """Summary statistics grouped by stress × ai_proportion × human_regime_bias_mean."""
    print("\n" + "="*80)
    print("SUMMARY: H003 Mixed-Regime Experiment")
    print(f"Total runs: {len(df)}")
    print("="*80)

    key_metrics = ['recovery_time', 'diversity_index', 'mean_cumulative_cost',
                   'cost_gini', 'max_fatigue_level', 'mean_regime_bias',
                   'regime_bias_variance']
    available = [m for m in key_metrics if m in df.columns]

    for stress in ['single', 'periodic']:
        stress_df = df[df['perturbation_regime'] == stress]
        print(f"\n{'─'*40}")
        print(f"STRESS: {stress.upper()} (n={len(stress_df)})")
        print(f"{'─'*40}")

        grouped = stress_df.groupby(['ai_proportion', 'human_regime_bias_mean'])

        # Cascade rates
        print("\n── Cascade Rates ──")
        cascade_rates = grouped['cascade'].mean()
        pivot = cascade_rates.unstack('human_regime_bias_mean')
        print(pivot.to_string(float_format=lambda x: f'{x:.0%}'))

        # Key metric means
        for metric in available:
            print(f"\n── {metric} (mean) ──")
            means = grouped[metric].mean()
            pivot = means.unstack('human_regime_bias_mean')
            print(pivot.to_string(float_format=lambda x: f'{x:.2f}'))


def cascade_analysis(df):
    """Detailed cascade failure analysis — the core H003 question."""
    print("\n" + "="*80)
    print("CASCADE ANALYSIS: Effect of Human Regime-Bias")
    print("="*80)

    for stress in ['single', 'periodic']:
        stress_df = df[df['perturbation_regime'] == stress]
        print(f"\n{'─'*40}")
        print(f"STRESS: {stress.upper()}")
        print(f"{'─'*40}")

        # Cascade rate by condition
        print("\nCascade rate (%) by AI proportion × human-regime-bias:")
        pivot = stress_df.groupby(['ai_proportion', 'human_regime_bias_mean'])['cascade'].mean() * 100
        print(pivot.unstack().to_string(float_format=lambda x: f'{x:.0f}%'))

        # Key comparison: does higher human regime-bias reduce cascades?
        for prop in sorted(stress_df['ai_proportion'].unique()):
            prop_df = stress_df[stress_df['ai_proportion'] == prop]
            biases = sorted(prop_df['human_regime_bias_mean'].unique())
            if len(biases) < 2:
                continue

            low = prop_df[prop_df['human_regime_bias_mean'] == biases[0]]['cascade']
            high = prop_df[prop_df['human_regime_bias_mean'] == biases[-1]]['cascade']
            if low.sum() > 0 or high.sum() > 0:
                # Fisher exact test for cascade proportion
                table = [[int(low.sum()), int(len(low) - low.sum())],
                         [int(high.sum()), int(len(high) - high.sum())]]
                odds, p = stats.fisher_exact(table)
                print(f"\n  AI={prop}: bias={biases[0]} ({low.mean():.0%} cascade) vs "
                      f"bias={biases[-1]} ({high.mean():.0%} cascade)")
                print(f"    Fisher exact: OR={odds:.2f}, p={p:.4f}")


def diversity_analysis(df):
    """Diversity preservation across the regime-bias spectrum."""
    print("\n" + "="*80)
    print("DIVERSITY ANALYSIS: Identity Preservation in Mixed Regimes")
    print("="*80)

    single = df[df['perturbation_regime'] == 'single']

    # Diversity by human-regime-bias (averaging over AI proportions)
    print("\n── Diversity index by human-regime-bias (single stress, non-cascade runs) ──")
    non_cascade = single[single['cascade'] == 0]
    for prop in sorted(non_cascade['ai_proportion'].unique()):
        print(f"\n  AI proportion = {prop}:")
        prop_df = non_cascade[non_cascade['ai_proportion'] == prop]
        for bias in sorted(prop_df['human_regime_bias_mean'].unique()):
            subset = prop_df[prop_df['human_regime_bias_mean'] == bias]
            div = subset['diversity_index']
            print(f"    bias={bias:.2f}: diversity={div.mean():.3f} ± {div.std():.3f} (n={len(subset)})")

    # Kruskal-Wallis for diversity across bias levels within each proportion
    print("\n── Kruskal-Wallis: diversity across bias levels ──")
    for prop in sorted(non_cascade['ai_proportion'].unique()):
        prop_df = non_cascade[non_cascade['ai_proportion'] == prop]
        groups = [prop_df[prop_df['human_regime_bias_mean'] == b]['diversity_index'].dropna()
                  for b in sorted(prop_df['human_regime_bias_mean'].unique())]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            print(f"  AI={prop}: H={h_stat:.2f}, p={p_val:.6f}")


def threshold_analysis(df):
    """Find the human-regime-bias threshold for resilience."""
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS: Where does human identity-preservation matter?")
    print("="*80)

    for stress in ['single', 'periodic']:
        stress_df = df[df['perturbation_regime'] == stress]
        print(f"\n{'─'*40}")
        print(f"STRESS: {stress.upper()}")

        for prop in sorted(stress_df['ai_proportion'].unique()):
            prop_df = stress_df[stress_df['ai_proportion'] == prop]
            biases = sorted(prop_df['human_regime_bias_mean'].unique())

            print(f"\n  AI proportion = {prop}:")
            print(f"    {'Bias':>6} | {'Cascade%':>8} | {'Diversity':>9} | {'Recovery':>10} | {'Cost':>10} | {'Fatigue':>8}")
            print(f"    {'─'*6}-+-{'─'*8}-+-{'─'*9}-+-{'─'*10}-+-{'─'*10}-+-{'─'*8}")

            for bias in biases:
                subset = prop_df[prop_df['human_regime_bias_mean'] == bias]
                cascade_rate = subset['cascade'].mean() * 100
                div = subset[subset['cascade'] == 0]['diversity_index'].mean() if subset['cascade'].sum() < len(subset) else np.nan
                rec = subset['recovery_time_cond'].mean() if subset['recovery_time_cond'].notna().any() else np.nan
                cost = subset['mean_cumulative_cost'].mean()
                fat = subset['max_fatigue_level'].mean()
                print(f"    {bias:6.2f} | {cascade_rate:7.0f}% | {div:9.3f} | {rec:10.1f} | {cost:10.1f} | {fat:8.3f}")


def comparison_with_binary(df):
    """Compare H003 boundary conditions with H001/H002 binary results."""
    print("\n" + "="*80)
    print("BOUNDARY CHECK: H003 at extreme bias vs H001/H002 binary regimes")
    print("="*80)

    # bias=0 should approximate coherence mode; bias=1 should approximate entrainment
    single = df[df['perturbation_regime'] == 'single']

    print("\n  Single stress, AI=0%:")
    for bias in [0, 1]:
        subset = single[(single['ai_proportion'] == 0) & (single['human_regime_bias_mean'] == bias)]
        if len(subset) > 0:
            label = "coherence-like" if bias == 0 else "entrainment-like"
            cascade = subset['cascade'].mean() * 100
            div = subset[subset['cascade'] == 0]['diversity_index'].mean()
            rec = subset['recovery_time_cond'].mean()
            print(f"    bias={bias} ({label}): cascade={cascade:.0f}%, diversity={div:.3f}, recovery={rec:.1f}")

    print("\n  Expected from H001 (binary mode, AI=0%):")
    print("    Coherence: cascade~0%, diversity~0.48, recovery~0-5")
    print("    Entrainment: cascade~10%, diversity~0.29, recovery~50-200")

    print("\n  Single stress, AI=20%:")
    for bias in [0, 1]:
        subset = single[(single['ai_proportion'] == 0.2) & (single['human_regime_bias_mean'] == bias)]
        if len(subset) > 0:
            label = "coherence-like" if bias == 0 else "entrainment-like"
            cascade = subset['cascade'].mean() * 100
            div = subset[subset['cascade'] == 0]['diversity_index'].mean()
            rec = subset['recovery_time_cond'].mean()
            print(f"    bias={bias} ({label}): cascade={cascade:.0f}%, diversity={div:.3f}, recovery={rec:.1f}")

    print("\n  Expected from H001 (binary mode, AI=20%):")
    print("    Coherence: cascade~0%, diversity~0.48")
    print("    Entrainment: cascade~27%, diversity~0.13")


def statistical_tests(df):
    """Core statistical tests for H003."""
    print("\n" + "="*80)
    print("STATISTICAL TESTS")
    print("="*80)

    single = df[df['perturbation_regime'] == 'single']
    key_metrics = ['diversity_index', 'mean_cumulative_cost', 'max_fatigue_level']

    # 1. Effect of human-regime-bias within each AI proportion (Kruskal-Wallis)
    print("\n── Human-regime-bias effect (Kruskal-Wallis) — single stress ──")
    for prop in sorted(single['ai_proportion'].unique()):
        prop_df = single[single['ai_proportion'] == prop]
        print(f"\n  AI proportion = {prop}:")
        for metric in key_metrics:
            groups = [prop_df[prop_df['human_regime_bias_mean'] == b][metric].dropna()
                      for b in sorted(prop_df['human_regime_bias_mean'].unique())]
            groups = [g for g in groups if len(g) >= 3]
            if len(groups) >= 2:
                h_stat, p_val = stats.kruskal(*groups)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"    {metric}: H={h_stat:.2f}, p={p_val:.6f}{sig}")

    # 2. Pairwise: bias=0 vs bias=1 at each proportion (Mann-Whitney)
    print("\n── Extreme bias comparison (Mann-Whitney U) — single stress ──")
    for prop in sorted(single['ai_proportion'].unique()):
        print(f"\n  AI proportion = {prop}:")
        low = single[(single['ai_proportion'] == prop) & (single['human_regime_bias_mean'] == 0)]
        high = single[(single['ai_proportion'] == prop) & (single['human_regime_bias_mean'] == 1)]
        for metric in key_metrics + ['recovery_time']:
            l_vals = low[metric].dropna()
            h_vals = high[metric].dropna()
            if len(l_vals) >= 3 and len(h_vals) >= 3:
                u_stat, p_val = stats.mannwhitneyu(l_vals, h_vals, alternative='two-sided')
                n1, n2 = len(l_vals), len(h_vals)
                r_rb = 1 - (2 * u_stat) / (n1 * n2)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"    {metric}: U={u_stat:.0f}, p={p_val:.6f}{sig}, r_rb={r_rb:.3f}")
                print(f"      bias=0: {l_vals.mean():.3f} ± {l_vals.std():.3f}")
                print(f"      bias=1: {h_vals.mean():.3f} ± {h_vals.std():.3f}")

    # 3. Stress × bias interaction at AI=0.2 (the peak-risk zone from H001)
    print("\n── Stress × Bias interaction at AI=0.2 ──")
    prop_df = df[df['ai_proportion'] == 0.2]
    for bias in sorted(prop_df['human_regime_bias_mean'].unique()):
        single_sub = prop_df[(prop_df['perturbation_regime'] == 'single') &
                             (prop_df['human_regime_bias_mean'] == bias)]
        periodic_sub = prop_df[(prop_df['perturbation_regime'] == 'periodic') &
                               (prop_df['human_regime_bias_mean'] == bias)]
        s_cascade = single_sub['cascade'].mean() * 100
        p_cascade = periodic_sub['cascade'].mean() * 100
        print(f"  bias={bias:.2f}: single={s_cascade:.0f}% cascade, periodic={p_cascade:.0f}% cascade")


# ── Figures ─────────────────────────────────────────────────────────────────

def plot_cascade_heatmap(df):
    """Heatmap: cascade rate by human-regime-bias × ai-proportion, for each stress type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, stress in zip(axes, ['single', 'periodic']):
        stress_df = df[df['perturbation_regime'] == stress]
        pivot = stress_df.groupby(['ai_proportion', 'human_regime_bias_mean'])['cascade'].mean()
        pivot = pivot.unstack('human_regime_bias_mean') * 100

        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{x:.0%}' for x in pivot.index])
        ax.set_xlabel('Human Regime-Bias Mean')
        ax.set_ylabel('AI Proportion')
        ax.set_title(f'{stress.title()} Stress')

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = 'white' if val > 50 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=10)

    fig.suptitle('H003: Cascade Failure Rate by Condition\n'
                 '(AI regime-bias=0.8 fixed; human bias varies)', fontsize=13)
    fig.colorbar(im, ax=axes, label='Cascade Rate (%)', shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'H003_cascade_heatmap.png'), bbox_inches='tight')
    print(f"  Saved: H003_cascade_heatmap.png")
    plt.close()


def plot_diversity_by_bias(df):
    """Diversity index vs human-regime-bias, one curve per AI proportion."""
    single = df[(df['perturbation_regime'] == 'single') & (df['cascade'] == 0)]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0: '#2ecc71', 0.2: '#3498db', 0.5: '#e74c3c'}
    markers = {0: 'o', 0.2: 's', 0.5: '^'}

    for prop in sorted(single['ai_proportion'].unique()):
        prop_df = single[single['ai_proportion'] == prop]
        grouped = prop_df.groupby('human_regime_bias_mean')['diversity_index']
        means = grouped.mean()
        sems = grouped.sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                    fmt=f'-{markers[prop]}', color=colors[prop],
                    label=f'AI={prop:.0%}', capsize=4, linewidth=2, markersize=8)

    ax.set_xlabel('Human Regime-Bias Mean', fontsize=12)
    ax.set_ylabel('Diversity Index', fontsize=12)
    ax.set_title('H003: Diversity vs Human Regime-Bias (Single Stress)\n'
                 'Non-cascade runs only', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0\n(coherence)', '0.25', '0.5', '0.75', '1.0\n(entrainment)'])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'H003_diversity_by_bias.png'))
    print(f"  Saved: H003_diversity_by_bias.png")
    plt.close()


def plot_recovery_surface(df):
    """Recovery time (conditional) vs human-regime-bias × AI proportion."""
    single = df[(df['perturbation_regime'] == 'single') & (df['cascade'] == 0)]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0: '#2ecc71', 0.2: '#3498db', 0.5: '#e74c3c'}
    markers = {0: 'o', 0.2: 's', 0.5: '^'}

    for prop in sorted(single['ai_proportion'].unique()):
        prop_df = single[single['ai_proportion'] == prop]
        grouped = prop_df.groupby('human_regime_bias_mean')['recovery_time']
        means = grouped.mean()
        sems = grouped.sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                    fmt=f'-{markers[prop]}', color=colors[prop],
                    label=f'AI={prop:.0%}', capsize=4, linewidth=2, markersize=8)

    ax.set_xlabel('Human Regime-Bias Mean', fontsize=12)
    ax.set_ylabel('Recovery Time (ticks)', fontsize=12)
    ax.set_title('H003: Recovery Time vs Human Regime-Bias (Single Stress)\n'
                 'Conditional on non-cascade', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'H003_recovery_by_bias.png'))
    print(f"  Saved: H003_recovery_by_bias.png")
    plt.close()


def plot_cost_by_bias(df):
    """Cost and fatigue vs human-regime-bias."""
    single = df[df['perturbation_regime'] == 'single']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {0: '#2ecc71', 0.2: '#3498db', 0.5: '#e74c3c'}
    markers = {0: 'o', 0.2: 's', 0.5: '^'}

    # Panel A: Cumulative cost
    for prop in sorted(single['ai_proportion'].unique()):
        prop_df = single[single['ai_proportion'] == prop]
        grouped = prop_df.groupby('human_regime_bias_mean')['mean_cumulative_cost']
        means = grouped.mean()
        sems = grouped.sem()
        ax1.errorbar(means.index, means.values, yerr=sems.values,
                     fmt=f'-{markers[prop]}', color=colors[prop],
                     label=f'AI={prop:.0%}', capsize=4, linewidth=2, markersize=8)
    ax1.set_xlabel('Human Regime-Bias Mean')
    ax1.set_ylabel('Mean Cumulative Cost')
    ax1.set_title('(a) Regulatory Cost')
    ax1.legend()

    # Panel B: Max fatigue
    for prop in sorted(single['ai_proportion'].unique()):
        prop_df = single[single['ai_proportion'] == prop]
        grouped = prop_df.groupby('human_regime_bias_mean')['max_fatigue_level']
        means = grouped.mean()
        sems = grouped.sem()
        ax2.errorbar(means.index, means.values, yerr=sems.values,
                     fmt=f'-{markers[prop]}', color=colors[prop],
                     label=f'AI={prop:.0%}', capsize=4, linewidth=2, markersize=8)
    ax2.set_xlabel('Human Regime-Bias Mean')
    ax2.set_ylabel('Max Fatigue Level')
    ax2.set_title('(b) Peak Fatigue')
    ax2.legend()

    fig.suptitle('H003: Cost and Fatigue vs Human Regime-Bias (Single Stress)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'H003_cost_fatigue_by_bias.png'))
    print(f"  Saved: H003_cost_fatigue_by_bias.png")
    plt.close()


def plot_periodic_comparison(df):
    """Compare single vs periodic stress across the bias spectrum at AI=0.2."""
    prop_df = df[df['ai_proportion'] == 0.2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'single': '#3498db', 'periodic': '#e74c3c'}

    # Panel A: Cascade rate
    for stress in ['single', 'periodic']:
        stress_sub = prop_df[prop_df['perturbation_regime'] == stress]
        grouped = stress_sub.groupby('human_regime_bias_mean')['cascade']
        rates = grouped.mean() * 100
        ax1.plot(rates.index, rates.values, f'-o', color=colors[stress],
                 label=stress.title(), linewidth=2, markersize=8)
    ax1.set_xlabel('Human Regime-Bias Mean')
    ax1.set_ylabel('Cascade Rate (%)')
    ax1.set_title('(a) Cascade Failure Rate')
    ax1.legend()

    # Panel B: Diversity (non-cascade)
    for stress in ['single', 'periodic']:
        stress_sub = prop_df[(prop_df['perturbation_regime'] == stress) & (prop_df['cascade'] == 0)]
        if len(stress_sub) > 0:
            grouped = stress_sub.groupby('human_regime_bias_mean')['diversity_index']
            means = grouped.mean()
            sems = grouped.sem()
            ax2.errorbar(means.index, means.values, yerr=sems.values,
                         fmt=f'-o', color=colors[stress], label=stress.title(),
                         capsize=4, linewidth=2, markersize=8)
    ax2.set_xlabel('Human Regime-Bias Mean')
    ax2.set_ylabel('Diversity Index')
    ax2.set_title('(b) Diversity (non-cascade runs)')
    ax2.legend()

    fig.suptitle('H003: Single vs Periodic Stress at AI=20%\n'
                 '(The peak-risk zone from H001)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'H003_periodic_comparison.png'))
    print(f"  Saved: H003_periodic_comparison.png")
    plt.close()


def plot_regime_bias_distribution(df):
    """Show actual regime-bias distributions (emergent vs configured)."""
    single = df[(df['perturbation_regime'] == 'single') & (df['cascade'] == 0)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Mean regime-bias vs configured human-regime-bias-mean
    for prop in sorted(single['ai_proportion'].unique()):
        prop_df = single[single['ai_proportion'] == prop]
        grouped = prop_df.groupby('human_regime_bias_mean')['mean_regime_bias']
        means = grouped.mean()
        colors = {0: '#2ecc71', 0.2: '#3498db', 0.5: '#e74c3c'}
        ax1.plot(means.index, means.values, '-o', color=colors[prop],
                 label=f'AI={prop:.0%}', linewidth=2, markersize=8)

    ax1.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='1:1 line')
    ax1.set_xlabel('Configured Human Regime-Bias Mean')
    ax1.set_ylabel('Population Mean Regime-Bias')
    ax1.set_title('(a) Configured vs Emergent Regime-Bias')
    ax1.legend()

    # Panel B: Regime-bias variance
    for prop in sorted(single['ai_proportion'].unique()):
        prop_df = single[single['ai_proportion'] == prop]
        grouped = prop_df.groupby('human_regime_bias_mean')['regime_bias_variance']
        means = grouped.mean()
        ax2.plot(means.index, means.values, '-o', color=colors[prop],
                 label=f'AI={prop:.0%}', linewidth=2, markersize=8)
    ax2.set_xlabel('Configured Human Regime-Bias Mean')
    ax2.set_ylabel('Regime-Bias Variance')
    ax2.set_title('(b) Within-Population Regime Heterogeneity')
    ax2.legend()

    fig.suptitle('H003: Regime-Bias Distribution Verification', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'H003_regime_bias_distribution.png'))
    print(f"  Saved: H003_regime_bias_distribution.png")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading H003 data...")
    df = load_h003(DATA_FILE)

    print(f"\nLoaded {len(df)} runs")
    print(f"Columns: {list(df.columns)}")
    print(f"\nConditions:")
    print(df.groupby(['perturbation_regime', 'ai_proportion', 'human_regime_bias_mean']).size().to_string())

    # Core analyses
    summary_table(df)
    cascade_analysis(df)
    diversity_analysis(df)
    threshold_analysis(df)
    comparison_with_binary(df)
    statistical_tests(df)

    # Figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    plot_cascade_heatmap(df)
    plot_diversity_by_bias(df)
    plot_recovery_surface(df)
    plot_cost_by_bias(df)
    plot_periodic_comparison(df)
    plot_regime_bias_distribution(df)

    print("\nDone.")
