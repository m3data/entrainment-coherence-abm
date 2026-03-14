"""
H001 Batch 1: AI Proportion Sweep (Exploratory)
TENCON 2026 — Coherence vs Entrainment in Human-AI Agentic Systems

Design: 4 AI proportions (0, 0.2, 0.5, 0.9) × 2 modes × 10 reps = 80 runs
Questions:
  1. Does AI proportion affect recovery time / heading variance?
  2. Is there a tipping point?
  3. Does coherence mode show more resilience than entrainment across proportions?
  4. Backward compatibility: does ai-proportion=0 match base model?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import csv
import sys
import os

# ── Config ──────────────────────────────────────────────────────────────────
EXPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports')
DATA_FILE = os.path.join(EXPORT_DIR,
    'coherence_model_tencon H001_batch1_proportion_sweep_exploratory-spreadsheet.csv')
FIG_DIR = EXPORT_DIR  # save figures alongside data

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# ── Parser ──────────────────────────────────────────────────────────────────

def load_h001_batch1(filepath):
    """
    Parse BehaviorSpace spreadsheet v2.0 (wide format).

    Structure:
      Rows 0-5: header metadata
      Row 6: [run number] repeated across columns
      Rows 7-31: parameter values (one row per parameter, repeated per run)
      Row 32: [total steps]
      Row 33: blank
      Row 34: [final value] header — metric names per run
      Row 35: final values — metric data per run
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Build a dict of parameter_name -> raw row values
    param_rows = {}
    run_number_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Parse as CSV
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
            # Next non-empty line is data
            for j in range(i+1, len(lines)):
                if lines[j].strip():
                    data_reader = csv.reader([lines[j].strip()])
                    param_rows['final_data'] = next(data_reader)
                    break
        elif run_number_idx is not None and first not in ('[run number]', '[total steps]', '[final value]'):
            # This is a parameter row
            param_rows[first] = values

    # Determine columns per run from final_header
    header = param_rows['final_header']
    # First col is label "[final value]", then repeating groups of metrics per run
    # Find the pattern: [step], metric1, metric2, ..., [step], metric1, ...
    # Count metrics in first run
    metrics_start = 1  # skip first column
    cols_per_run = 0
    for idx in range(metrics_start, len(header)):
        if header[idx] == '[step]':
            if cols_per_run == 0:
                # Start counting
                cols_per_run = 1
            else:
                break
        elif cols_per_run > 0:
            cols_per_run += 1

    if cols_per_run == 0:
        # Count from first [step] to second [step]
        first_step = None
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

    print(f"Columns per run: {cols_per_run}")
    print(f"Total runs: {num_runs}")

    # Extract metric names from header (first run's block)
    metric_names = []
    for idx in range(1, 1 + cols_per_run):
        metric_names.append(header[idx])
    print(f"Metrics: {metric_names}")

    # Build records
    records = []
    for run in range(num_runs):
        start = 1 + run * cols_per_run
        record = {}

        # Get parameters for this run
        for param_name, row_vals in param_rows.items():
            if param_name in ('final_header', 'final_data', 'total_steps', 'run_number'):
                continue
            try:
                val = row_vals[start].strip('"')
                # Try numeric
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

        # Get run number
        try:
            record['run_number'] = int(param_rows['run_number'][start].strip('"'))
        except (IndexError, KeyError, ValueError):
            record['run_number'] = run + 1

        # Get metric values
        for m_idx, m_name in enumerate(metric_names):
            col = start + m_idx
            try:
                val = data[col].strip('"')
                record[m_name.replace('-', '_')] = float(val) if val else np.nan
            except (IndexError, ValueError):
                record[m_name.replace('-', '_')] = np.nan

        records.append(record)

    df = pd.DataFrame(records)

    # Rename for convenience
    rename = {
        'entrainment_mode?': 'entrainment_mode',
        'ai_proportion': 'ai_proportion',
        'recovery_time': 'recovery_time',
        'max_deviation': 'max_deviation',
        'heading_variance': 'heading_variance',
        'diversity_index': 'diversity_index',
        'mean_cumulative_cost': 'mean_cumulative_cost',
        'human_ai_cost_ratio': 'human_ai_cost_ratio',
        'cost_gini': 'cost_gini',
        'human_heading_variance': 'human_heading_variance',
        'ai_heading_variance': 'ai_heading_variance',
        'human_diversity_index': 'human_diversity_index',
        'ai_diversity_index': 'ai_diversity_index',
        'human_mean_cost': 'human_mean_cost',
        'ai_mean_cost': 'ai_mean_cost',
        'mean_alignment_work': 'mean_alignment_work',
        'human_mean_alignment_work': 'human_mean_alignment_work',
        'ai_mean_alignment_work': 'ai_mean_alignment_work',
        'human_ai_work_ratio': 'human_ai_work_ratio',
        'max_fatigue_level': 'max_fatigue_level',
        'human_max_fatigue': 'human_max_fatigue',
        'ai_max_fatigue': 'ai_max_fatigue',
        'mean_fatigue_level': 'mean_fatigue_level',
        'human_mean_fatigue': 'human_mean_fatigue',
        'ai_mean_fatigue': 'ai_mean_fatigue',
        'agents_fatigued': 'agents_fatigued',
        'initial_coupling_bias_variance': 'initial_coupling_bias_variance',
        'num_humans': 'num_humans',
        'num_ais': 'num_ais',
    }
    existing_renames = {k: v for k, v in rename.items() if k in df.columns and k != v}
    df = df.rename(columns=existing_renames)

    # Normalize column names: hyphens to underscores, strip ?
    df.columns = [c.replace('-', '_').rstrip('?') for c in df.columns]

    # Fix entrainment_mode column
    if 'entrainment_mode' in df.columns:
        df['entrainment_mode'] = df['entrainment_mode'].map(
            {True: True, False: False, 'true': True, 'false': False, 1: True, 0: False})

    # Add mode label
    df['mode'] = df['entrainment_mode'].map({True: 'Entrainment', False: 'Coherence'})

    # Ensure ai_proportion is float
    if 'ai_proportion' in df.columns:
        df['ai_proportion'] = df['ai_proportion'].astype(float)

    return df


# ── Analysis ────────────────────────────────────────────────────────────────

def summary_table(df):
    """Print summary statistics grouped by mode × ai_proportion."""
    key_metrics = ['recovery_time', 'max_deviation', 'heading_variance',
                   'diversity_index', 'mean_cumulative_cost', 'human_ai_cost_ratio',
                   'cost_gini', 'max_fatigue_level', 'agents_fatigued']

    available = [m for m in key_metrics if m in df.columns]

    print("\n" + "="*80)
    print("SUMMARY: H001 Batch 1 — AI Proportion × Mode")
    print("="*80)

    grouped = df.groupby(['mode', 'ai_proportion'])

    for metric in available:
        print(f"\n── {metric} ──")
        table = grouped[metric].agg(['mean', 'std', 'median', 'min', 'max'])
        print(table.to_string())

    # Compact comparison table
    print("\n" + "="*80)
    print("COMPACT: Mean ± SD by condition")
    print("="*80)
    for metric in available:
        print(f"\n{metric}:")
        for mode in ['Coherence', 'Entrainment']:
            vals = []
            for prop in sorted(df['ai_proportion'].unique()):
                subset = df[(df['mode'] == mode) & (df['ai_proportion'] == prop)]
                if len(subset) > 0:
                    m = subset[metric].mean()
                    s = subset[metric].std()
                    vals.append(f"  {prop:.0%}: {m:.2f} ± {s:.2f}")
            print(f"  {mode}: " + " | ".join(vals))


def backward_compat_check(df):
    """Check ai-proportion=0 matches base model expectations."""
    print("\n" + "="*80)
    print("BACKWARD COMPATIBILITY: ai-proportion = 0")
    print("="*80)

    baseline = df[df['ai_proportion'] == 0]
    if len(baseline) == 0:
        print("  No ai-proportion=0 runs found!")
        return

    for mode in ['Coherence', 'Entrainment']:
        subset = baseline[baseline['mode'] == mode]
        print(f"\n  {mode} (n={len(subset)}):")
        for metric in ['recovery_time', 'max_deviation', 'heading_variance',
                        'mean_cumulative_cost', 'max_fatigue_level']:
            if metric in subset.columns:
                m = subset[metric].mean()
                s = subset[metric].std()
                print(f"    {metric}: {m:.2f} ± {s:.2f}")

    # Compare with known E003 results at strength=60
    print("\n  Expected from prior experiments (E003, strength=60):")
    print("    Coherence recovery ~0-5 ticks, Entrainment ~80-200 ticks")
    print("    Coherence max_deviation ~5-10, Entrainment ~20-30")

    coh = baseline[baseline['mode'] == 'Coherence']['recovery_time']
    ent = baseline[baseline['mode'] == 'Entrainment']['recovery_time']
    if len(coh) > 0 and len(ent) > 0:
        ratio = ent.mean() / max(coh.mean(), 1)
        print(f"\n  Recovery ratio (Ent/Coh): {ratio:.1f}×")
        if ratio > 3:
            print("  ✓ Mode distinction preserved at ai-proportion=0")
        else:
            print("  ⚠ Mode distinction may be weak — investigate")


def statistical_tests(df):
    """Between-condition statistical comparisons."""
    print("\n" + "="*80)
    print("STATISTICAL TESTS")
    print("="*80)

    proportions = sorted(df['ai_proportion'].unique())
    key_metrics = ['recovery_time', 'max_deviation', 'heading_variance',
                   'mean_cumulative_cost', 'cost_gini']
    available = [m for m in key_metrics if m in df.columns]

    # 1. Mode effect at each proportion (Mann-Whitney U)
    print("\n── Mode Effect (Coherence vs Entrainment) at each AI proportion ──")
    print("  Mann-Whitney U tests:")
    for prop in proportions:
        print(f"\n  ai-proportion = {prop}:")
        coh = df[(df['mode'] == 'Coherence') & (df['ai_proportion'] == prop)]
        ent = df[(df['mode'] == 'Entrainment') & (df['ai_proportion'] == prop)]
        for metric in available:
            if metric in coh.columns and metric in ent.columns:
                c_vals = coh[metric].dropna()
                e_vals = ent[metric].dropna()
                if len(c_vals) >= 3 and len(e_vals) >= 3:
                    u_stat, p_val = stats.mannwhitneyu(c_vals, e_vals, alternative='two-sided')
                    # Rank-biserial effect size
                    n1, n2 = len(c_vals), len(e_vals)
                    r_rb = 1 - (2 * u_stat) / (n1 * n2)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"    {metric}: U={u_stat:.0f}, p={p_val:.4f}{sig}, r_rb={r_rb:.3f}")

    # 2. Proportion effect within each mode (Kruskal-Wallis)
    print("\n── Proportion Effect within each mode (Kruskal-Wallis) ──")
    for mode in ['Coherence', 'Entrainment']:
        print(f"\n  {mode}:")
        mode_df = df[df['mode'] == mode]
        for metric in available:
            groups = [mode_df[mode_df['ai_proportion'] == p][metric].dropna()
                      for p in proportions]
            groups = [g for g in groups if len(g) >= 3]
            if len(groups) >= 2:
                h_stat, p_val = stats.kruskal(*groups)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"    {metric}: H={h_stat:.2f}, p={p_val:.4f}{sig}")

    # 3. Pairwise: 0% vs each proportion within entrainment
    print("\n── Pairwise: ai-proportion=0 vs others (Entrainment mode) ──")
    ent_baseline = df[(df['mode'] == 'Entrainment') & (df['ai_proportion'] == 0)]
    for prop in proportions:
        if prop == 0:
            continue
        ent_test = df[(df['mode'] == 'Entrainment') & (df['ai_proportion'] == prop)]
        print(f"\n  0% vs {prop:.0%}:")
        for metric in available:
            b_vals = ent_baseline[metric].dropna()
            t_vals = ent_test[metric].dropna()
            if len(b_vals) >= 3 and len(t_vals) >= 3:
                u_stat, p_val = stats.mannwhitneyu(b_vals, t_vals, alternative='two-sided')
                n1, n2 = len(b_vals), len(t_vals)
                r_rb = 1 - (2 * u_stat) / (n1 * n2)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"    {metric}: p={p_val:.4f}{sig}, r_rb={r_rb:.3f}, "
                      f"means: {b_vals.mean():.2f} vs {t_vals.mean():.2f}")


def plot_recovery_by_proportion(df):
    """Fig 2 pilot: Recovery time vs AI proportion, two curves."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, color, marker in [('Coherence', '#2ecc71', 'o'), ('Entrainment', '#e74c3c', 's')]:
        mode_df = df[df['mode'] == mode]
        grouped = mode_df.groupby('ai_proportion')['recovery_time']
        means = grouped.mean()
        sems = grouped.sem()

        ax.errorbar(means.index, means.values, yerr=sems.values,
                    fmt=f'-{marker}', color=color, label=mode,
                    capsize=4, linewidth=2, markersize=8)

        # Individual points (jittered)
        for prop in means.index:
            subset = mode_df[mode_df['ai_proportion'] == prop]['recovery_time']
            jitter = np.random.normal(0, 0.008, len(subset))
            ax.scatter(prop + jitter, subset, color=color, alpha=0.3, s=20, zorder=1)

    ax.set_xlabel('AI Proportion', fontsize=12)
    ax.set_ylabel('Recovery Time (ticks)', fontsize=12)
    ax.set_title('H001 Batch 1: Recovery Time vs AI Proportion\n'
                 '(perturbation-strength=60, n=10 per condition)', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(sorted(df['ai_proportion'].unique()))
    ax.set_xticklabels([f'{x:.0%}' for x in sorted(df['ai_proportion'].unique())])

    fig.savefig(os.path.join(FIG_DIR, 'H001_batch1_recovery_vs_proportion.png'))
    print(f"\n  Saved: H001_batch1_recovery_vs_proportion.png")
    plt.close()


def plot_dual_panel(df):
    """Fig 3 pilot: (a) diversity collapse, (b) cost asymmetry."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Heading variance (or diversity_index) vs proportion
    metric_a = 'diversity_index' if 'diversity_index' in df.columns else 'heading_variance'
    for mode, color, marker in [('Coherence', '#2ecc71', 'o'), ('Entrainment', '#e74c3c', 's')]:
        mode_df = df[df['mode'] == mode]
        grouped = mode_df.groupby('ai_proportion')[metric_a]
        means = grouped.mean()
        sems = grouped.sem()
        ax1.errorbar(means.index, means.values, yerr=sems.values,
                    fmt=f'-{marker}', color=color, label=mode,
                    capsize=4, linewidth=2, markersize=8)

    ax1.set_xlabel('AI Proportion')
    ax1.set_ylabel(metric_a.replace('_', ' ').title())
    ax1.set_title('(a) Diversity vs AI Proportion')
    ax1.legend()
    ax1.set_xticks(sorted(df['ai_proportion'].unique()))
    ax1.set_xticklabels([f'{x:.0%}' for x in sorted(df['ai_proportion'].unique())])

    # Panel B: Cost asymmetry
    if 'human_ai_cost_ratio' in df.columns:
        for mode, color, marker in [('Coherence', '#2ecc71', 'o'), ('Entrainment', '#e74c3c', 's')]:
            mode_df = df[df['mode'] == mode]
            # Only where AI exists
            mode_df_ai = mode_df[mode_df['ai_proportion'] > 0]
            if len(mode_df_ai) > 0:
                grouped = mode_df_ai.groupby('ai_proportion')['human_ai_cost_ratio']
                means = grouped.mean()
                sems = grouped.sem()
                ax2.errorbar(means.index, means.values, yerr=sems.values,
                            fmt=f'-{marker}', color=color, label=mode,
                            capsize=4, linewidth=2, markersize=8)

        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Parity')
        ax2.set_xlabel('AI Proportion')
        ax2.set_ylabel('Human / AI Cost Ratio')
        ax2.set_title('(b) Cost Asymmetry vs AI Proportion')
        ax2.legend()
        props = sorted(df[df['ai_proportion'] > 0]['ai_proportion'].unique())
        ax2.set_xticks(props)
        ax2.set_xticklabels([f'{x:.0%}' for x in props])

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'H001_batch1_dual_panel.png'))
    print(f"  Saved: H001_batch1_dual_panel.png")
    plt.close()


def plot_fatigue_panel(df):
    """Fatigue and cost burden across conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A: Max fatigue
    ax = axes[0]
    for mode, color, marker in [('Coherence', '#2ecc71', 'o'), ('Entrainment', '#e74c3c', 's')]:
        mode_df = df[df['mode'] == mode]
        grouped = mode_df.groupby('ai_proportion')['max_fatigue_level']
        means = grouped.mean()
        sems = grouped.sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                    fmt=f'-{marker}', color=color, label=mode,
                    capsize=4, linewidth=2, markersize=8)
    ax.set_xlabel('AI Proportion')
    ax.set_ylabel('Max Fatigue Level')
    ax.set_title('(a) Peak Fatigue')
    ax.legend()

    # B: Agents fatigued
    if 'agents_fatigued' in df.columns:
        ax = axes[1]
        for mode, color, marker in [('Coherence', '#2ecc71', 'o'), ('Entrainment', '#e74c3c', 's')]:
            mode_df = df[df['mode'] == mode]
            grouped = mode_df.groupby('ai_proportion')['agents_fatigued']
            means = grouped.mean()
            sems = grouped.sem()
            ax.errorbar(means.index, means.values, yerr=sems.values,
                        fmt=f'-{marker}', color=color, label=mode,
                        capsize=4, linewidth=2, markersize=8)
        ax.set_xlabel('AI Proportion')
        ax.set_ylabel('Agents Fatigued (count)')
        ax.set_title('(b) Fatigued Agent Count')
        ax.legend()

    # C: Cost Gini
    if 'cost_gini' in df.columns:
        ax = axes[2]
        for mode, color, marker in [('Coherence', '#2ecc71', 'o'), ('Entrainment', '#e74c3c', 's')]:
            mode_df = df[df['mode'] == mode]
            grouped = mode_df.groupby('ai_proportion')['cost_gini']
            means = grouped.mean()
            sems = grouped.sem()
            ax.errorbar(means.index, means.values, yerr=sems.values,
                        fmt=f'-{marker}', color=color, label=mode,
                        capsize=4, linewidth=2, markersize=8)
        ax.set_xlabel('AI Proportion')
        ax.set_ylabel('Cost Gini Coefficient')
        ax.set_title('(c) Cost Inequality')
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'H001_batch1_fatigue_panel.png'))
    print(f"  Saved: H001_batch1_fatigue_panel.png")
    plt.close()


def plot_human_vs_ai_variance(df):
    """Compare human and AI heading variance across conditions."""
    if 'human_heading_variance' not in df.columns or 'ai_heading_variance' not in df.columns:
        print("  Skipping human/AI variance plot — columns missing")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mode in [(ax1, 'Coherence'), (ax2, 'Entrainment')]:
        mode_df = df[(df['mode'] == mode) & (df['ai_proportion'] > 0)]
        props = sorted(mode_df['ai_proportion'].unique())

        human_means = [mode_df[mode_df['ai_proportion'] == p]['human_heading_variance'].mean() for p in props]
        ai_means = [mode_df[mode_df['ai_proportion'] == p]['ai_heading_variance'].mean() for p in props]

        x = np.arange(len(props))
        width = 0.35
        ax.bar(x - width/2, human_means, width, label='Human', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, ai_means, width, label='AI', color='#e67e22', alpha=0.8)
        ax.set_xlabel('AI Proportion')
        ax.set_ylabel('Heading Variance')
        ax.set_title(f'{mode} Mode')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p:.0%}' for p in props])
        ax.legend()

    fig.suptitle('Human vs AI Heading Variance by Condition', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'H001_batch1_human_ai_variance.png'))
    print(f"  Saved: H001_batch1_human_ai_variance.png")
    plt.close()


def tipping_point_analysis(df):
    """Look for nonlinear regime shifts in entrainment mode."""
    print("\n" + "="*80)
    print("TIPPING POINT ANALYSIS")
    print("="*80)

    ent = df[df['mode'] == 'Entrainment']
    coh = df[df['mode'] == 'Coherence']
    proportions = sorted(df['ai_proportion'].unique())

    print("\n── Recovery time ratios (Entrainment / Coherence) ──")
    for prop in proportions:
        e_rt = ent[ent['ai_proportion'] == prop]['recovery_time'].mean()
        c_rt = coh[coh['ai_proportion'] == prop]['recovery_time'].mean()
        ratio = e_rt / max(c_rt, 1)
        print(f"  {prop:.0%}: Ent={e_rt:.1f}, Coh={c_rt:.1f}, ratio={ratio:.1f}×")

    print("\n── Entrainment mode: step changes between proportions ──")
    metrics = ['recovery_time', 'max_deviation', 'heading_variance']
    available = [m for m in metrics if m in ent.columns]

    for metric in available:
        print(f"\n  {metric}:")
        prev_mean = None
        for prop in proportions:
            curr = ent[ent['ai_proportion'] == prop][metric]
            curr_mean = curr.mean()
            if prev_mean is not None:
                change = curr_mean - prev_mean
                pct = (change / max(abs(prev_mean), 0.01)) * 100
                marker = " ← JUMP" if abs(pct) > 100 else ""
                print(f"    {prop:.0%}: {curr_mean:.2f} (Δ={change:+.2f}, {pct:+.0f}%){marker}")
            else:
                print(f"    {prop:.0%}: {curr_mean:.2f}")
            prev_mean = curr_mean


def signal_assessment(df):
    """Overall assessment: is there enough signal for Batch 2?"""
    print("\n" + "="*80)
    print("SIGNAL ASSESSMENT — Go/No-Go for Batch 2")
    print("="*80)

    checks = []

    # 1. Mode distinction at baseline
    coh_0 = df[(df['mode'] == 'Coherence') & (df['ai_proportion'] == 0)]['recovery_time']
    ent_0 = df[(df['mode'] == 'Entrainment') & (df['ai_proportion'] == 0)]['recovery_time']
    if len(coh_0) > 0 and len(ent_0) > 0:
        ratio = ent_0.mean() / max(coh_0.mean(), 1)
        ok = ratio > 2
        checks.append(('Mode distinction at 0% AI', ok, f'{ratio:.1f}× recovery ratio'))
    else:
        checks.append(('Mode distinction at 0% AI', False, 'No data'))

    # 2. AI proportion effect in entrainment
    ent = df[df['mode'] == 'Entrainment']
    if len(ent[ent['ai_proportion'] == 0]) > 0:
        highest_prop = max(df['ai_proportion'].unique())
        ent_high = ent[ent['ai_proportion'] == highest_prop]['recovery_time']
        if len(ent_high) >= 3 and len(ent_0) >= 3:
            _, p = stats.mannwhitneyu(ent_0, ent_high, alternative='two-sided')
            ok = p < 0.1  # relaxed for exploratory
            checks.append(('Proportion effect (Ent: 0% vs highest)', ok, f'p={p:.4f}'))

    # 3. Coherence resilience
    coh = df[df['mode'] == 'Coherence']
    coh_high = coh[coh['ai_proportion'] == max(df['ai_proportion'].unique())]['recovery_time']
    if len(coh_0) > 0 and len(coh_high) > 0:
        coh_change = coh_high.mean() - coh_0.mean()
        ent_change = ent_high.mean() - ent_0.mean() if len(ent_high) > 0 else 0
        ok = abs(coh_change) < abs(ent_change)
        checks.append(('Coherence more resilient than entrainment', ok,
                       f'Coh Δ={coh_change:.1f} vs Ent Δ={ent_change:.1f}'))

    # 4. Cost asymmetry signal
    if 'human_ai_cost_ratio' in df.columns:
        ratio_vals = df[df['ai_proportion'] > 0]['human_ai_cost_ratio'].dropna()
        if len(ratio_vals) > 0:
            ok = ratio_vals.mean() != 1.0  # any deviation from parity
            checks.append(('Cost asymmetry detected', ok, f'mean ratio={ratio_vals.mean():.3f}'))

    print()
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}: {detail}")

    print()
    if all_ok:
        print("  → SIGNAL EXISTS. Proceed to Batch 2 (420 runs).")
    else:
        failed = [c[0] for c in checks if not c[1]]
        print(f"  → WEAK SIGNAL. Failed: {', '.join(failed)}")
        print("  → Review contingencies in TENCON_EXPERIMENT_PROTOCOL.md")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading H001 Batch 1 data...")
    df = load_h001_batch1(DATA_FILE)

    print(f"\nLoaded {len(df)} runs")
    print(f"Conditions: {df.groupby(['mode', 'ai_proportion']).size().to_string()}")
    print(f"Columns: {list(df.columns)}")

    # Core analyses
    summary_table(df)
    backward_compat_check(df)
    tipping_point_analysis(df)
    statistical_tests(df)
    signal_assessment(df)

    # Plots
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    plot_recovery_by_proportion(df)
    plot_dual_panel(df)
    plot_fatigue_panel(df)
    plot_human_vs_ai_variance(df)

    print("\nDone.")
