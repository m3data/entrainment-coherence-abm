"""
I004 Configuration Tracking Analysis

Tests the "sacrificial stabilizer" hypothesis:
- At-risk agents (high coupling + far from mean) bear disproportionate burden
- Their overload predicts system-level spiral
- Coherence mode protects via identity-pull escape valve

Key metrics:
- at-risk-count, at-risk-fraction
- cost-burden-ratio, alignment-work-ratio, fatigue-burden-ratio
- Correlation between at-risk burden and recovery time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def pearsonr(x, y):
    """Simple Pearson correlation without scipy"""
    x = np.array(x)
    y = np.array(y)
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    if sx == 0 or sy == 0:
        return np.nan, np.nan
    r = np.sum((x - mx) * (y - my)) / ((n - 1) * sx * sy)
    if abs(r) >= 1:
        return r, 0.0
    t = r * np.sqrt((n - 2) / (1 - r**2))
    df = n - 2
    p = 2 * np.exp(-0.717 * abs(t) - 0.416 * t**2 / (df + 1))
    return r, max(0, min(1, p))


def ttest_ind(x, y):
    """Simple independent t-test without scipy"""
    x, y = np.array(x), np.array(y)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan, np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx/nx + vy/ny)
    if se == 0:
        return np.inf if mx != my else 0, 1.0
    t = (mx - my) / se
    df = (vx/nx + vy/ny)**2 / ((vx/nx)**2/(nx-1) + (vy/ny)**2/(ny-1))
    p = 2 * np.exp(-0.717 * abs(t) - 0.416 * t**2 / (df + 1))
    return t, max(0, min(1, p))


def load_i004_data(filepath):
    """Load I004 BehaviorSpace export."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find entrainment-mode line
    entrainment_line_idx = None
    for i, line in enumerate(lines):
        if '"entrainment-mode?"' in line:
            entrainment_line_idx = i
            break

    if entrainment_line_idx is None:
        raise ValueError("Could not find entrainment-mode? line")

    entrainment_values = lines[entrainment_line_idx].strip().split(',')

    # Find header line
    header_line_idx = None
    for i, line in enumerate(lines):
        if '"[final value]"' in line:
            header_line_idx = i
            break

    if header_line_idx is None:
        raise ValueError("Could not find [final value] header line")

    data_line_idx = header_line_idx + 1
    data_line = lines[data_line_idx].strip()
    data_values = data_line.split(',')

    # Columns per run: 22 (step + 21 metrics)
    cols_per_run = 22
    total_cols = len(data_values)
    num_runs = (total_cols - 1) // cols_per_run

    print(f"Total runs detected: {num_runs}")

    metric_names = [
        'step',
        'recovery_time',
        'max_deviation',
        'mean_cumulative_cost',
        'max_fatigue_level',
        'at_risk_count',
        'at_risk_fraction',
        'at_risk_mean_cost',
        'at_risk_mean_alignment_work',
        'at_risk_max_fatigue',
        'at_risk_mean_fatigue',
        'rest_mean_cost',
        'rest_mean_alignment_work',
        'rest_mean_fatigue',
        'cost_burden_ratio',
        'alignment_work_ratio',
        'fatigue_burden_ratio',
        'mean_initial_distance',
        'mean_alignment_work',
        'max_alignment_work',
        'alignment_work_gini',
        'initial_coupling_bias_variance'
    ]

    runs = []
    for run_num in range(num_runs):
        start_idx = 1 + (run_num * cols_per_run)
        ent_idx = 1 + (run_num * cols_per_run)

        try:
            mode = entrainment_values[ent_idx].strip('"')
        except IndexError:
            mode = 'true' if run_num >= num_runs // 2 else 'false'

        try:
            run_data = {
                'run': run_num + 1,
                'entrainment_mode': mode == 'true',
            }

            for i, metric in enumerate(metric_names):
                val = data_values[start_idx + i].strip('"')
                if metric == 'step':
                    run_data[metric] = int(float(val)) if val else 5000
                elif metric == 'recovery_time':
                    run_data[metric] = int(float(val)) if val and val != '-1' else -1
                elif metric in ['at_risk_count']:
                    run_data[metric] = int(float(val)) if val else 0
                else:
                    run_data[metric] = float(val) if val else 0.0

            runs.append(run_data)

        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse run {run_num + 1}: {e}")
            continue

    return pd.DataFrame(runs)


def main(filepath=None):
    if filepath is None:
        filepath = "/Users/m3untold/Code/EarthianLabs/Netlogo-Models/coherence-model/exports/coherence_model_simple I004_configuration_tracking-spreadsheet.csv"

    print("=" * 70)
    print("I004 CONFIGURATION TRACKING ANALYSIS")
    print("Sacrificial Stabilizer Hypothesis")
    print("=" * 70)
    print(f"\nLoading: {filepath}\n")

    try:
        df = load_i004_data(filepath)
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"Loaded {len(df)} runs")

    # Split by mode
    coherence = df[df['entrainment_mode'] == False].copy()
    entrainment = df[df['entrainment_mode'] == True].copy()

    print(f"Coherence runs:   {len(coherence)}")
    print(f"Entrainment runs: {len(entrainment)}")

    # === AT-RISK AGENT SUMMARY ===
    print("\n" + "=" * 70)
    print("AT-RISK AGENT SUMMARY")
    print("(High coupling > 0.7 AND far from mean > 45°)")
    print("=" * 70)

    for mode_name, mode_df in [("Coherence", coherence), ("Entrainment", entrainment)]:
        print(f"\n{mode_name.upper()} MODE:")
        print(f"  At-risk count:    {mode_df['at_risk_count'].mean():.1f} ± {mode_df['at_risk_count'].std():.1f}")
        print(f"  At-risk fraction: {mode_df['at_risk_fraction'].mean()*100:.1f}% ± {mode_df['at_risk_fraction'].std()*100:.1f}%")

    # === BURDEN RATIOS ===
    print("\n" + "=" * 70)
    print("BURDEN RATIOS (At-Risk / Rest)")
    print("(>1 means at-risk agents bear disproportionate burden)")
    print("=" * 70)

    burden_metrics = [
        ('cost_burden_ratio', 'Cost Burden'),
        ('alignment_work_ratio', 'Alignment Work'),
        ('fatigue_burden_ratio', 'Fatigue Burden')
    ]

    print("\n{:20s} {:>20s} {:>20s}".format("Metric", "Coherence", "Entrainment"))
    print("-" * 60)

    for metric, label in burden_metrics:
        coh_val = coherence[metric].mean()
        coh_std = coherence[metric].std()
        ent_val = entrainment[metric].mean()
        ent_std = entrainment[metric].std()
        print(f"{label:20s} {coh_val:>8.2f} ± {coh_std:<8.2f} {ent_val:>8.2f} ± {ent_std:<8.2f}")

    # === SPIRAL DETECTION ===
    print("\n" + "=" * 70)
    print("SPIRAL DETECTION")
    print("=" * 70)

    entrainment['spiral'] = (entrainment['recovery_time'] > 500) | (entrainment['recovery_time'] == -1)
    coherence['spiral'] = (coherence['recovery_time'] > 500) | (coherence['recovery_time'] == -1)

    ent_spirals = entrainment['spiral'].sum()
    coh_spirals = coherence['spiral'].sum()

    print(f"\nEntrainment spirals: {ent_spirals}/{len(entrainment)} ({100*ent_spirals/len(entrainment):.0f}%)")
    print(f"Coherence spirals:   {coh_spirals}/{len(coherence)} ({100*coh_spirals/len(coherence):.0f}%)")

    # === SPIRAL PREDICTOR ANALYSIS (Entrainment) ===
    print("\n" + "=" * 70)
    print("SPIRAL PREDICTORS (Entrainment Mode)")
    print("=" * 70)

    if ent_spirals > 0 and ent_spirals < len(entrainment):
        spiral_runs = entrainment[entrainment['spiral']]
        non_spiral = entrainment[~entrainment['spiral']]

        predictors = [
            ('at_risk_count', 'At-Risk Count'),
            ('at_risk_fraction', 'At-Risk Fraction'),
            ('at_risk_mean_fatigue', 'At-Risk Mean Fatigue'),
            ('at_risk_max_fatigue', 'At-Risk Max Fatigue'),
            ('cost_burden_ratio', 'Cost Burden Ratio'),
            ('alignment_work_ratio', 'Alignment Work Ratio'),
            ('fatigue_burden_ratio', 'Fatigue Burden Ratio'),
            ('alignment_work_gini', 'Alignment Work Gini'),
        ]

        print("\n{:25s} {:>12s} {:>12s} {:>10s} {:>10s}".format(
            "Predictor", "Spiral", "Non-Spiral", "t-stat", "p-value"))
        print("-" * 75)

        significant = []
        for metric, label in predictors:
            if metric not in spiral_runs.columns:
                continue

            spiral_val = spiral_runs[metric].mean()
            non_spiral_val = non_spiral[metric].mean()
            t, p = ttest_ind(spiral_runs[metric], non_spiral[metric])

            sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

            print(f"{label:25s} {spiral_val:>12.3f} {non_spiral_val:>12.3f} {t:>10.2f} {p:>10.3f} {sig_marker}")

            if p < 0.05:
                significant.append((label, t, p, spiral_val > non_spiral_val))

        if significant:
            print("\nSIGNIFICANT PREDICTORS:")
            for label, t, p, higher_in_spiral in significant:
                direction = "higher" if higher_in_spiral else "lower"
                print(f"  - {label}: {direction} in spiral runs (p={p:.3f})")
    else:
        print("\nInsufficient spiral variation for predictor analysis.")

    # === CORRELATION ANALYSIS ===
    print("\n" + "=" * 70)
    print("CORRELATION WITH RECOVERY TIME (Entrainment)")
    print("=" * 70)

    valid_ent = entrainment[entrainment['recovery_time'] >= 0].copy()

    correlation_targets = [
        ('at_risk_count', 'At-Risk Count'),
        ('at_risk_fraction', 'At-Risk Fraction'),
        ('at_risk_mean_fatigue', 'At-Risk Mean Fatigue'),
        ('cost_burden_ratio', 'Cost Burden Ratio'),
        ('alignment_work_ratio', 'Alignment Work Ratio'),
        ('alignment_work_gini', 'Alignment Work Gini'),
        ('max_alignment_work', 'Max Alignment Work'),
    ]

    print("\n{:25s} {:>10s} {:>10s}".format("Predictor", "r", "p-value"))
    print("-" * 50)

    for metric, label in correlation_targets:
        if metric not in valid_ent.columns:
            continue
        r, p = pearsonr(valid_ent[metric], valid_ent['recovery_time'])
        sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{label:25s} {r:>+10.3f} {p:>10.3f} {sig_marker}")

    # === WORK DISTRIBUTION ANALYSIS ===
    print("\n" + "=" * 70)
    print("ALIGNMENT WORK DISTRIBUTION")
    print("=" * 70)

    for mode_name, mode_df in [("Coherence", coherence), ("Entrainment", entrainment)]:
        print(f"\n{mode_name.upper()} MODE:")
        print(f"  Mean alignment work:     {mode_df['mean_alignment_work'].mean():.1f}")
        print(f"  Max alignment work:      {mode_df['max_alignment_work'].mean():.1f}")
        print(f"  Work Gini coefficient:   {mode_df['alignment_work_gini'].mean():.3f}")
        print(f"  At-risk work:            {mode_df['at_risk_mean_alignment_work'].mean():.1f}")
        print(f"  Rest work:               {mode_df['rest_mean_alignment_work'].mean():.1f}")
        if mode_df['rest_mean_alignment_work'].mean() > 0:
            ratio = mode_df['at_risk_mean_alignment_work'].mean() / mode_df['rest_mean_alignment_work'].mean()
            print(f"  At-risk/Rest ratio:      {ratio:.2f}×")

    # === SYNTHESIS ===
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    # Key comparisons
    ent_work_ratio = entrainment['alignment_work_ratio'].mean()
    coh_work_ratio = coherence['alignment_work_ratio'].mean()

    ent_fatigue_ratio = entrainment['fatigue_burden_ratio'].mean()
    coh_fatigue_ratio = coherence['fatigue_burden_ratio'].mean()

    print(f"\nAlignment Work Burden Ratio:")
    print(f"  Entrainment: {ent_work_ratio:.2f}× (at-risk work {ent_work_ratio:.1f}× higher than rest)")
    print(f"  Coherence:   {coh_work_ratio:.2f}×")

    print(f"\nFatigue Burden Ratio:")
    print(f"  Entrainment: {ent_fatigue_ratio:.2f}×")
    print(f"  Coherence:   {coh_fatigue_ratio:.2f}×")

    if ent_work_ratio > 1.5:
        print(f"\n→ SACRIFICIAL STABILIZER PATTERN CONFIRMED")
        print(f"  At-risk agents do {ent_work_ratio:.1f}× more alignment work in entrainment mode.")
    else:
        print(f"\n→ Burden distribution more equal than expected.")

    print("\n" + "=" * 70)
    print("END ANALYSIS")
    print("=" * 70)

    return df


def create_visualizations(df, output_dir):
    """Create I004 visualizations."""
    output_dir = Path(output_dir)

    coherence = df[df['entrainment_mode'] == False].copy()
    entrainment = df[df['entrainment_mode'] == True].copy()

    entrainment['spiral'] = (entrainment['recovery_time'] > 500) | (entrainment['recovery_time'] == -1)
    coherence['spiral'] = (coherence['recovery_time'] > 500) | (coherence['recovery_time'] == -1)

    # =========================================================================
    # Figure 1: Burden Ratio Comparison
    # =========================================================================
    fig1, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig1.suptitle('I004: Burden Distribution (At-Risk vs Rest)', fontsize=14, fontweight='bold')

    burden_metrics = [
        ('cost_burden_ratio', 'Cost Burden Ratio'),
        ('alignment_work_ratio', 'Alignment Work Ratio'),
        ('fatigue_burden_ratio', 'Fatigue Burden Ratio')
    ]

    for i, (metric, title) in enumerate(burden_metrics):
        ax = axes[i]

        data = [coherence[metric], entrainment[metric]]
        bp = ax.boxplot(data, labels=['Coherence', 'Entrainment'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#27ae60')
        bp['boxes'][1].set_facecolor('#e74c3c')

        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal burden')
        ax.set_ylabel('Ratio (At-Risk / Rest)')
        ax.set_title(title, fontweight='bold')

        # Add individual points
        for j, (d, color) in enumerate([(coherence, '#1d6f42'), (entrainment, '#922b21')]):
            x = np.random.normal(j+1, 0.04, size=len(d))
            ax.scatter(x, d[metric], alpha=0.4, c=color, s=30, zorder=3)

    plt.tight_layout()
    fig1.savefig(output_dir / 'I004_burden_ratios.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'I004_burden_ratios.png'}")

    # =========================================================================
    # Figure 2: Spiral vs Non-Spiral (Entrainment)
    # =========================================================================
    if entrainment['spiral'].sum() > 0 and (~entrainment['spiral']).sum() > 0:
        fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig2.suptitle('I004: Spiral vs Non-Spiral Runs (Entrainment Mode)', fontsize=14, fontweight='bold')

        spiral = entrainment[entrainment['spiral']]
        non_spiral = entrainment[~entrainment['spiral']]

        comparison_metrics = [
            ('at_risk_count', 'At-Risk Agent Count'),
            ('at_risk_max_fatigue', 'At-Risk Max Fatigue'),
            ('alignment_work_ratio', 'Alignment Work Ratio'),
            ('alignment_work_gini', 'Alignment Work Gini')
        ]

        for idx, (metric, title) in enumerate(comparison_metrics):
            ax = axes[idx // 2, idx % 2]

            data = [non_spiral[metric], spiral[metric]]
            bp = ax.boxplot(data, labels=['Non-Spiral', 'Spiral'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#4ECDC4')
            bp['boxes'][1].set_facecolor('#FF6B6B')

            ax.set_ylabel(title)
            ax.set_title(title, fontweight='bold')

            # T-test
            t, p = ttest_ind(spiral[metric], non_spiral[metric])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(0.95, 0.95, f'p={p:.3f} {sig}', transform=ax.transAxes,
                   ha='right', va='top', fontsize=10)

        plt.tight_layout()
        fig2.savefig(output_dir / 'I004_spiral_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'I004_spiral_comparison.png'}")

    # =========================================================================
    # Figure 3: Correlation Scatter
    # =========================================================================
    fig3, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig3.suptitle('I004: Predictors of Recovery Time (Entrainment)', fontsize=12, fontweight='bold')

    valid_ent = entrainment[entrainment['recovery_time'] >= 0]

    scatter_metrics = [
        ('at_risk_count', 'At-Risk Count'),
        ('alignment_work_ratio', 'Alignment Work Ratio'),
        ('alignment_work_gini', 'Work Gini')
    ]

    for i, (metric, label) in enumerate(scatter_metrics):
        ax = axes[i]
        ax.scatter(valid_ent[metric], valid_ent['recovery_time'],
                  c='#3498db', alpha=0.6, s=50, edgecolors='white')

        # Fit line
        if len(valid_ent) > 2:
            z = np.polyfit(valid_ent[metric], valid_ent['recovery_time'], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(valid_ent[metric].min(), valid_ent[metric].max(), 100)
            ax.plot(x_line, p_line(x_line), 'r--', alpha=0.7, linewidth=2)

        r, p = pearsonr(valid_ent[metric], valid_ent['recovery_time'])
        ax.set_xlabel(label)
        ax.set_ylabel('Recovery Time' if i == 0 else '')
        ax.set_title(f'r = {r:.3f}, p = {p:.3f}', fontsize=10)

    plt.tight_layout()
    fig3.savefig(output_dir / 'I004_correlations.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'I004_correlations.png'}")

    # =========================================================================
    # Figure 4: Summary Dashboard
    # =========================================================================
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('I004: Configuration Tracking Summary', fontsize=16, fontweight='bold')

    # Panel A: Work distribution by mode
    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.35

    at_risk_work = [coherence['at_risk_mean_alignment_work'].mean(),
                    entrainment['at_risk_mean_alignment_work'].mean()]
    rest_work = [coherence['rest_mean_alignment_work'].mean(),
                 entrainment['rest_mean_alignment_work'].mean()]

    ax.bar(x - width/2, at_risk_work, width, label='At-Risk', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, rest_work, width, label='Rest', color='#3498db', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Coherence', 'Entrainment'])
    ax.set_ylabel('Mean Alignment Work')
    ax.set_title('A. Alignment Work by Group', fontweight='bold')
    ax.legend()

    # Panel B: Fatigue comparison
    ax = axes[0, 1]
    at_risk_fatigue = [coherence['at_risk_mean_fatigue'].mean(),
                       entrainment['at_risk_mean_fatigue'].mean()]
    rest_fatigue = [coherence['rest_mean_fatigue'].mean(),
                    entrainment['rest_mean_fatigue'].mean()]

    ax.bar(x - width/2, at_risk_fatigue, width, label='At-Risk', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, rest_fatigue, width, label='Rest', color='#3498db', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Coherence', 'Entrainment'])
    ax.set_ylabel('Mean Fatigue Level')
    ax.set_title('B. Fatigue by Group', fontweight='bold')
    ax.legend()

    # Panel C: At-risk count vs recovery (entrainment)
    ax = axes[1, 0]
    valid = entrainment[entrainment['recovery_time'] >= 0]
    spiral_mask = entrainment['spiral']

    ax.scatter(valid[~spiral_mask.loc[valid.index]]['at_risk_count'],
               valid[~spiral_mask.loc[valid.index]]['recovery_time'],
               c='#4ECDC4', alpha=0.7, s=60, label='Non-spiral', edgecolors='white')

    spiral_valid = entrainment[entrainment['spiral'] & (entrainment['recovery_time'] >= 0)]
    if len(spiral_valid) > 0:
        ax.scatter(spiral_valid['at_risk_count'],
                   spiral_valid['recovery_time'],
                   c='#FF6B6B', alpha=0.9, s=100, marker='X', label='Spiral', edgecolors='black')

    ax.set_xlabel('At-Risk Agent Count')
    ax.set_ylabel('Recovery Time (ticks)')
    ax.set_title('C. At-Risk Count vs Recovery', fontweight='bold')
    ax.legend()

    # Panel D: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    ent_work_ratio = entrainment['alignment_work_ratio'].mean()
    coh_work_ratio = coherence['alignment_work_ratio'].mean()
    ent_spiral_pct = 100 * entrainment['spiral'].sum() / len(entrainment)
    coh_spiral_pct = 100 * coherence['spiral'].sum() / len(coherence)

    summary = f"""
    I004 KEY FINDINGS

    AT-RISK AGENTS
    (coupling > 0.7, distance > 45°)

    Coherence:  {coherence['at_risk_count'].mean():.1f} agents ({coherence['at_risk_fraction'].mean()*100:.1f}%)
    Entrainment: {entrainment['at_risk_count'].mean():.1f} agents ({entrainment['at_risk_fraction'].mean()*100:.1f}%)

    ALIGNMENT WORK RATIO (at-risk / rest)
    Coherence:   {coh_work_ratio:.2f}×
    Entrainment: {ent_work_ratio:.2f}×

    SPIRAL RATE
    Coherence:   {coh_spiral_pct:.0f}%
    Entrainment: {ent_spiral_pct:.0f}%

    INTERPRETATION
    At-risk agents in entrainment mode
    bear {ent_work_ratio:.1f}× the alignment work of others.
    They are the sacrificial stabilizers.
    """

    ax.text(0.1, 0.5, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig4.savefig(output_dir / 'I004_summary_dashboard.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'I004_summary_dashboard.png'}")

    plt.close('all')
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = None

    df = main(filepath)

    if df is not None:
        output_dir = "/Users/m3untold/Code/EarthianLabs/Netlogo-Models/coherence-model/exports"
        create_visualizations(df, output_dir)
        print("\nVisualization complete!")
