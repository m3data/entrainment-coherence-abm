"""
I003 Controlled Variance Sweep Analysis

Controlled experiment to establish causal relationship between
coupling-bias variance and spiral probability.

Design:
- 5 variance levels: 0.02, 0.04, 0.06, 0.08, 0.10
- 2 modes: entrainment (primary), coherence (control)
- 30 repetitions per condition (300 total runs)

Key metrics:
- Spiral rate per variance bin
- Median recovery time per bin
- 95th percentile recovery time (tail risk)
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


def binomial_confidence_interval(successes, n, confidence=0.95):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0, 0, 0
    p = successes / n
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator

    return p, max(0, center - margin), min(1, center + margin)


def load_i003_data(filepath):
    """
    Load I003 BehaviorSpace export.

    Expected format: 300 runs (5 variance × 2 modes × 30 reps)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find key variable lines
    entrainment_line_idx = None
    variance_line_idx = None

    for i, line in enumerate(lines):
        if '"entrainment-mode?"' in line:
            entrainment_line_idx = i
        if '"target-coupling-bias-variance"' in line:
            variance_line_idx = i

    if entrainment_line_idx is None:
        raise ValueError("Could not find entrainment-mode? line")
    if variance_line_idx is None:
        raise ValueError("Could not find target-coupling-bias-variance line")

    entrainment_values = lines[entrainment_line_idx].strip().split(',')
    variance_values = lines[variance_line_idx].strip().split(',')

    # Find the [final value] header line
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

    # Columns per run: 13 (step + 12 metrics)
    cols_per_run = 13
    total_cols = len(data_values)
    num_runs = (total_cols - 1) // cols_per_run

    print(f"Total runs detected: {num_runs}")

    metric_names = [
        'step',
        'initial_heading_variance',
        'initial_coupling_bias_mean',
        'initial_coupling_bias_variance',
        'initial_inertia_mean',
        'initial_inertia_variance',
        'pre_perturbation_cost',
        'pre_perturbation_variance',
        'recovery_time',
        'max_deviation',
        'mean_cumulative_cost',
        'mean_fatigue_level',
        'max_fatigue_level'
    ]

    runs = []
    for run_num in range(num_runs):
        start_idx = 1 + (run_num * cols_per_run)
        ent_idx = 1 + (run_num * cols_per_run)

        try:
            mode = entrainment_values[ent_idx].strip('"')
            target_var = float(variance_values[ent_idx].strip('"'))
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not get parameters for run {run_num + 1}: {e}")
            continue

        try:
            run_data = {
                'run': run_num + 1,
                'entrainment_mode': mode == 'true',
                'target_variance': target_var,
            }

            for i, metric in enumerate(metric_names):
                val = data_values[start_idx + i].strip('"')
                if metric == 'step':
                    run_data[metric] = int(float(val)) if val else 5000
                elif metric == 'recovery_time':
                    run_data[metric] = int(float(val)) if val and val != '-1' else -1
                else:
                    run_data[metric] = float(val) if val else 0.0

            runs.append(run_data)

        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse run {run_num + 1}: {e}")
            continue

    return pd.DataFrame(runs)


def analyze_by_variance_bin(df):
    """
    Analyze entrainment runs by target variance bin.

    Returns DataFrame with per-bin statistics.
    """
    entrainment = df[df['entrainment_mode'] == True].copy()

    # Define spiral threshold (same as I002: >500 ticks or never recovered)
    entrainment['spiral'] = (entrainment['recovery_time'] > 500) | (entrainment['recovery_time'] == -1)

    # Get unique target variances
    variance_levels = sorted(entrainment['target_variance'].unique())

    results = []
    for var in variance_levels:
        bin_data = entrainment[entrainment['target_variance'] == var]
        n = len(bin_data)
        n_spiral = bin_data['spiral'].sum()

        # Spiral rate with CI
        rate, ci_low, ci_high = binomial_confidence_interval(n_spiral, n)

        # Recovery times (excluding never-recovered)
        valid_recovery = bin_data[bin_data['recovery_time'] >= 0]['recovery_time']

        if len(valid_recovery) > 0:
            median_recovery = valid_recovery.median()
            p95_recovery = valid_recovery.quantile(0.95)
            mean_recovery = valid_recovery.mean()
            max_recovery = valid_recovery.max()
        else:
            median_recovery = np.nan
            p95_recovery = np.nan
            mean_recovery = np.nan
            max_recovery = np.nan

        # Realized variance (what we actually got after generation)
        realized_var_mean = bin_data['initial_coupling_bias_variance'].mean()
        realized_var_std = bin_data['initial_coupling_bias_variance'].std()

        # Fatigue metrics
        max_fatigue_mean = bin_data['max_fatigue_level'].mean()
        max_fatigue_std = bin_data['max_fatigue_level'].std()

        results.append({
            'target_variance': var,
            'n_runs': n,
            'n_spiral': n_spiral,
            'spiral_rate': rate,
            'spiral_ci_low': ci_low,
            'spiral_ci_high': ci_high,
            'median_recovery': median_recovery,
            'mean_recovery': mean_recovery,
            'p95_recovery': p95_recovery,
            'max_recovery': max_recovery,
            'realized_variance_mean': realized_var_mean,
            'realized_variance_std': realized_var_std,
            'max_fatigue_mean': max_fatigue_mean,
            'max_fatigue_std': max_fatigue_std,
        })

    return pd.DataFrame(results)


def analyze_coherence_control(df):
    """Verify coherence mode remains stable across all variance levels."""
    coherence = df[df['entrainment_mode'] == False].copy()
    coherence['spiral'] = (coherence['recovery_time'] > 500) | (coherence['recovery_time'] == -1)

    variance_levels = sorted(coherence['target_variance'].unique())

    results = []
    for var in variance_levels:
        bin_data = coherence[coherence['target_variance'] == var]
        n = len(bin_data)
        n_spiral = bin_data['spiral'].sum()
        max_fatigue = bin_data['max_fatigue_level'].max()
        mean_recovery = bin_data['recovery_time'].mean()

        results.append({
            'target_variance': var,
            'n_runs': n,
            'n_spiral': n_spiral,
            'max_fatigue': max_fatigue,
            'mean_recovery': mean_recovery
        })

    return pd.DataFrame(results)


def main(filepath=None):
    if filepath is None:
        filepath = "/Users/m3untold/Code/EarthianLabs/Netlogo-Models/coherence-model/exports/coherence_model_simple I003_controlled_variance_sweep-spreadsheet.csv"

    print("=" * 70)
    print("I003 CONTROLLED VARIANCE SWEEP ANALYSIS")
    print("=" * 70)
    print(f"\nLoading: {filepath}\n")

    try:
        df = load_i003_data(filepath)
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        print("\nPlease run the I003 experiment in NetLogo BehaviorSpace and export to:")
        print("  exports/coherence_model_simple I003_controlled_variance_sweep-spreadsheet.csv")
        return None, None
    except Exception as e:
        print(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    print(f"Loaded {len(df)} runs")

    # Split by mode
    coherence = df[df['entrainment_mode'] == False]
    entrainment = df[df['entrainment_mode'] == True]
    print(f"Coherence runs:   {len(coherence)}")
    print(f"Entrainment runs: {len(entrainment)}")

    # === ENTRAINMENT ANALYSIS BY VARIANCE BIN ===
    print("\n" + "=" * 70)
    print("ENTRAINMENT MODE: VARIANCE BIN ANALYSIS")
    print("=" * 70)

    bin_stats = analyze_by_variance_bin(df)

    print("\n{:^12} {:^8} {:^8} {:^20} {:^12} {:^12} {:^12}".format(
        "Target Var", "N", "Spirals", "Spiral Rate (95%CI)", "Median Rec", "P95 Rec", "Realized Var"
    ))
    print("-" * 95)

    for _, row in bin_stats.iterrows():
        ci_str = f"{row['spiral_rate']*100:.1f}% ({row['spiral_ci_low']*100:.1f}-{row['spiral_ci_high']*100:.1f})"
        med_str = f"{row['median_recovery']:.0f}" if not np.isnan(row['median_recovery']) else "N/A"
        p95_str = f"{row['p95_recovery']:.0f}" if not np.isnan(row['p95_recovery']) else "N/A"
        real_var_str = f"{row['realized_variance_mean']:.4f}"

        print(f"{row['target_variance']:^12.2f} {int(row['n_runs']):^8d} {int(row['n_spiral']):^8d} {ci_str:^20s} {med_str:^12s} {p95_str:^12s} {real_var_str:^12s}")

    # === DOSE-RESPONSE ANALYSIS ===
    print("\n" + "=" * 70)
    print("DOSE-RESPONSE RELATIONSHIP")
    print("=" * 70)

    # Correlation between variance and spiral rate
    if len(bin_stats) >= 3:
        r = np.corrcoef(bin_stats['target_variance'], bin_stats['spiral_rate'])[0, 1]
        print(f"\nCorrelation (target variance → spiral rate): r = {r:.3f}")

        # Linear regression
        slope, intercept = np.polyfit(bin_stats['target_variance'], bin_stats['spiral_rate'], 1)
        print(f"Linear trend: spiral_rate = {slope:.2f} × variance + {intercept:.3f}")
        print(f"Interpretation: Each 0.01 increase in variance → {slope*0.01*100:.1f}% increase in spiral rate")

    # === COHERENCE CONTROL ===
    print("\n" + "=" * 70)
    print("COHERENCE MODE CONTROL (Escape Valve Verification)")
    print("=" * 70)

    coherence_stats = analyze_coherence_control(df)

    total_spirals = coherence_stats['n_spiral'].sum()
    total_runs = coherence_stats['n_runs'].sum()
    max_fatigue_overall = coherence_stats['max_fatigue'].max()

    print(f"\nTotal coherence spirals: {total_spirals}/{total_runs} ({100*total_spirals/total_runs:.0f}%)")
    print(f"Max fatigue across all coherence runs: {max_fatigue_overall:.3f}")
    print("\n(Expected: 0 spirals, near-zero max fatigue - identity-pull escape valve)")

    # === STATISTICAL SUMMARY ===
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    # Threshold identification
    low_var = bin_stats[bin_stats['target_variance'] <= 0.04]
    high_var = bin_stats[bin_stats['target_variance'] >= 0.08]

    if len(low_var) > 0 and len(high_var) > 0:
        low_spiral_rate = low_var['n_spiral'].sum() / low_var['n_runs'].sum()
        high_spiral_rate = high_var['n_spiral'].sum() / high_var['n_runs'].sum()

        print(f"\nLow variance (≤0.04) spiral rate:  {low_spiral_rate*100:.1f}%")
        print(f"High variance (≥0.08) spiral rate: {high_spiral_rate*100:.1f}%")

        if high_spiral_rate > 0 and low_spiral_rate > 0:
            ratio = high_spiral_rate / low_spiral_rate
            print(f"Risk ratio (high/low): {ratio:.1f}×")
        elif high_spiral_rate > 0:
            print(f"Risk ratio: High variance has spirals, low variance has none")

    # Tail risk
    print("\nTail Risk (95th percentile recovery time):")
    for _, row in bin_stats.iterrows():
        if not np.isnan(row['p95_recovery']):
            print(f"  Variance {row['target_variance']:.2f}: {row['p95_recovery']:.0f} ticks")

    print("\n" + "=" * 70)
    print("END ANALYSIS")
    print("=" * 70)

    return df, bin_stats


def create_visualizations(df, bin_stats, output_dir):
    """Create publication-quality I003 visualizations."""

    output_dir = Path(output_dir)

    # =========================================================================
    # Figure 1: Main Result - Spiral Rate by Variance (with CI)
    # =========================================================================
    fig1, ax = plt.subplots(figsize=(10, 6))

    x = bin_stats['target_variance']
    y = bin_stats['spiral_rate'] * 100
    yerr_low = (bin_stats['spiral_rate'] - bin_stats['spiral_ci_low']) * 100
    yerr_high = (bin_stats['spiral_ci_high'] - bin_stats['spiral_rate']) * 100

    # Bar chart with error bars
    bars = ax.bar(x, y, width=0.015, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt='none', color='black', capsize=5, capthick=2, linewidth=2)

    # Add n values on bars
    for i, (xi, yi, n) in enumerate(zip(x, y, bin_stats['n_runs'])):
        ax.text(xi, yi + yerr_high.iloc[i] + 3, f'n={n}', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Target Coupling-Bias Variance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spiral Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('I003: Spiral Rate by Controlled Variance\n(Entrainment Mode, 95% CI)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 0.12)

    # Add trend line
    if len(bin_stats) >= 3:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0.02, 0.10, 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2, label=f'Trend (r={np.corrcoef(x, y)[0,1]:.2f})')
        ax.legend(loc='upper left')

    plt.tight_layout()
    fig1.savefig(output_dir / 'I003_spiral_rate_by_variance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'I003_spiral_rate_by_variance.png'}")

    # =========================================================================
    # Figure 2: Recovery Time Distribution
    # =========================================================================
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Median recovery
    ax1 = axes[0]
    ax1.bar(bin_stats['target_variance'], bin_stats['median_recovery'],
            width=0.015, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Target Coupling-Bias Variance', fontsize=11)
    ax1.set_ylabel('Median Recovery Time (ticks)', fontsize=11)
    ax1.set_title('Median Recovery Time', fontsize=12, fontweight='bold')

    # 95th percentile (tail risk)
    ax2 = axes[1]
    ax2.bar(bin_stats['target_variance'], bin_stats['p95_recovery'],
            width=0.015, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Target Coupling-Bias Variance', fontsize=11)
    ax2.set_ylabel('95th Percentile Recovery Time (ticks)', fontsize=11)
    ax2.set_title('Tail Risk (P95 Recovery)', fontsize=12, fontweight='bold')

    fig2.suptitle('I003: Recovery Time by Variance Level', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig2.savefig(output_dir / 'I003_recovery_time_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'I003_recovery_time_distribution.png'}")

    # =========================================================================
    # Figure 3: Mode Comparison (Coherence as Control)
    # =========================================================================
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

    entrainment = df[df['entrainment_mode'] == True]
    coherence = df[df['entrainment_mode'] == False]

    variance_levels = sorted(df['target_variance'].unique())

    # Spiral rate comparison
    ax1 = axes[0]
    x_pos = np.arange(len(variance_levels))
    width = 0.35

    ent_rates = []
    coh_rates = []
    for var in variance_levels:
        ent_bin = entrainment[entrainment['target_variance'] == var]
        coh_bin = coherence[coherence['target_variance'] == var]

        ent_spiral = ((ent_bin['recovery_time'] > 500) | (ent_bin['recovery_time'] == -1)).sum()
        coh_spiral = ((coh_bin['recovery_time'] > 500) | (coh_bin['recovery_time'] == -1)).sum()

        ent_rates.append(100 * ent_spiral / len(ent_bin) if len(ent_bin) > 0 else 0)
        coh_rates.append(100 * coh_spiral / len(coh_bin) if len(coh_bin) > 0 else 0)

    ax1.bar(x_pos - width/2, ent_rates, width, label='Entrainment', color='#e74c3c', alpha=0.8)
    ax1.bar(x_pos + width/2, coh_rates, width, label='Coherence', color='#27ae60', alpha=0.8)
    ax1.set_xlabel('Target Variance', fontsize=11)
    ax1.set_ylabel('Spiral Rate (%)', fontsize=11)
    ax1.set_title('Spiral Rate: Entrainment vs Coherence', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{v:.2f}' for v in variance_levels])
    ax1.legend()
    ax1.set_ylim(0, 100)

    # Max fatigue comparison
    ax2 = axes[1]
    ent_fatigue = []
    coh_fatigue = []
    for var in variance_levels:
        ent_bin = entrainment[entrainment['target_variance'] == var]
        coh_bin = coherence[coherence['target_variance'] == var]
        ent_fatigue.append(ent_bin['max_fatigue_level'].mean())
        coh_fatigue.append(coh_bin['max_fatigue_level'].mean())

    ax2.bar(x_pos - width/2, ent_fatigue, width, label='Entrainment', color='#e74c3c', alpha=0.8)
    ax2.bar(x_pos + width/2, coh_fatigue, width, label='Coherence', color='#27ae60', alpha=0.8)
    ax2.set_xlabel('Target Variance', fontsize=11)
    ax2.set_ylabel('Mean Max Fatigue', fontsize=11)
    ax2.set_title('Fatigue: Entrainment vs Coherence', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{v:.2f}' for v in variance_levels])
    ax2.legend()

    fig3.suptitle('I003: Mode Comparison (Coherence = Escape Valve Control)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig3.savefig(output_dir / 'I003_mode_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'I003_mode_comparison.png'}")

    # =========================================================================
    # Figure 4: Summary Dashboard
    # =========================================================================
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('I003: Controlled Variance Sweep - Summary Dashboard', fontsize=16, fontweight='bold')

    # Panel A: Spiral rate with CI
    ax = axes[0, 0]
    ax.errorbar(bin_stats['target_variance'], bin_stats['spiral_rate']*100,
                yerr=[(bin_stats['spiral_rate'] - bin_stats['spiral_ci_low'])*100,
                      (bin_stats['spiral_ci_high'] - bin_stats['spiral_rate'])*100],
                fmt='o-', color='#e74c3c', markersize=10, linewidth=2, capsize=5, capthick=2)
    ax.set_xlabel('Target Variance')
    ax.set_ylabel('Spiral Rate (%)')
    ax.set_title('A. Spiral Rate (95% CI)', fontweight='bold')
    ax.set_ylim(0, 100)

    # Panel B: Median + P95 recovery
    ax = axes[0, 1]
    ax.plot(bin_stats['target_variance'], bin_stats['median_recovery'], 'o-',
            color='#3498db', markersize=10, linewidth=2, label='Median')
    ax.plot(bin_stats['target_variance'], bin_stats['p95_recovery'], 's--',
            color='#9b59b6', markersize=10, linewidth=2, label='95th Percentile')
    ax.set_xlabel('Target Variance')
    ax.set_ylabel('Recovery Time (ticks)')
    ax.set_title('B. Recovery Time Distribution', fontweight='bold')
    ax.legend()

    # Panel C: Individual run scatter
    ax = axes[1, 0]
    entrainment = df[df['entrainment_mode'] == True].copy()
    entrainment['spiral'] = (entrainment['recovery_time'] > 500) | (entrainment['recovery_time'] == -1)

    non_spiral = entrainment[~entrainment['spiral']]
    spiral = entrainment[entrainment['spiral']]

    ax.scatter(non_spiral['target_variance'] + np.random.normal(0, 0.002, len(non_spiral)),
               non_spiral['recovery_time'], c='#4ECDC4', alpha=0.6, s=50, label='Non-spiral')

    spiral_y = spiral['recovery_time'].replace(-1, 5000)
    ax.scatter(spiral['target_variance'] + np.random.normal(0, 0.002, len(spiral)),
               spiral_y, c='#FF6B6B', alpha=0.8, s=100, marker='X', label='Spiral')

    ax.axhline(y=500, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Target Variance')
    ax.set_ylabel('Recovery Time (ticks)')
    ax.set_title('C. Individual Runs (Entrainment)', fontweight='bold')
    ax.legend()
    ax.set_ylim(-100, 5500)

    # Panel D: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate key stats
    total_ent = len(entrainment)
    total_spiral = entrainment['spiral'].sum()

    low_var_ent = entrainment[entrainment['target_variance'] <= 0.04]
    high_var_ent = entrainment[entrainment['target_variance'] >= 0.08]

    low_spiral = low_var_ent['spiral'].sum() / len(low_var_ent) * 100 if len(low_var_ent) > 0 else 0
    high_spiral = high_var_ent['spiral'].sum() / len(high_var_ent) * 100 if len(high_var_ent) > 0 else 0

    coherence = df[df['entrainment_mode'] == False]
    coh_spiral = ((coherence['recovery_time'] > 500) | (coherence['recovery_time'] == -1)).sum()

    summary = f"""
    I003 KEY FINDINGS

    Total entrainment runs: {total_ent}
    Total spirals: {total_spiral} ({100*total_spiral/total_ent:.1f}%)

    Low variance (≤0.04):  {low_spiral:.1f}% spiral
    High variance (≥0.08): {high_spiral:.1f}% spiral

    Risk ratio: {high_spiral/low_spiral:.1f}× (high vs low)

    COHERENCE CONTROL
    Spirals: {coh_spiral}/{len(coherence)} ({100*coh_spiral/len(coherence):.0f}%)
    Max fatigue: {coherence['max_fatigue_level'].max():.3f}

    INTERPRETATION
    Coupling-bias variance causally
    increases spiral probability.
    Identity-pull (coherence mode)
    provides complete protection.
    """

    ax.text(0.1, 0.5, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig4.savefig(output_dir / 'I003_summary_dashboard.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'I003_summary_dashboard.png'}")

    plt.close('all')
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = None

    df, bin_stats = main(filepath)

    if df is not None and bin_stats is not None:
        output_dir = "/Users/m3untold/Code/EarthianLabs/Netlogo-Models/coherence-model/exports"
        create_visualizations(df, bin_stats, output_dir)
        print("\nVisualization complete!")
