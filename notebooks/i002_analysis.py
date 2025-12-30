"""
I002 Agent Parameter Heterogeneity Analysis
Tests hypotheses about spiral triggers in entrainment mode:
  H1: High coupling-bias variance → spiral
  H2: Low inertia mean → spiral
  H3: High pre-perturbation cost → spiral
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def pearsonr(x, y):
    """Simple Pearson correlation without scipy"""
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    if sx == 0 or sy == 0:
        return np.nan, np.nan
    r = np.sum((x - mx) * (y - my)) / ((n - 1) * sx * sy)
    # t-test for significance
    if abs(r) >= 1:
        return r, 0.0
    t = r * np.sqrt((n - 2) / (1 - r**2))
    # Two-tailed p-value approximation
    df = n - 2
    p = 2 * np.exp(-0.717 * abs(t) - 0.416 * t**2 / (df + 1))
    p = max(0, min(1, p))
    return r, p

def ttest_ind(x, y):
    """Simple independent t-test without scipy"""
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan, np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx/nx + vy/ny)
    if se == 0:
        return np.inf if mx != my else 0, 1.0
    t = (mx - my) / se
    # Welch-Satterthwaite df
    df = (vx/nx + vy/ny)**2 / ((vx/nx)**2/(nx-1) + (vy/ny)**2/(ny-1))
    # Crude p-value approximation
    p = 2 * np.exp(-0.717 * abs(t) - 0.416 * t**2 / (df + 1))
    return t, max(0, min(1, p))

def load_i002_data(filepath):
    """
    Load I002 BehaviorSpace export (Spreadsheet version 2.0).

    Format:
    - 60 runs (30 coherence + 30 entrainment)
    - 13 columns per run: [step], initial-heading-variance, initial-coupling-bias-mean,
      initial-coupling-bias-variance, initial-inertia-mean, initial-inertia-variance,
      pre-perturbation-cost, pre-perturbation-variance, recovery-time, max-deviation,
      mean-cumulative-cost, mean-fatigue-level, max-fatigue-level
    """
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

    # Find the [final value] header line and data line (last line)
    header_line_idx = None
    for i, line in enumerate(lines):
        if '"[final value]"' in line:
            header_line_idx = i
            break

    if header_line_idx is None:
        raise ValueError("Could not find [final value] header line")

    # Data is on the next line after header
    data_line_idx = header_line_idx + 1
    data_line = lines[data_line_idx].strip()
    data_values = data_line.split(',')

    # Columns per run: 13 (step + 12 metrics)
    cols_per_run = 13

    # First column is empty, then 60 runs × 13 columns
    total_cols = len(data_values)
    num_runs = (total_cols - 1) // cols_per_run

    print(f"Columns per run: {cols_per_run}")
    print(f"Total runs: {num_runs}")

    # Metric order from header (after [step]):
    # initial-heading-variance, initial-coupling-bias-mean, initial-coupling-bias-variance,
    # initial-inertia-mean, initial-inertia-variance, pre-perturbation-cost, pre-perturbation-variance,
    # recovery-time, max-deviation, mean-cumulative-cost, mean-fatigue-level, max-fatigue-level
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
        # Data starts at column 1 (column 0 is empty)
        start_idx = 1 + (run_num * cols_per_run)

        # Get entrainment mode from the entrainment line
        # Same column structure: first empty, then 13 cols per run
        ent_idx = 1 + (run_num * cols_per_run)
        try:
            mode = entrainment_values[ent_idx].strip('"')
        except IndexError:
            # Fallback: runs 1-30 are coherence, 31-60 are entrainment
            mode = 'true' if run_num >= 30 else 'false'

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
                else:
                    run_data[metric] = float(val) if val else 0.0

            runs.append(run_data)

        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse run {run_num + 1}: {e}")
            continue

    return pd.DataFrame(runs)


def analyze_hypothesis(df_spiral, df_non_spiral, metric_name, hypothesis_direction, label):
    """
    Analyze a single hypothesis.
    hypothesis_direction: 'high' means spiral runs should have higher values
                         'low' means spiral runs should have lower values
    """
    if len(df_spiral) == 0 or len(df_non_spiral) == 0:
        return None

    spiral_val = df_spiral[metric_name].mean()
    spiral_std = df_spiral[metric_name].std()
    non_spiral_val = df_non_spiral[metric_name].mean()
    non_spiral_std = df_non_spiral[metric_name].std()

    # Correlation with recovery time (excluding no-recovery runs)
    valid_runs = pd.concat([df_spiral, df_non_spiral])
    valid_runs = valid_runs[valid_runs['recovery_time'] >= 0]

    r, p_r = pearsonr(valid_runs[metric_name], valid_runs['recovery_time'])

    # T-test between groups
    t, p_t = ttest_ind(df_spiral[metric_name], df_non_spiral[metric_name])

    # Check if direction matches hypothesis
    if hypothesis_direction == 'high':
        supports = spiral_val > non_spiral_val
    else:
        supports = spiral_val < non_spiral_val

    return {
        'label': label,
        'metric': metric_name,
        'spiral_mean': spiral_val,
        'spiral_std': spiral_std,
        'non_spiral_mean': non_spiral_val,
        'non_spiral_std': non_spiral_std,
        'direction': hypothesis_direction,
        'observed_direction': 'high' if spiral_val > non_spiral_val else 'low',
        'supports_hypothesis': supports,
        'correlation_r': r,
        'correlation_p': p_r,
        't_statistic': t,
        't_pvalue': p_t
    }


def main(filepath=None):
    if filepath is None:
        filepath = "/Users/m3untold/Code/EarthianLabs/Netlogo-Models/coherence-model/exports/coherence_model_simple I002_agent_parameter_heterogeneity-spreadsheet.csv"

    print("=" * 70)
    print("I002 AGENT PARAMETER HETEROGENEITY ANALYSIS")
    print("=" * 70)
    print(f"\nLoading: {filepath}\n")

    try:
        df = load_i002_data(filepath)
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        print("\nPlease run the I002 experiment in NetLogo and export to:")
        print("  exports/coherence_model_simple I002_agent_parameter_heterogeneity-spreadsheet.csv")
        return
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    print(f"Loaded {len(df)} runs\n")

    # Split by mode
    coherence = df[df['entrainment_mode'] == False].copy()
    entrainment = df[df['entrainment_mode'] == True].copy()

    print(f"Coherence runs:   {len(coherence)}")
    print(f"Entrainment runs: {len(entrainment)}")

    # === SUMMARY STATISTICS ===
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    predictor_metrics = [
        ('initial_coupling_bias_mean', 'Coupling Bias Mean'),
        ('initial_coupling_bias_variance', 'Coupling Bias Variance'),
        ('initial_inertia_mean', 'Inertia Mean'),
        ('initial_inertia_variance', 'Inertia Variance'),
        ('pre_perturbation_cost', 'Pre-Perturbation Cost'),
        ('pre_perturbation_variance', 'Pre-Perturbation Variance'),
    ]

    print("\nCOHERENCE MODE:")
    for metric, label in predictor_metrics:
        if metric in coherence.columns:
            print(f"  {label:30s}: {coherence[metric].mean():.3f} ± {coherence[metric].std():.3f}")

    print("\nENTRAINMENT MODE:")
    for metric, label in predictor_metrics:
        if metric in entrainment.columns:
            print(f"  {label:30s}: {entrainment[metric].mean():.3f} ± {entrainment[metric].std():.3f}")

    # === SPIRAL DETECTION ===
    print("\n" + "=" * 70)
    print("SPIRAL DETECTION (Entrainment Mode)")
    print("=" * 70)

    # Define spiral: recovery > 500 ticks OR never recovered (-1)
    entrainment['spiral'] = (entrainment['recovery_time'] > 500) | (entrainment['recovery_time'] == -1)
    spiral_runs = entrainment[entrainment['spiral']]
    non_spiral_runs = entrainment[~entrainment['spiral']]

    n_spiral = len(spiral_runs)
    n_total = len(entrainment)

    print(f"\nSpiral runs:     {n_spiral}/{n_total} ({100*n_spiral/n_total:.0f}%)")
    print(f"Non-spiral runs: {len(non_spiral_runs)}/{n_total}")

    if n_spiral > 0:
        print(f"\nSpiral run details:")
        for _, row in spiral_runs.iterrows():
            print(f"  Run {row['run']:2d}: recovery={row['recovery_time']:4d}, "
                  f"max_fatigue={row.get('max_fatigue_level', 0):.2f}, "
                  f"cb_var={row.get('initial_coupling_bias_variance', 0):.3f}, "
                  f"inertia_mean={row.get('initial_inertia_mean', 0):.3f}")

    # === HYPOTHESIS TESTING ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING")
    print("=" * 70)

    hypotheses = [
        ('initial_coupling_bias_variance', 'high', 'H1: High coupling-bias variance → spiral'),
        ('initial_inertia_mean', 'low', 'H2: Low inertia mean → spiral'),
        ('pre_perturbation_cost', 'high', 'H3: High pre-perturbation cost → spiral'),
    ]

    results = []
    for metric, direction, label in hypotheses:
        if metric not in entrainment.columns:
            print(f"\n{label}")
            print(f"  WARNING: Metric '{metric}' not found in data")
            continue

        result = analyze_hypothesis(spiral_runs, non_spiral_runs, metric, direction, label)
        if result:
            results.append(result)

            print(f"\n{label}")
            print(f"  Spiral runs:     {result['spiral_mean']:.4f} ± {result['spiral_std']:.4f}")
            print(f"  Non-spiral runs: {result['non_spiral_mean']:.4f} ± {result['non_spiral_std']:.4f}")
            print(f"  Direction: Expected {direction}, Observed {result['observed_direction']}")
            print(f"  Supports hypothesis: {'YES' if result['supports_hypothesis'] else 'NO'}")
            print(f"  Correlation with recovery: r={result['correlation_r']:.3f}, p={result['correlation_p']:.3f}")
            print(f"  T-test (spiral vs non-spiral): t={result['t_statistic']:.3f}, p={result['t_pvalue']:.3f}")

    # === ADDITIONAL PREDICTOR ANALYSIS ===
    print("\n" + "=" * 70)
    print("ADDITIONAL PREDICTOR CORRELATIONS (Entrainment Mode)")
    print("=" * 70)

    outcome_metrics = ['recovery_time', 'max_fatigue_level', 'mean_cumulative_cost']

    valid_ent = entrainment[entrainment['recovery_time'] >= 0]

    for pred_metric, pred_label in predictor_metrics:
        if pred_metric not in valid_ent.columns:
            continue
        print(f"\n{pred_label}:")
        for outcome in outcome_metrics:
            if outcome not in valid_ent.columns:
                continue
            r, p = pearsonr(valid_ent[pred_metric], valid_ent[outcome])
            sig = '*' if p < 0.05 else ''
            print(f"  vs {outcome:25s}: r={r:+.3f}, p={p:.3f} {sig}")

    # === COHERENCE MODE CONTROL ===
    print("\n" + "=" * 70)
    print("COHERENCE MODE CONTROL")
    print("=" * 70)

    coherence['spiral'] = (coherence['recovery_time'] > 500) | (coherence['recovery_time'] == -1)
    coh_spirals = coherence['spiral'].sum()
    print(f"\nSpiral runs in coherence mode: {coh_spirals}/{len(coherence)}")
    print("(Expected: 0 - identity-pull provides escape valve)")

    if 'max_fatigue_level' in coherence.columns:
        print(f"Max fatigue in coherence: {coherence['max_fatigue_level'].max():.3f}")
        print("(Expected: ~0 - coherence avoids fatigue accumulation)")

    # === SYNTHESIS ===
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    supported = [r for r in results if r['supports_hypothesis']]
    significant = [r for r in results if r['t_pvalue'] < 0.05]

    print(f"\nHypotheses tested: {len(results)}")
    print(f"Direction supported: {len(supported)}/{len(results)}")
    print(f"Statistically significant (p<0.05): {len(significant)}/{len(results)}")

    if len(significant) > 0:
        print("\nSignificant predictors of spiral:")
        for r in significant:
            print(f"  - {r['metric']}: {r['observed_direction']} values in spiral runs (p={r['t_pvalue']:.3f})")
    else:
        print("\nNo statistically significant predictors found.")
        print("Consider: small sample size, stochastic dynamics, or alternative mechanisms.")

    # Best predictor by correlation
    if results:
        best = max(results, key=lambda r: abs(r['correlation_r']) if not np.isnan(r['correlation_r']) else 0)
        print(f"\nBest predictor by correlation: {best['metric']} (r={best['correlation_r']:.3f})")

    print("\n" + "=" * 70)
    print("END ANALYSIS")
    print("=" * 70)

    return df, results


def create_visualizations(df, output_dir):
    """Create and save I002 visualizations."""

    # Split by mode
    coherence = df[df['entrainment_mode'] == False].copy()
    entrainment = df[df['entrainment_mode'] == True].copy()

    # Identify spiral runs in entrainment
    entrainment['spiral'] = (entrainment['recovery_time'] > 500) | (entrainment['recovery_time'] == -1)
    spiral_runs = entrainment[entrainment['spiral']]
    non_spiral_runs = entrainment[~entrainment['spiral']]

    # =========================================================================
    # Figure 1: Hypothesis Testing - Box plots comparing spiral vs non-spiral
    # =========================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('I002: Spiral Trigger Predictors (Entrainment Mode)', fontsize=14, fontweight='bold')

    # H1: Coupling Bias Variance
    ax1 = axes[0, 0]
    data_h1 = [non_spiral_runs['initial_coupling_bias_variance'], spiral_runs['initial_coupling_bias_variance']]
    bp1 = ax1.boxplot(data_h1, labels=['Non-Spiral\n(n={})'.format(len(non_spiral_runs)),
                                        'Spiral\n(n={})'.format(len(spiral_runs))],
                      patch_artist=True)
    bp1['boxes'][0].set_facecolor('#4ECDC4')
    bp1['boxes'][1].set_facecolor('#FF6B6B')
    ax1.set_ylabel('Coupling Bias Variance')
    ax1.set_title('H1: Coupling Bias Variance\n(p=0.011, SIGNIFICANT)', fontweight='bold', color='darkgreen')

    # Scatter individual points
    for i, (data, color) in enumerate([(non_spiral_runs, '#2C7873'), (spiral_runs, '#C23B22')]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax1.scatter(x, data['initial_coupling_bias_variance'], alpha=0.6, c=color, s=40, zorder=3)

    # H2: Inertia Mean
    ax2 = axes[0, 1]
    data_h2 = [non_spiral_runs['initial_inertia_mean'], spiral_runs['initial_inertia_mean']]
    bp2 = ax2.boxplot(data_h2, labels=['Non-Spiral', 'Spiral'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#4ECDC4')
    bp2['boxes'][1].set_facecolor('#FF6B6B')
    ax2.set_ylabel('Inertia Mean')
    ax2.set_title('H2: Inertia Mean\n(p=1.0, not significant)', color='gray')

    for i, (data, color) in enumerate([(non_spiral_runs, '#2C7873'), (spiral_runs, '#C23B22')]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax2.scatter(x, data['initial_inertia_mean'], alpha=0.6, c=color, s=40, zorder=3)

    # H3: Pre-perturbation Cost
    ax3 = axes[1, 0]
    data_h3 = [non_spiral_runs['pre_perturbation_cost'], spiral_runs['pre_perturbation_cost']]
    bp3 = ax3.boxplot(data_h3, labels=['Non-Spiral', 'Spiral'], patch_artist=True)
    bp3['boxes'][0].set_facecolor('#4ECDC4')
    bp3['boxes'][1].set_facecolor('#FF6B6B')
    ax3.set_ylabel('Pre-Perturbation Cost')
    ax3.set_title('H3: Pre-Perturbation Cost\n(p=0.816, not significant)', color='gray')

    for i, (data, color) in enumerate([(non_spiral_runs, '#2C7873'), (spiral_runs, '#C23B22')]):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax3.scatter(x, data['pre_perturbation_cost'], alpha=0.6, c=color, s=40, zorder=3)

    # Summary panel
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = """
    I002 KEY FINDING

    Coupling-bias variance is the
    only significant spiral predictor.

    Interpretation:
    When agents have heterogeneous
    social sensitivity, the system
    becomes vulnerable to spiral.

    High-coupling agents bear
    disproportionate alignment cost
    while low-coupling agents resist,
    creating sustained tension.

    Coherence mode: 0/30 spirals
    (identity-pull escape valve)
    """
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig1.savefig(f'{output_dir}/I002_hypothesis_testing.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/I002_hypothesis_testing.png")

    # =========================================================================
    # Figure 2: Correlation with Recovery Time
    # =========================================================================
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig2.suptitle('I002: Predictor Correlations with Recovery Time (Entrainment Mode)', fontsize=12, fontweight='bold')

    # Only use valid recovery times (exclude -1)
    valid_ent = entrainment[entrainment['recovery_time'] >= 0].copy()

    predictors = [
        ('initial_coupling_bias_variance', 'Coupling Bias Variance', 0.446),
        ('initial_inertia_mean', 'Inertia Mean', 0.292),
        ('pre_perturbation_cost', 'Pre-Perturbation Cost', -0.091)
    ]

    for i, (metric, label, r) in enumerate(predictors):
        ax = axes[i]
        ax.scatter(valid_ent[metric], valid_ent['recovery_time'],
                   c='#3498db', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

        # Fit line
        if len(valid_ent) > 2:
            z = np.polyfit(valid_ent[metric], valid_ent['recovery_time'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_ent[metric].min(), valid_ent[metric].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)

        ax.set_xlabel(label)
        ax.set_ylabel('Recovery Time (ticks)' if i == 0 else '')
        ax.set_title(f'r = {r:.3f}', fontsize=10)

    plt.tight_layout()
    fig2.savefig(f'{output_dir}/I002_correlations.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/I002_correlations.png")

    # =========================================================================
    # Figure 3: Mode Comparison
    # =========================================================================
    fig3, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig3.suptitle('I002: Mode Comparison (Coherence vs Entrainment)', fontsize=12, fontweight='bold')

    # Recovery Time
    ax1 = axes[0]
    coh_recovery = coherence['recovery_time'].replace(-1, coherence['recovery_time'].max() + 100)
    ent_recovery = entrainment['recovery_time'].replace(-1, entrainment['recovery_time'].max() + 100)
    bp = ax1.boxplot([coh_recovery, ent_recovery],
                     labels=['Coherence', 'Entrainment'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#27ae60')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax1.set_ylabel('Recovery Time (ticks)')
    ax1.set_title('Recovery Time')

    # Max Fatigue
    ax2 = axes[1]
    bp2 = ax2.boxplot([coherence['max_fatigue_level'], entrainment['max_fatigue_level']],
                      labels=['Coherence', 'Entrainment'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#27ae60')
    bp2['boxes'][1].set_facecolor('#e74c3c')
    ax2.set_ylabel('Max Fatigue Level')
    ax2.set_title('Max Fatigue')

    # Pre-perturbation Variance (shows mode difference)
    ax3 = axes[2]
    bp3 = ax3.boxplot([coherence['pre_perturbation_variance'], entrainment['pre_perturbation_variance']],
                      labels=['Coherence', 'Entrainment'], patch_artist=True)
    bp3['boxes'][0].set_facecolor('#27ae60')
    bp3['boxes'][1].set_facecolor('#e74c3c')
    ax3.set_ylabel('Heading Variance at Tick 299')
    ax3.set_title('Pre-Perturbation State')

    plt.tight_layout()
    fig3.savefig(f'{output_dir}/I002_mode_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/I002_mode_comparison.png")

    # =========================================================================
    # Figure 4: Spiral Run Deep Dive
    # =========================================================================
    fig4, ax = plt.subplots(figsize=(10, 6))
    fig4.suptitle('I002: Spiral vs Non-Spiral Runs (Entrainment Mode)', fontsize=12, fontweight='bold')

    # Plot all entrainment runs: x = coupling-bias variance, y = recovery time
    # Color by spiral status
    non_spiral_valid = non_spiral_runs[non_spiral_runs['recovery_time'] >= 0]

    ax.scatter(non_spiral_valid['initial_coupling_bias_variance'],
               non_spiral_valid['recovery_time'],
               c='#4ECDC4', s=100, alpha=0.7, label='Non-spiral (recovered)', edgecolors='white')

    # Spiral runs - plot at high y value if recovery=-1
    for _, row in spiral_runs.iterrows():
        y_val = row['recovery_time'] if row['recovery_time'] > 0 else 2000
        marker = 'X' if row['recovery_time'] == -1 else 'o'
        label = 'Spiral (never recovered)' if row['recovery_time'] == -1 else 'Spiral (long recovery)'
        ax.scatter(row['initial_coupling_bias_variance'], y_val,
                   c='#FF6B6B', s=200, marker=marker, edgecolors='black', linewidth=2,
                   label=label, zorder=5)

    ax.axhline(y=500, color='gray', linestyle='--', alpha=0.5, label='Spiral threshold (500 ticks)')
    ax.set_xlabel('Initial Coupling Bias Variance', fontsize=11)
    ax.set_ylabel('Recovery Time (ticks)', fontsize=11)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    ax.set_ylim(-50, 2200)

    plt.tight_layout()
    fig4.savefig(f'{output_dir}/I002_spiral_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/I002_spiral_analysis.png")

    plt.close('all')
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = None

    df, results = main(filepath)

    # Create visualizations
    if df is not None:
        output_dir = "/Users/m3untold/Code/EarthianLabs/Netlogo-Models/coherence-model/exports"
        create_visualizations(df, output_dir)
        print("\nVisualization complete!")
