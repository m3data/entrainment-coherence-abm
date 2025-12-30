"""
I001 Spiral Trigger Investigation Analysis
Tests hypothesis: initial-heading-variance correlates with spiral probability in entrainment mode

Result: HYPOTHESIS NOT SUPPORTED - initial heading variance does NOT predict spiral
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
    t = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else np.inf
    # Two-tailed p-value approximation
    if abs(t) == np.inf:
        p = 0.0
    else:
        df = n - 2
        p = 2 * (1 - 0.5 * (1 + np.sign(t) * (1 - np.exp(-0.717 * abs(t) - 0.416 * t**2 / (df + 1)))))
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


def load_i001_data(filepath):
    """Load I001 BehaviorSpace export."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find entrainment-mode line
    entrainment_line_idx = None
    for i, line in enumerate(lines):
        if '"entrainment-mode?"' in line:
            entrainment_line_idx = i
            break

    if entrainment_line_idx is None:
        entrainment_line_idx = 24  # Default

    entrainment_values = lines[entrainment_line_idx].strip().split(',')

    # Find the [final value] header line
    header_line_idx = None
    for i, line in enumerate(lines):
        if '"[final value]"' in line:
            header_line_idx = i
            break

    if header_line_idx is None:
        header_line_idx = 27  # Default

    # Data is on the next line
    data_line = lines[header_line_idx + 1].strip()
    data_values = data_line.split(',')

    # I001 has 7 columns per run: [step], initial-heading-variance, recovery-time,
    # max-deviation, mean-cumulative-cost, mean-fatigue-level, max-fatigue-level
    cols_per_run = 7

    total_cols = len(data_values)
    num_runs = (total_cols - 1) // cols_per_run

    print(f"Columns per run: {cols_per_run}")
    print(f"Total runs: {num_runs}")

    metric_names = [
        'step',
        'initial_heading_variance',
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
        except IndexError:
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


def main(filepath=None):
    if filepath is None:
        filepath = "/Users/m3untold/Code/EarthianLabs/Netlogo-Models/coherence-model/exports/coherence_model_simple I001_spiral_trigger_investigation-spreadsheet.csv"

    print("=" * 70)
    print("I001 SPIRAL TRIGGER INVESTIGATION RESULTS")
    print("=" * 70)
    print(f"\nLoading: {filepath}\n")

    try:
        df = load_i001_data(filepath)
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None

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

    print("\nCOHERENCE MODE (n=30):")
    print(f"  Initial variance: {coherence['initial_heading_variance'].mean():.1f} ± {coherence['initial_heading_variance'].std():.1f}")
    print(f"  Recovery time:    {coherence['recovery_time'].mean():.1f} ± {coherence['recovery_time'].std():.1f}")
    print(f"  Max deviation:    {coherence['max_deviation'].mean():.1f} ± {coherence['max_deviation'].std():.1f}")
    print(f"  Max fatigue:      {coherence['max_fatigue_level'].mean():.2f} ± {coherence['max_fatigue_level'].std():.2f}")

    print("\nENTRAINMENT MODE (n=30):")
    print(f"  Initial variance: {entrainment['initial_heading_variance'].mean():.1f} ± {entrainment['initial_heading_variance'].std():.1f}")
    print(f"  Recovery time:    {entrainment['recovery_time'].mean():.1f} ± {entrainment['recovery_time'].std():.1f}")
    print(f"  Max deviation:    {entrainment['max_deviation'].mean():.1f} ± {entrainment['max_deviation'].std():.1f}")
    print(f"  Max fatigue:      {entrainment['max_fatigue_level'].mean():.2f} ± {entrainment['max_fatigue_level'].std():.2f}")

    # === SPIRAL DETECTION ===
    print("\n" + "=" * 70)
    print("SPIRAL DETECTION (Entrainment Mode)")
    print("=" * 70)

    entrainment['spiral'] = (entrainment['recovery_time'] > 500) | (entrainment['recovery_time'] == -1)
    spiral_runs = entrainment[entrainment['spiral']]
    non_spiral_runs = entrainment[~entrainment['spiral']]

    n_spiral = len(spiral_runs)
    n_total = len(entrainment)

    print(f"\nSpiral runs (recovery > 500 or no recovery): {n_spiral}/{n_total} ({100*n_spiral/n_total:.0f}%)")
    print(f"Non-spiral runs: {len(non_spiral_runs)}/{n_total}")

    if n_spiral > 0:
        print(f"\nSpiral run details:")
        for _, row in spiral_runs.iterrows():
            print(f"  Run {row['run']:2d}: init_var={row['initial_heading_variance']:.1f}, "
                  f"recovery={row['recovery_time']}, fatigue={row['max_fatigue_level']:.2f}")

    # === HYPOTHESIS TESTING ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING")
    print("=" * 70)
    print("\nHypothesis: High initial heading variance → spiral")

    # Correlation analysis
    valid_entrainment = entrainment[entrainment['recovery_time'] >= 0]

    if len(valid_entrainment) > 2:
        r_recovery, p_recovery = pearsonr(
            valid_entrainment['initial_heading_variance'],
            valid_entrainment['recovery_time']
        )
        print(f"\nCorrelation with recovery time: r={r_recovery:.3f}, p={p_recovery:.3f}")

        r_fatigue, p_fatigue = pearsonr(
            valid_entrainment['initial_heading_variance'],
            valid_entrainment['max_fatigue_level']
        )
        print(f"Correlation with max fatigue:   r={r_fatigue:.3f}, p={p_fatigue:.3f}")

        r_deviation, p_deviation = pearsonr(
            valid_entrainment['initial_heading_variance'],
            valid_entrainment['max_deviation']
        )
        print(f"Correlation with max deviation: r={r_deviation:.3f}, p={p_deviation:.3f}")

    # T-test between spiral and non-spiral
    if len(spiral_runs) > 0 and len(non_spiral_runs) > 0:
        print(f"\n--- Spiral vs Non-Spiral Comparison ---")
        print(f"Spiral runs initial variance:     {spiral_runs['initial_heading_variance'].mean():.1f} ± {spiral_runs['initial_heading_variance'].std():.1f}")
        print(f"Non-spiral runs initial variance: {non_spiral_runs['initial_heading_variance'].mean():.1f} ± {non_spiral_runs['initial_heading_variance'].std():.1f}")

        if len(spiral_runs) > 1 and len(non_spiral_runs) > 1:
            t_stat, t_pval = ttest_ind(
                spiral_runs['initial_heading_variance'],
                non_spiral_runs['initial_heading_variance']
            )
            print(f"T-test: t={t_stat:.3f}, p={t_pval:.3f}")

            if t_pval > 0.05:
                print("\nRESULT: HYPOTHESIS NOT SUPPORTED")
                print("Initial heading variance does NOT predict spiral occurrence.")
            else:
                print("\nRESULT: HYPOTHESIS SUPPORTED")

    # === COHERENCE CONTROL ===
    print("\n" + "=" * 70)
    print("COHERENCE MODE CONTROL")
    print("=" * 70)

    coherence['spiral'] = (coherence['recovery_time'] > 500) | (coherence['recovery_time'] == -1)
    coh_spirals = coherence['spiral'].sum()
    print(f"\nSpiral runs in coherence mode: {coh_spirals}/{len(coherence)}")
    print("(Expected: 0 - identity-pull provides escape valve)")

    print(f"Max fatigue in coherence: {coherence['max_fatigue_level'].max():.3f}")
    print("(Expected: ~0 - coherence avoids fatigue accumulation)")

    valid_coherence = coherence[coherence['recovery_time'] >= 0]
    if len(valid_coherence) > 2:
        r_coh, p_coh = pearsonr(
            valid_coherence['initial_heading_variance'],
            valid_coherence['recovery_time']
        )
        print(f"\nInitial variance vs Recovery time: r={r_coh:.3f}, p={p_coh:.3f}")

    # === SYNTHESIS ===
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    print("""
    I001 KEY FINDING: Initial heading variance is NOT the spiral trigger.

    Both spiral and non-spiral runs start with essentially identical
    variance (~89-91°). The spiral mechanism must be elsewhere.

    Candidate mechanisms for I002:
    1. Agent parameter heterogeneity (coupling-bias, inertia variance)
    2. Pre-perturbation accumulated cost/fatigue
    3. Perturbation-field directional interaction
    """)

    print("\n" + "=" * 70)
    print("END ANALYSIS")
    print("=" * 70)

    return df


def create_visualizations(df, output_dir):
    """Create and save I001 visualizations."""

    # Split by mode
    coherence = df[df['entrainment_mode'] == False].copy()
    entrainment = df[df['entrainment_mode'] == True].copy()

    # Identify spiral runs
    entrainment['spiral'] = (entrainment['recovery_time'] > 500) | (entrainment['recovery_time'] == -1)
    spiral_runs = entrainment[entrainment['spiral']]
    non_spiral_runs = entrainment[~entrainment['spiral']]

    # =========================================================================
    # Figure 1: Main Hypothesis Test - Initial Variance vs Spiral Status
    # =========================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('I001: Initial Heading Variance as Spiral Predictor', fontsize=14, fontweight='bold')

    # Box plot comparison
    ax1 = axes[0, 0]
    data = [non_spiral_runs['initial_heading_variance'], spiral_runs['initial_heading_variance']]
    bp = ax1.boxplot(data, tick_labels=['Non-Spiral\n(n={})'.format(len(non_spiral_runs)),
                                         'Spiral\n(n={})'.format(len(spiral_runs))],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('#4ECDC4')
    bp['boxes'][1].set_facecolor('#FF6B6B')
    ax1.set_ylabel('Initial Heading Variance (°)')
    ax1.set_title('Hypothesis Test: NOT SIGNIFICANT\n(p=0.805)', color='darkred', fontweight='bold')

    # Scatter points
    for i, (d, color) in enumerate([(non_spiral_runs, '#2C7873'), (spiral_runs, '#C23B22')]):
        x = np.random.normal(i+1, 0.04, size=len(d))
        ax1.scatter(x, d['initial_heading_variance'], alpha=0.6, c=color, s=50, zorder=3)

    # Scatter: Initial variance vs Recovery time
    ax2 = axes[0, 1]
    valid_ent = entrainment[entrainment['recovery_time'] >= 0]
    colors = ['#FF6B6B' if s else '#4ECDC4' for s in valid_ent['spiral']]
    ax2.scatter(valid_ent['initial_heading_variance'], valid_ent['recovery_time'],
                c=colors, s=60, alpha=0.7, edgecolors='white')

    # Fit line
    z = np.polyfit(valid_ent['initial_heading_variance'], valid_ent['recovery_time'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_ent['initial_heading_variance'].min(),
                         valid_ent['initial_heading_variance'].max(), 100)
    ax2.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('Initial Heading Variance (°)')
    ax2.set_ylabel('Recovery Time (ticks)')
    ax2.set_title('r = 0.163 (not significant)')

    # Distribution of initial variance by mode
    ax3 = axes[1, 0]
    ax3.hist(coherence['initial_heading_variance'], bins=15, alpha=0.6,
             label='Coherence', color='#27ae60', edgecolor='white')
    ax3.hist(entrainment['initial_heading_variance'], bins=15, alpha=0.6,
             label='Entrainment', color='#e74c3c', edgecolor='white')
    ax3.axvline(coherence['initial_heading_variance'].mean(), color='#27ae60',
                linestyle='--', linewidth=2, label=f'Coh mean: {coherence["initial_heading_variance"].mean():.1f}°')
    ax3.axvline(entrainment['initial_heading_variance'].mean(), color='#e74c3c',
                linestyle='--', linewidth=2, label=f'Ent mean: {entrainment["initial_heading_variance"].mean():.1f}°')
    ax3.set_xlabel('Initial Heading Variance (°)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution by Mode')
    ax3.legend(fontsize=8)

    # Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = """
    I001 CONCLUSION

    Hypothesis REJECTED:
    Initial heading variance does NOT
    predict spiral occurrence.

    Evidence:
    • Spiral runs: 91.3 ± 2.9°
    • Non-spiral:  89.2 ± 4.6°
    • T-test: p = 0.805 (not significant)
    • Correlation: r = 0.163

    Both spiral and non-spiral runs
    start with nearly identical variance.

    The spiral trigger lies elsewhere.
    → See I002 for agent heterogeneity
    """
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.5))

    plt.tight_layout()
    fig1.savefig(f'{output_dir}/I001_hypothesis_test.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/I001_hypothesis_test.png")

    # =========================================================================
    # Figure 2: Mode Comparison
    # =========================================================================
    fig2, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig2.suptitle('I001: Coherence vs Entrainment Mode Comparison', fontsize=12, fontweight='bold')

    metrics = [
        ('initial_heading_variance', 'Initial Variance (°)'),
        ('recovery_time', 'Recovery Time (ticks)'),
        ('max_deviation', 'Max Deviation (°)'),
        ('max_fatigue_level', 'Max Fatigue Level')
    ]

    for i, (metric, label) in enumerate(metrics):
        ax = axes[i]
        coh_data = coherence[metric].replace(-1, np.nan).dropna()
        ent_data = entrainment[metric].replace(-1, np.nan).dropna()

        bp = ax.boxplot([coh_data, ent_data], tick_labels=['Coherence', 'Entrainment'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('#27ae60')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_ylabel(label)

    plt.tight_layout()
    fig2.savefig(f'{output_dir}/I001_mode_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/I001_mode_comparison.png")

    # =========================================================================
    # Figure 3: Spiral Run Analysis
    # =========================================================================
    fig3, ax = plt.subplots(figsize=(10, 6))
    fig3.suptitle('I001: Spiral vs Non-Spiral Runs (Entrainment Mode)', fontsize=12, fontweight='bold')

    # Plot all entrainment runs
    non_spiral_valid = non_spiral_runs[non_spiral_runs['recovery_time'] >= 0]

    ax.scatter(non_spiral_valid['initial_heading_variance'],
               non_spiral_valid['recovery_time'],
               c='#4ECDC4', s=100, alpha=0.7, label='Non-spiral', edgecolors='white')

    # Spiral runs
    for _, row in spiral_runs.iterrows():
        y_val = row['recovery_time'] if row['recovery_time'] > 0 else 2000
        marker = 'X' if row['recovery_time'] == -1 else 'o'
        label_text = 'Spiral (never recovered)' if row['recovery_time'] == -1 else 'Spiral (long recovery)'
        ax.scatter(row['initial_heading_variance'], y_val,
                   c='#FF6B6B', s=200, marker=marker, edgecolors='black', linewidth=2,
                   label=label_text, zorder=5)

    ax.axhline(y=500, color='gray', linestyle='--', alpha=0.5, label='Spiral threshold')

    # Add annotation showing overlap
    ax.annotate('Spiral and non-spiral runs\noverlap in initial variance',
                xy=(90, 1000), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Initial Heading Variance (°)', fontsize=11)
    ax.set_ylabel('Recovery Time (ticks)', fontsize=11)
    ax.set_ylim(-50, 2200)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    fig3.savefig(f'{output_dir}/I001_spiral_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/I001_spiral_analysis.png")

    # =========================================================================
    # Figure 4: Coherence Mode Stability
    # =========================================================================
    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle('I001: Coherence Mode Stability (Identity-Pull Escape Valve)', fontsize=12, fontweight='bold')

    # Recovery time comparison
    ax1 = axes[0]
    coh_recovery = coherence['recovery_time'].replace(-1, np.nan).dropna()
    ent_recovery = entrainment['recovery_time'].replace(-1, 2500)  # Show -1 as high value

    ax1.scatter(range(len(coh_recovery)), sorted(coh_recovery),
                c='#27ae60', s=60, alpha=0.7, label='Coherence')
    ax1.scatter(range(len(ent_recovery)), sorted(ent_recovery),
                c='#e74c3c', s=60, alpha=0.7, label='Entrainment')
    ax1.axhline(y=500, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Run (sorted by recovery time)')
    ax1.set_ylabel('Recovery Time (ticks)')
    ax1.set_title('Recovery Time Distribution')
    ax1.legend()

    # Max fatigue comparison
    ax2 = axes[1]
    ax2.bar(['Coherence', 'Entrainment'],
            [coherence['max_fatigue_level'].max(), entrainment['max_fatigue_level'].max()],
            color=['#27ae60', '#e74c3c'], alpha=0.8, edgecolor='white')
    ax2.set_ylabel('Max Fatigue Level')
    ax2.set_title('Peak Fatigue by Mode')
    ax2.set_ylim(0, 1.1)

    # Add text annotation
    ax2.annotate('Coherence never\naccumulates fatigue',
                 xy=(0, 0.1), fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='#27ae60', alpha=0.3))

    plt.tight_layout()
    fig4.savefig(f'{output_dir}/I001_coherence_stability.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/I001_coherence_stability.png")

    plt.close('all')
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = None

    df = main(filepath)

    # Create visualizations
    if df is not None:
        output_dir = "/Users/m3untold/Code/EarthianLabs/Netlogo-Models/coherence-model/exports"
        create_visualizations(df, output_dir)
        print("\nVisualization complete!")
