"""
C001 Selective Escape Valve Experiment Analysis
================================================

Tests whether relief pathway (identity-pull) availability is the causal
mechanism preventing cascade failure in the coherence model.

This analysis tests the pre-registered predictions about which agents need
the identity-pull escape valve to prevent cascade failure.

Author: Data Scientist (Claude Code ecosystem)
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# File paths
DATA_PATH = Path("/Users/m3untold/Code/EarthianLabs/Netlogo-Models/entrainment-coherence-abm/exports/coherence_model_simple C001_selective_escape_valve-spreadsheet.csv")
OUTPUT_DIR = Path("/Users/m3untold/Code/EarthianLabs/Netlogo-Models/entrainment-coherence-abm/exports")
TIMESTAMP = "2026-01-11"

print("="*80)
print("C001 SELECTIVE ESCAPE VALVE EXPERIMENT ANALYSIS")
print("="*80)
print()

# ============================================================================
# PART 1: DATA LOADING AND PARSING
# ============================================================================

print("PART 1: Data Loading and Parsing")
print("-" * 80)

# Load wide format
df_wide = pd.read_csv(DATA_PATH, skiprows=6)
param_names = df_wide.iloc[:, 0].values

# Parse runs - each base run (like "1") has parameters, sub-runs (like "1.1") have metrics
runs = []
current_run = None

for col in df_wide.columns[1:]:
    # Check if this is a base run (no decimal) or a sub-run (has decimal)
    if '.' not in str(col):
        # This is a new base run - store its parameters
        current_run = {'run_id': col}
        for i, param in enumerate(param_names):
            current_run[param] = df_wide.iloc[i][col]
    else:
        # This is a metric sub-run - combine with base run parameters
        if current_run is None:
            continue

        run_with_metric = current_run.copy()
        run_with_metric['sub_run_id'] = col

        # Extract metric name and value from this sub-run
        for i, param in enumerate(param_names):
            value = df_wide.iloc[i][col]
            if pd.notna(value):
                # Look for the [final value] row which contains metric name
                if param == '[final value]':
                    run_with_metric['metric_name'] = value
                # The previous row contains the metric value
                elif i > 0 and param_names[i-1] == '[final value]':
                    run_with_metric['metric_value'] = value

        if 'metric_name' in run_with_metric and 'metric_value' in run_with_metric:
            runs.append(run_with_metric)

df_long = pd.DataFrame(runs)

print(f"Parsed {len(df_long)} metric records from {df_wide.shape[1]-1} columns")
print()

# Pivot so each run has all metrics as columns
df_pivot = df_long.pivot_table(
    index='run_id',
    columns='metric_name',
    values='metric_value',
    aggfunc='first'
)

# Add back the parameters from the first occurrence of each run_id
param_cols = ['population', 'coupling-strength', 'noise-level', 'perturb-duration',
              'recovery-tolerance', 'entrainment-mode?', 'selective-identity-pull',
              'identity-pull-weight', 'social-pull-weight']

for param in param_cols:
    if param in df_long.columns:
        df_pivot[param] = df_long.groupby('run_id')[param].first()

# Reset index to make run_id a column
df = df_pivot.reset_index()

print("Pivoted to wide format with metrics as columns:")
print(df.head())
print()
print(f"Shape: {df.shape}")
print()

# Convert numeric columns
numeric_metrics = ['recovery-time', 'max-fatigue-level', 'at-risk-max-fatigue',
                   'at-risk-mean-fatigue', 'population']
for col in numeric_metrics:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Converted numeric columns")
print()

# ============================================================================
# PART 2: CONDITION DEFINITION
# ============================================================================

print("PART 2: Condition Definition")
print("-" * 80)

# Define conditions
def classify_condition(row):
    """Classify each run into experimental condition."""
    entrainment = row.get('entrainment-mode?', '')
    identity_pull = row.get('selective-identity-pull', '')

    if entrainment == 'true':
        return 'D (entrainment baseline)'
    elif entrainment == 'false':
        if identity_pull == 'load-bearers-only':
            return 'A (load-bearers protected)'
        elif identity_pull == 'non-load-bearers-only':
            return 'B (load-bearers exposed)'
        elif identity_pull == 'all':
            return 'C (coherence baseline)'

    return 'Unknown'

df['condition'] = df.apply(classify_condition, axis=1)

print("Condition distribution:")
print(df['condition'].value_counts().sort_index())
print()

# Display sample from each condition
print("Sample from each condition:")
for condition in sorted(df['condition'].unique()):
    sample = df[df['condition'] == condition].iloc[0]
    print(f"\n{condition}:")
    print(f"  entrainment-mode? = {sample.get('entrainment-mode?')}")
    print(f"  selective-identity-pull = {sample.get('selective-identity-pull')}")
print()

# ============================================================================
# PART 3: SPIRAL CLASSIFICATION
# ============================================================================

print("PART 3: Spiral Classification")
print("-" * 80)

# Define spiral criteria (based on I001/I002 findings)
RECOVERY_THRESHOLD = 1000  # Spirals show very long or infinite recovery
FATIGUE_THRESHOLD = 0.5    # Spirals show high fatigue

def classify_spiral(row):
    """Classify whether a run resulted in cascade spiral."""
    recovery = row.get('recovery-time', 0)
    fatigue = row.get('max-fatigue-level', 0)

    # Handle missing values
    if pd.isna(recovery):
        recovery = 0
    if pd.isna(fatigue):
        fatigue = 0

    # Spiral if recovery time is very long OR fatigue is very high
    if recovery > RECOVERY_THRESHOLD or fatigue > FATIGUE_THRESHOLD:
        return True
    return False

df['is_spiral'] = df.apply(classify_spiral, axis=1)

print(f"Spiral classification using:")
print(f"  - recovery-time > {RECOVERY_THRESHOLD} ticks, OR")
print(f"  - max-fatigue-level > {FATIGUE_THRESHOLD}")
print()

print("Spiral distribution:")
print(df['is_spiral'].value_counts())
print()

# ============================================================================
# PART 4: DESCRIPTIVE STATISTICS BY CONDITION
# ============================================================================

print("PART 4: Descriptive Statistics by Condition")
print("-" * 80)

# Calculate spiral rate and key metrics per condition
summary_stats = []

for condition in sorted(df['condition'].unique()):
    cond_data = df[df['condition'] == condition]

    n = len(cond_data)
    spirals = cond_data['is_spiral'].sum()
    spiral_rate = spirals / n if n > 0 else 0

    recovery = cond_data['recovery-time']
    fatigue = cond_data['max-fatigue-level']

    summary_stats.append({
        'Condition': condition,
        'N': n,
        'Spirals': spirals,
        'Spiral Rate': f"{spiral_rate:.1%}",
        'Recovery Mean ± SD': f"{recovery.mean():.0f} ± {recovery.std():.0f}",
        'Recovery Median': f"{recovery.median():.0f}",
        'Max Fatigue Mean ± SD': f"{fatigue.mean():.3f} ± {fatigue.std():.3f}",
        'Max Fatigue Median': f"{fatigue.median():.3f}"
    })

summary_df = pd.DataFrame(summary_stats)
print(summary_df.to_string(index=False))
print()

# Save summary table
summary_path = OUTPUT_DIR / f"c001_summary_{TIMESTAMP}.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary table to: {summary_path}")
print()

# ============================================================================
# PART 5: HYPOTHESIS TESTING
# ============================================================================

print("PART 5: Hypothesis Testing")
print("-" * 80)
print()

print("Pre-registered Predictions:")
print("  P1: Condition A (load-bearers-only) → near-zero spiral rate")
print("  P2: Condition B (non-load-bearers-only) → spiral rate ≈ Condition D")
print("  P3: Condition C (all) → zero spirals (baseline coherence)")
print("  P4: Condition D (entrainment) → ~10% spiral rate")
print()

# Extract spiral rates
conditions = {
    'A': 'A (load-bearers protected)',
    'B': 'B (load-bearers exposed)',
    'C': 'C (coherence baseline)',
    'D': 'D (entrainment baseline)'
}

rates = {}
for key, label in conditions.items():
    cond_data = df[df['condition'] == label]
    if len(cond_data) > 0:
        rates[key] = cond_data['is_spiral'].sum() / len(cond_data)
    else:
        rates[key] = np.nan

print("OBSERVED SPIRAL RATES:")
for key in ['A', 'B', 'C', 'D']:
    if not np.isnan(rates[key]):
        print(f"  Condition {key}: {rates[key]:.1%} ({df[df['condition']==conditions[key]]['is_spiral'].sum()}/{len(df[df['condition']==conditions[key]])})")
print()

# Test each prediction
print("HYPOTHESIS TEST RESULTS:")
print()

# P1: Condition A should have near-zero spiral rate
print(f"P1: Condition A → near-zero spiral rate")
print(f"    Observed: {rates['A']:.1%}")
if rates['A'] <= 0.05:  # Allow up to 5% as "near-zero"
    print(f"    Result: ✓ SUPPORTED (≤5%)")
else:
    print(f"    Result: ✗ FALSIFIED (>{5}%)")
print()

# P2: Condition B should be similar to Condition D
print(f"P2: Condition B ≈ Condition D")
print(f"    Observed B: {rates['B']:.1%}")
print(f"    Observed D: {rates['D']:.1%}")
print(f"    Ratio B/D: {rates['B']/rates['D']:.2f}x" if rates['D'] > 0 else "    D has zero rate")
if abs(rates['B'] - rates['D']) < 0.10:  # Within 10 percentage points
    print(f"    Result: ✓ SUPPORTED (within 10 pp)")
else:
    print(f"    Result: ✗ NOT SUPPORTED (difference = {abs(rates['B']-rates['D']):.1%})")
print()

# P3: Condition C should have zero spirals
print(f"P3: Condition C → zero spirals")
print(f"    Observed: {rates['C']:.1%}")
if rates['C'] == 0:
    print(f"    Result: ✓ SUPPORTED (exactly 0%)")
else:
    print(f"    Result: ✗ FALSIFIED ({rates['C']:.1%} > 0%)")
print()

# P4: Condition D should have ~10% spiral rate (based on I001 findings)
print(f"P4: Condition D → ~10% spiral rate")
print(f"    Observed: {rates['D']:.1%}")
print(f"    Expected: ~10% (from I001 findings)")
if 0.05 <= rates['D'] <= 0.15:  # 5-15% range
    print(f"    Result: ✓ SUPPORTED (5-15% range)")
else:
    print(f"    Result: ? INCONCLUSIVE (outside expected range)")
print()

# ============================================================================
# PART 6: EFFECT SIZE ANALYSIS
# ============================================================================

print("PART 6: Effect Size Analysis")
print("-" * 80)
print()

# Compare recovery times between conditions
def cohen_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

# Key comparison: A vs B (both coherence mode, different identity-pull)
cond_a = df[df['condition'] == conditions['A']]['recovery-time'].dropna()
cond_b = df[df['condition'] == conditions['B']]['recovery-time'].dropna()
cond_d = df[df['condition'] == conditions['D']]['recovery-time'].dropna()

if len(cond_a) > 0 and len(cond_b) > 0:
    print("Comparison: A (load-bearers protected) vs B (load-bearers exposed)")
    print(f"  A recovery: {cond_a.mean():.1f} ± {cond_a.std():.1f}")
    print(f"  B recovery: {cond_b.mean():.1f} ± {cond_b.std():.1f}")

    d = cohen_d(cond_a, cond_b)
    if not np.isnan(d):
        print(f"  Cohen's d: {d:.3f}")
        if abs(d) < 0.2:
            print(f"  Effect size: NEGLIGIBLE")
        elif abs(d) < 0.5:
            print(f"  Effect size: SMALL")
        elif abs(d) < 0.8:
            print(f"  Effect size: MEDIUM")
        else:
            print(f"  Effect size: LARGE")
    print()

if len(cond_b) > 0 and len(cond_d) > 0:
    print("Comparison: B (load-bearers exposed) vs D (entrainment)")
    print(f"  B recovery: {cond_b.mean():.1f} ± {cond_b.std():.1f}")
    print(f"  D recovery: {cond_d.mean():.1f} ± {cond_d.std():.1f}")

    d = cohen_d(cond_b, cond_d)
    if not np.isnan(d):
        print(f"  Cohen's d: {d:.3f}")
        if abs(d) < 0.2:
            print(f"  Effect size: NEGLIGIBLE")
        elif abs(d) < 0.5:
            print(f"  Effect size: SMALL")
        elif abs(d) < 0.8:
            print(f"  Effect size: MEDIUM")
        else:
            print(f"  Effect size: LARGE")
    print()

# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

print("PART 7: Creating Visualizations")
print("-" * 80)

# Figure 1: Spiral rates by condition
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Spiral rate bar chart
ax = axes[0, 0]
spiral_counts = []
for key in ['A', 'B', 'C', 'D']:
    cond_data = df[df['condition'] == conditions[key]]
    spiral_counts.append({
        'Condition': key,
        'Spiral Rate': cond_data['is_spiral'].sum() / len(cond_data) if len(cond_data) > 0 else 0,
        'N': len(cond_data)
    })

spiral_df = pd.DataFrame(spiral_counts)
ax.bar(spiral_df['Condition'], spiral_df['Spiral Rate'] * 100, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
ax.set_ylabel('Spiral Rate (%)', fontsize=12)
ax.set_xlabel('Condition', fontsize=12)
ax.set_title('A. Spiral Rate by Condition', fontsize=14, fontweight='bold')
ax.axhline(10, color='gray', linestyle='--', alpha=0.5, label='Expected D (~10%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Panel B: Recovery time distributions
ax = axes[0, 1]
recovery_data = []
labels = []
for key in ['A', 'B', 'C', 'D']:
    cond_data = df[df['condition'] == conditions[key]]['recovery-time'].dropna()
    if len(cond_data) > 0:
        recovery_data.append(cond_data.values)
        labels.append(key)

bp = ax.boxplot(recovery_data, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Recovery Time (ticks)', fontsize=12)
ax.set_xlabel('Condition', fontsize=12)
ax.set_title('B. Recovery Time Distribution', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

# Panel C: Max fatigue distributions
ax = axes[1, 0]
fatigue_data = []
labels = []
for key in ['A', 'B', 'C', 'D']:
    cond_data = df[df['condition'] == conditions[key]]['max-fatigue-level'].dropna()
    if len(cond_data) > 0:
        fatigue_data.append(cond_data.values)
        labels.append(key)

bp = ax.boxplot(fatigue_data, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Max Fatigue Level', fontsize=12)
ax.set_xlabel('Condition', fontsize=12)
ax.set_title('C. Max Fatigue Distribution', fontsize=14, fontweight='bold')
ax.axhline(FATIGUE_THRESHOLD, color='red', linestyle='--', alpha=0.5, label=f'Spiral threshold ({FATIGUE_THRESHOLD})')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Panel D: Summary table
ax = axes[1, 1]
ax.axis('off')

# Create table data
table_data = []
table_data.append(['Condition', 'N', 'Spiral\nRate', 'Recovery\n(median)', 'Max Fatigue\n(median)'])
for key in ['A', 'B', 'C', 'D']:
    cond_data = df[df['condition'] == conditions[key]]
    n = len(cond_data)
    spiral_rate = f"{cond_data['is_spiral'].sum() / n * 100:.0f}%"
    recovery_med = f"{cond_data['recovery-time'].median():.0f}"
    fatigue_med = f"{cond_data['max-fatigue-level'].median():.3f}"
    table_data.append([key, str(n), spiral_rate, recovery_med, fatigue_med])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                bbox=[0, 0.3, 1, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
for i, color in enumerate(colors, start=1):
    for j in range(5):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_alpha(0.3)

ax.set_title('D. Summary Statistics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
fig_path = OUTPUT_DIR / f"c001_analysis_{TIMESTAMP}.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to: {fig_path}")
print()

plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()

print("Output files saved:")
print(f"  - {summary_path}")
print(f"  - {fig_path}")
print()
