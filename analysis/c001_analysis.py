"""
C001 Selective Escape Valve Experiment Analysis
================================================

Tests whether relief pathway (identity-pull) availability is the causal
mechanism preventing cascade failure in the coherence model.

Experiment Conditions:
- Condition A: entrainment-mode? = false, selective-identity-pull = "load-bearers-only"
- Condition B: entrainment-mode? = false, selective-identity-pull = "non-load-bearers-only"
- Condition C: entrainment-mode? = false, selective-identity-pull = "all"
- Condition D: entrainment-mode? = true

Pre-registered Predictions:
1. Condition A (load-bearers-only) → near-zero spiral rate
2. Condition B (non-load-bearers-only) → spiral rate ≈ Condition D
3. Condition C → zero spirals (baseline coherence)
4. Condition D → ~10% spiral rate (baseline entrainment)

Author: Data Scientist (Claude Code ecosystem)
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('ggplot')

# File paths
DATA_PATH = Path("/Users/m3untold/Code/EarthianLabs/Netlogo-Models/entrainment-coherence-abm/exports/coherence_model_simple C001_selective_escape_valve-spreadsheet.csv")
OUTPUT_DIR = Path("/Users/m3untold/Code/EarthianLabs/Netlogo-Models/entrainment-coherence-abm/exports")

print("="*80)
print("C001 SELECTIVE ESCAPE VALVE EXPERIMENT ANALYSIS")
print("="*80)
print()

# Load data - BehaviorSpace uses wide format where runs are columns
print("Loading BehaviorSpace data...")
df_wide = pd.read_csv(DATA_PATH, skiprows=6)
print(f"Loaded wide format: {df_wide.shape[0]} parameters × {df_wide.shape[1]} columns")
print()

# Transform to long format
# The first column contains parameter names
param_names = df_wide.iloc[:, 0].values
print(f"Found {len(param_names)} parameters/metrics")
print()

# Each remaining column is a run
run_data = []
for col in df_wide.columns[1:]:
    run_dict = {}
    run_dict['run_id'] = col
    for i, param in enumerate(param_names):
        run_dict[param] = df_wide.iloc[i][col]
    run_data.append(run_dict)

df = pd.DataFrame(run_data)
print(f"Transformed to long format: {len(df)} runs × {len(df.columns)} parameters")
print()

# Display first few runs
print("First 5 runs:")
print(df.head())
print()

# Check key columns
print("Available columns:")
print(df.columns.tolist())
print()

# Data types
print("Data types:")
print(df.dtypes)
print()

# Convert numeric columns
numeric_cols = ['population', 'coupling-strength', 'noise-level', 'perturb-duration',
                'recovery-tolerance', 'recovery-time', 'max-fatigue-level',
                'at-risk-max-fatigue', 'at-risk-mean-fatigue']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Converted numeric columns")
print()

# Check for condition columns
if 'entrainment-mode?' in df.columns:
    print("entrainment-mode? values:")
    print(df['entrainment-mode?'].value_counts())
    print()

if 'selective-identity-pull' in df.columns:
    print("selective-identity-pull values:")
    print(df['selective-identity-pull'].value_counts())
    print()

# Basic statistics for key metrics
print("Basic statistics for key metrics:")
key_metrics = ['recovery-time', 'max-fatigue-level', 'at-risk-max-fatigue', 'at-risk-mean-fatigue']
for metric in key_metrics:
    if metric in df.columns:
        print(f"\n{metric}:")
        print(df[metric].describe())
print()

# Save cleaned data
cleaned_path = OUTPUT_DIR / "c001_cleaned_data.csv"
df.to_csv(cleaned_path, index=False)
print(f"Saved cleaned data to: {cleaned_path}")
print()
