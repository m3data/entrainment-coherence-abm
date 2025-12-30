"""
S001a Scale Sensitivity Analysis
Comparing F001 baseline (150 agents, 2000 ticks) with extended runs
"""

import pandas as pd
import numpy as np

# Parse the BehaviorSpace export
def load_behaviorspace_wide(filepath):
    """Load BehaviorSpace spreadsheet format into run-wise dataframes"""

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find key rows
    for i, line in enumerate(lines):
        if '"[run number]"' in line:
            run_row = i
        if '"entrainment-mode?"' in line:
            mode_row = i
        if '"population"' in line:
            pop_row = i
        if '"[reporter]"' in line:
            header_row = i
            break

    # Parse run configurations
    run_line = lines[run_row].strip().split(',')
    mode_line = lines[mode_row].strip().split(',')
    pop_line = lines[pop_row].strip().split(',')

    # Build run metadata
    runs = []
    current_run = None
    for j, val in enumerate(run_line[1:], 1):
        if val.strip('"'):
            current_run = int(val.strip('"'))
        if current_run and j < len(mode_line) and j < len(pop_line):
            mode_val = mode_line[j].strip('"')
            pop_val = pop_line[j].strip('"')
            if mode_val and pop_val:
                runs.append({
                    'run': current_run,
                    'entrainment_mode': mode_val == 'true',
                    'population': int(pop_val),
                    'col_start': j
                })

    # Deduplicate runs
    seen = set()
    unique_runs = []
    for r in runs:
        key = (r['run'], r['entrainment_mode'], r['population'])
        if key not in seen:
            seen.add(key)
            unique_runs.append(r)

    return unique_runs, header_row, lines

def extract_summary_stats(lines, header_row):
    """Extract [final], [min], [max], [mean] rows"""
    summaries = {}
    for line in lines[header_row+1:]:
        parts = line.strip().split(',')
        if parts[0].strip('"') in ['[final]', '[min]', '[max]', '[mean]']:
            summaries[parts[0].strip('"').strip('[]')] = parts
    return summaries

# Load data
filepath = '/Users/m3untold/Code/EarthianLabs/Netlogo-Models/coherence-model/exports/coherence_model_simple S001a_scale_sensitivity_5000-spreadsheet.csv'
runs, header_row, lines = load_behaviorspace_wide(filepath)
summaries = extract_summary_stats(lines, header_row)

# Parse header to get metric positions
header = lines[header_row].strip().split(',')
metrics_per_run = ['step', 'recovery-time', 'baseline', 'max-deviation', 'heading-variance',
                   'mean-metabolic-cost', 'mean-shock-cost', 'mean-recovery-cost',
                   'mean-cumulative-cost', 'cost-variance', 'cost-gini',
                   'mean-fatigue-level', 'max-fatigue-level', 'agents-fatigued',
                   'mean-effective-inertia']

print("=" * 80)
print("S001a SCALE SENSITIVITY RESULTS (5000 ticks)")
print("=" * 80)
print()

# Build results table
results = []
for i, run in enumerate(runs):
    run_offset = 1 + i * len(metrics_per_run)  # offset into the data columns

    # Get final values
    final = summaries['final']
    max_vals = summaries['max']

    result = {
        'Run': run['run'],
        'Mode': 'Entrainment' if run['entrainment_mode'] else 'Coherence',
        'Population': run['population'],
        'Recovery Time (final)': final[run_offset + 1].strip('"'),
        'Max Deviation': max_vals[run_offset + 3].strip('"'),
        'Cumulative Cost': final[run_offset + 8].strip('"'),
        'Cost Gini (final)': final[run_offset + 10].strip('"'),
        'Max Fatigue': max_vals[run_offset + 12].strip('"'),
        'Peak Agents Fatigued': max_vals[run_offset + 13].strip('"'),
    }
    results.append(result)

# Print as table
df = pd.DataFrame(results)
print("KEY METRICS BY CONDITION")
print("-" * 80)
print(df.to_string(index=False))
print()

# Compare with F001 baseline
print("=" * 80)
print("COMPARISON WITH F001 BASELINE (150 agents, 2000 ticks)")
print("=" * 80)
print()
print("F001 findings:")
print("  Coherence + Fatigue:   Recovery 50 ticks,  Peak Fatigue 0.04")
print("  Entrainment + Fatigue: Recovery 1330 ticks, Peak Fatigue 0.65")
print()

# Extract coherence 150 and entrainment 150 for direct comparison
coh_150 = [r for r in results if r['Mode'] == 'Coherence' and r['Population'] == 150][0]
ent_150 = [r for r in results if r['Mode'] == 'Entrainment' and r['Population'] == 150][0]

print("S001a findings (150 agents, 5000 ticks):")
print(f"  Coherence + Fatigue:   Recovery {coh_150['Recovery Time (final)']} ticks, Peak Fatigue {coh_150['Max Fatigue']}")
print(f"  Entrainment + Fatigue: Recovery {ent_150['Recovery Time (final)']} ticks, Peak Fatigue {ent_150['Max Fatigue']}")
print()

# Population scaling
print("=" * 80)
print("POPULATION SCALING EFFECTS")
print("=" * 80)
print()

for mode in ['Coherence', 'Entrainment']:
    print(f"{mode}:")
    mode_results = [r for r in results if r['Mode'] == mode]
    for r in sorted(mode_results, key=lambda x: x['Population']):
        print(f"  Pop {r['Population']}: Recovery={r['Recovery Time (final)']}, MaxDev={float(r['Max Deviation']):.1f}, Gini={float(r['Cost Gini (final)']):.3f}, MaxFatigue={float(r['Max Fatigue']):.3f}")
    print()
