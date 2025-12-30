"""
S001a_rep Scale Sensitivity Analysis with Replication
Analyzes 30 runs (6 conditions × 5 repetitions) for confidence intervals
"""

import numpy as np

filepath = '/Users/m3untold/Code/EarthianLabs/Netlogo-Models/coherence-model/exports/coherence_model_simple S001a_rep_scale_sensitivity_5000-spreadsheet.csv'

with open(filepath, 'r') as f:
    lines = f.readlines()

# Find key rows
for i, line in enumerate(lines):
    if '"entrainment-mode?"' in line:
        mode_row = i
    if '"population"' in line:
        pop_row = i
    if '"[reporter]"' in line:
        header_row = i
        break

# Parse run configurations
mode_line = lines[mode_row].strip().split(',')
pop_line = lines[pop_row].strip().split(',')

# 15 metrics per run (step + 14 metrics)
metrics_per_run = 15

# Build condition mapping: (mode, population) -> list of run indices
conditions = {}
run_idx = 0
for j in range(1, len(mode_line), metrics_per_run):
    if j < len(mode_line) and j < len(pop_line):
        mode_val = mode_line[j].strip('"')
        pop_val = pop_line[j].strip('"')
        if mode_val and pop_val:
            key = ('Entrainment' if mode_val == 'true' else 'Coherence', int(pop_val))
            if key not in conditions:
                conditions[key] = []
            conditions[key].append(run_idx)
            run_idx += 1

# Parse summary rows
summaries = {}
for line in lines[header_row+1:header_row+5]:
    parts = line.strip().split(',')
    label = parts[0].strip('"[]')
    summaries[label] = [p.strip('"') for p in parts]

# Extract metrics by condition
# Metric offsets: recovery-time=1, max-deviation=3, cost-gini=10, max-fatigue-level=12
results = {}

for (mode, pop), run_indices in conditions.items():
    recovery_times = []
    max_deviations = []
    max_fatigues = []
    cost_ginis = []

    for run_idx in run_indices:
        offset = 1 + run_idx * metrics_per_run

        # Get max values for this run
        max_row = summaries['max']
        if offset + 12 < len(max_row):
            try:
                recovery_times.append(float(max_row[offset + 1]))
                max_deviations.append(float(max_row[offset + 3]))
                max_fatigues.append(float(max_row[offset + 12]))
            except (ValueError, IndexError):
                pass

        # Get final gini
        final_row = summaries['final']
        if offset + 10 < len(final_row):
            try:
                cost_ginis.append(float(final_row[offset + 10]))
            except (ValueError, IndexError):
                pass

    results[(mode, pop)] = {
        'recovery_time': (np.mean(recovery_times), np.std(recovery_times)),
        'max_deviation': (np.mean(max_deviations), np.std(max_deviations)),
        'max_fatigue': (np.mean(max_fatigues), np.std(max_fatigues)),
        'cost_gini': (np.mean(cost_ginis), np.std(cost_ginis)),
        'n': len(recovery_times)
    }

# Print results
print("=" * 90)
print("S001a_rep SCALE SENSITIVITY RESULTS (5 repetitions per condition)")
print("=" * 90)
print()

print(f"{'Mode':<12} {'Pop':>4} | {'Recovery Time':>20} | {'Max Fatigue':>18} | {'Max Deviation':>18}")
print(f"{'':12} {'':>4} | {'mean':>9} {'±sd':>10} | {'mean':>8} {'±sd':>9} | {'mean':>8} {'±sd':>9}")
print("-" * 90)

for mode in ['Coherence', 'Entrainment']:
    for pop in [150, 300, 500]:
        key = (mode, pop)
        if key in results:
            r = results[key]
            rec_mean, rec_sd = r['recovery_time']
            fat_mean, fat_sd = r['max_fatigue']
            dev_mean, dev_sd = r['max_deviation']
            print(f"{mode:<12} {pop:>4} | {rec_mean:>9.1f} {rec_sd:>10.1f} | {fat_mean:>8.3f} {fat_sd:>9.3f} | {dev_mean:>8.1f} {dev_sd:>9.1f}")

print()
print("=" * 90)
print("KEY QUESTIONS")
print("=" * 90)
print()

# Check if 300-agent spike is consistent
ent_150 = results.get(('Entrainment', 150))
ent_300 = results.get(('Entrainment', 300))
ent_500 = results.get(('Entrainment', 500))

if ent_300:
    rec_300_mean, rec_300_sd = ent_300['recovery_time']
    fat_300_mean, fat_300_sd = ent_300['max_fatigue']

    print(f"1. Is the 300-agent spike consistent?")
    print(f"   Entrainment 300: Recovery = {rec_300_mean:.0f} ± {rec_300_sd:.0f}")
    print(f"   Entrainment 300: Max Fatigue = {fat_300_mean:.3f} ± {fat_300_sd:.3f}")

    # Coefficient of variation
    cv = rec_300_sd / rec_300_mean if rec_300_mean > 0 else float('inf')
    print(f"   Coefficient of Variation (CV): {cv:.2f}")

    if cv < 0.3:
        print(f"   → LOW VARIANCE: 300-agent effect appears CONSISTENT")
    elif cv < 0.6:
        print(f"   → MODERATE VARIANCE: Effect present but variable")
    else:
        print(f"   → HIGH VARIANCE: May be stochastic artifact")

print()

# Compare populations
if ent_150 and ent_300 and ent_500:
    print(f"2. Population comparison (Entrainment):")
    for pop, data in [(150, ent_150), (300, ent_300), (500, ent_500)]:
        mean, sd = data['recovery_time']
        print(f"   Pop {pop}: {mean:.0f} ± {sd:.0f} ticks")

    # Is 300 significantly different from 150 and 500?
    rec_150_mean, rec_150_sd = ent_150['recovery_time']
    rec_500_mean, rec_500_sd = ent_500['recovery_time']

    # Simple check: is 300 more than 2 SDs above the others?
    threshold_150 = rec_150_mean + 2 * rec_150_sd
    threshold_500 = rec_500_mean + 2 * rec_500_sd

    print()
    if rec_300_mean > threshold_150 and rec_300_mean > threshold_500:
        print(f"   → 300-agent recovery ({rec_300_mean:.0f}) is >2σ above 150 ({threshold_150:.0f}) and 500 ({threshold_500:.0f})")
        print(f"   → SIGNIFICANT POPULATION EFFECT detected")
    else:
        print(f"   → Population differences may not be significant")

print()

# Coherence stability
coh_150 = results.get(('Coherence', 150))
coh_300 = results.get(('Coherence', 300))
coh_500 = results.get(('Coherence', 500))

if coh_150 and coh_300 and coh_500:
    print(f"3. Coherence stability:")
    all_coh_rec = []
    for pop, data in [(150, coh_150), (300, coh_300), (500, coh_500)]:
        mean, sd = data['recovery_time']
        all_coh_rec.append(mean)
        print(f"   Pop {pop}: {mean:.0f} ± {sd:.0f} ticks")

    max_coh = max(all_coh_rec)
    if max_coh < 50:
        print(f"   → Coherence STABLE across all population sizes (max recovery {max_coh:.0f} ticks)")
    else:
        print(f"   → Some coherence runs showing extended recovery")
