# Cost Derivation Findings

**Date:** 2025-12-26
**Data sources:** E003, E004a, E004b
**Reference:** `cost-types.md`

---

## 1. Cost Proxy Mapping (Validated)

| Cost Type | Proxy Metric | Interpretation |
|-----------|--------------|----------------|
| **Shock Cost** | `max_deviation` | Instantaneous perturbation load |
| **Repair Cost** | `recovery_time` | Time system remains impaired |
| **Maintenance Cost** | `baseline` | Ongoing regulatory effort |

---

## 2. E003: Mode Contrast Findings

### 2.1 Shock Cost (max_deviation)

| Strength | Coherence | Entrainment | Ratio (E/C) |
|----------|-----------|-------------|-------------|
| 10 | 4.5 | 4.5 | 1.0x |
| 20 | 6.0 | 4.8 | 0.8x |
| **40** | **8.1** | **34.3** | **4.2x** |
| **80** | **11.3** | **46.7** | **4.1x** |

**Finding:** Below threshold (~20-40), shock costs are equivalent. Above threshold, entrainment shock costs scale superlinearly while coherence remains linear.

### 2.2 Repair Cost (recovery_time)

| Strength | Coherence | Entrainment | Ratio (E/C) |
|----------|-----------|-------------|-------------|
| 10 | 0 | 0 | — |
| 20 | 0 | 0 | — |
| **40** | **8** | **49** | **6.1x** |
| **80** | **10** | **120** | **12.0x** |

**Finding:** Repair cost ratio increases with stress. At strength 80, entrainment takes 12x longer to recover than coherence.

### 2.3 Maintenance Cost (baseline)

| Strength | Coherence | Entrainment | Ratio (C/E) |
|----------|-----------|-------------|-------------|
| 10 | 102.8 | 36.3 | 2.8x |
| 20 | 93.2 | 21.5 | 4.3x |
| 40 | 79.7 | 29.8 | 2.7x |
| 80 | 87.2 | 24.0 | 3.6x |

**Finding:** Coherence pays 3-4x higher maintenance cost (baseline variance). This is the "cost of staying ready."

---

## 3. Cost Profiles Summary

### Coherence Regime
- **Profile:** "Pay as you go"
- **Maintenance:** High, stable (~91 avg)
- **Shock:** Linear scaling (4.5 → 11.3)
- **Repair:** Minimal (0 → 10 ticks)
- **Temporal pattern:** Continuous, bounded

### Entrainment Regime
- **Profile:** "Pay when stressed"
- **Maintenance:** Low (~28 avg)
- **Shock:** Superlinear scaling (4.5 → 46.7)
- **Repair:** Superlinear scaling (0 → 120 ticks)
- **Temporal pattern:** Episodic, unbounded

---

## 4. Critical Threshold

**Location:** Between perturbation strength 20 and 40

**Evidence:**
- Shock ratio jumps from ~1x to 4x
- Repair ratio jumps from 0 to 6x
- Entrainment's episodic costs begin exceeding coherence's maintenance advantage

**Interpretation:** This is the stress level where entrainment's brittleness becomes empirically visible. Below threshold, both regimes perform equivalently on episodic costs; entrainment appears "cheaper" due to lower maintenance. Above threshold, coherence's higher maintenance is offset by dramatically lower shock and repair costs.

---

## 5. E004b: Identity-Pull Parameter Sensitivity

Within coherence mode, identity-pull weight affects cost distribution:

| Identity-Pull | Avg Shock | Avg Repair | Avg Maintenance |
|---------------|-----------|------------|-----------------|
| 0.1 | 8.4 | 3.6 | 94.2 |
| **0.2** | **5.8** | **0.0** | **85.7** |
| 0.3 | 6.6 | 7.0 | 85.4 |
| 0.4 | 4.6 | 0.4 | 91.1 |

**Finding:** Identity-pull = 0.2 shows optimal repair cost (zero average) with moderate shock. Higher identity-pull (0.4) reduces shock cost but slightly increases maintenance. The relationship is non-monotonic, suggesting a sweet spot around 0.2-0.4.

---

## 6. Crossover Frequency Analysis

At what perturbation frequency does coherence become cost-advantageous?

Using simplified total cost model:
```
Total Cost = Maintenance + Frequency × (Shock + Repair)
```

**Result:** Coherence becomes preferable above ~0.5 perturbations per time unit (at average stress levels). This crossover point shifts lower as perturbation magnitude increases.

**Implication:** In volatile environments (frequent perturbations), coherence's higher maintenance cost is offset by its superior shock/repair performance. In stable environments (rare perturbations), entrainment's lower maintenance cost dominates.

---

## 7. Key Insights

### 7.1 The Core Tradeoff

| Dimension | Coherence | Entrainment |
|-----------|-----------|-------------|
| When costs are paid | Continuously | Episodically |
| Cost visibility | Always visible | Hidden until crisis |
| Scaling behavior | Linear | Superlinear |
| Risk profile | Bounded, predictable | Unbounded, catastrophic |

### 7.2 Distributional Implications

The cost-types document identified **distributional cost** as potentially underexplored. Current metrics are system-level aggregates. Future work should examine:

- Per-agent variance in heading deviation during recovery
- Identification of "high-burden" agents (those doing disproportionate regulatory work)
- Whether coherence distributes costs more evenly than entrainment

### 7.3 Mapping to Living Systems

From `cost-types.md`:

| Cost Type | Living Systems Analogue |
|-----------|------------------------|
| Maintenance | Autonomic regulation, immune readiness |
| Shock | Acute stress response, tissue damage |
| Repair | Tissue repair, emotional repair, institutional rebuilding |

The model confirms that "resilience is never free" — coherence's apparent advantage under stress comes at the cost of continuous regulatory effort.

---

## 8. Artifacts Generated

| File | Description |
|------|-------------|
| `exports/cost_derivation_E003.csv` | E003 with cost columns |
| `exports/cost_derivation_E004a.csv` | E004a with cost columns |
| `exports/cost_derivation_E004b.csv` | E004b with cost columns |
| `exports/cost_profiles_E003.png` | 4-panel cost profile comparison |
| `exports/cost_payment_patterns.png` | Stacked costs + frequency crossover |
| `exports/cost_identity_pull_E004b.png` | Identity-pull parameter effect |
| `notebooks/cost_derivation_analysis.ipynb` | Full analysis notebook |

---

## 9. Next Steps

1. **Temporal cost distribution**: Analyze variance time series to compute integrated shock/repair cost (area under curve)
2. **Distributional cost**: Track per-agent burden variance
3. **E004 threshold surface**: Map how threshold shifts with coupling and identity-pull parameters
4. **Capacity depletion**: Model fatigue accumulation under repeated perturbation

---

*"The coherence vs entrainment distinction is not about order vs chaos. It is about how costs are paid — continuously or episodically, locally or globally, visibly or catastrophically."*
