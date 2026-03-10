---
id: EXP-CM-003
type: experiment
status: complete
scale: dyad
registered: 2025-12-26
completed: 2025-12-26
---

## Hypothesis

HYP-CM-001: Entrainment degrades superlinearly above a critical stress threshold, while coherence maintains proportional (linear) response across all stress levels.

## Addresses

- **RQ3** — How do coherence and entrainment shape the durability or brittleness of adaptation processes? This experiment directly tests the durability claim: entrainment appears stable under low stress but is structurally brittle.
- **RQ2** — What conditions cultivate or constrain adaptive capacity? The critical threshold between strength 20-40 is itself a condition: systems below it look equivalent, systems above it diverge sharply.

## Technical Uncertainty (R&DTI)

Whether identity-preserving coupling (coherence) would maintain proportional stress response at high perturbation strengths was unknown. Prior work on phase-locking showed synchronisation advantages at low stress, but the scaling behaviour under increasing perturbation had not been tested in a model that distinguishes coherence from entrainment as separate coupling regimes.

## Method

**Design:** 2x4 factorial (coupling mode x perturbation strength), single perturbation.

**Variables:**
- IV1: Coupling mode (entrainment vs coherence)
- IV2: Perturbation strength (10, 20, 40, 80)
- DV: Recovery time (ticks from perturbation end to tolerance return), max deviation from baseline

**Controls:** Population size, coupling strength, noise level, identity-pull weight, social-pull weight held constant across conditions.

**Platform:** NetLogo BehaviorSpace (`coherence_model_simple.nlogox`).

**Analysis:** Stress amplification ratios (entrainment / coherence) computed for recovery time and max deviation at each strength level.

## Instruments

- INST-coherence-model — NetLogo ABM with recovery tracking v2

## Kill Criteria

- If coherence mode showed equivalent or worse degradation to entrainment, the Coherence Theorem would be falsified at the stress-scaling level.

## Observations

Stress amplification ratios (entrainment / coherence):

| Perturbation Strength | Recovery Ratio | Max Deviation Ratio |
|-----------------------|----------------|---------------------|
| 10 | 1x | 1x |
| 20 | 1x | 0.8x |
| 40 | **6x** | **4x** |
| 80 | **12x** | **4x** |

Critical threshold between strength 20-40. Below threshold, both regimes perform equivalently. Above threshold, coherence maintains linear scaling while entrainment shows superlinear degradation.

## Findings

**HYP-CM-001: Confirmed.**

Entrainment brittleness is non-linear — it appears stable at low stress but degrades superlinearly above a critical threshold (~20-40 perturbation strength). Coherence maintains proportional response across all tested stress levels. The effect sizes are large: 6-12x recovery time ratio and 4x max deviation ratio at high stress.

The threshold behaviour means that systems operating under entrainment may appear functional and even efficient under normal conditions, with their structural vulnerability invisible until stress exceeds the critical threshold. This has direct implications for organisational and socio-technical systems that optimise for alignment.

## Triples Produced

```spl
; --- Entities ---
(entity EXP-CM-003 experiment "Stress scaling response: entrainment vs coherence")
(entity HYP-CM-001 hypothesis "Entrainment degrades superlinearly above critical stress threshold")
(entity INST-coherence-model instrument "NetLogo entrainment-coherence ABM")

; --- Epistemic ---
(evidence EXP-CM-003 addresses RQ3)
(evidence EXP-CM-003 addresses RQ2)
(evidence EXP-CM-003 confirms HYP-CM-001)
(evidence EXP-CM-003 at-scale dyad)

; --- Instrumentation ---
(evidence EXP-CM-003 uses-instrument INST-coherence-model)

; --- R&DTI ---
(rdti EXP-CM-003 classification core)
```

## Revisions

| Date | What Changed | Trigger | Prior Position |
|------|-------------|---------|----------------|

## New Knowledge (R&DTI)

1. **Threshold dynamics are non-linear and discontinuous.** The transition between equivalent performance (strength 10-20) and dramatic divergence (strength 40-80) is not gradual — it's a regime shift. This could not have been determined without the factorial design crossing the threshold.

2. **Coherence's advantage is structural, not parametric.** The linear scaling of coherence mode under increasing stress is a property of the coupling architecture (identity-pull providing a return attractor), not a parameter tuning. This distinguishes the finding from optimisation results.

3. **The critical threshold range (20-40) identifies a diagnostic window.** Systems within this range are where the entrainment/coherence distinction first becomes empirically visible. Below it, the distinction is theoretical. Above it, the distinction is obvious. The threshold range is where the claim is most testable and most useful.

---

*Data: `exports/coherence_model_simple_E003-spreadsheet.csv`*
*Plots: `exports/E003_theorem_validation.png`, `exports/E003_stress_scaling_v2.png`*
*Analysis: `notebooks/behaviorspace_analysis.ipynb`*
