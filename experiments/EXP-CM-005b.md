---
id: EXP-CM-005b
type: experiment
status: complete
scale: dyad
registered: 2025-12-27
completed: 2025-12-27
---

## Hypothesis

HYP-CM-003: Under repeated perturbation, entrainment's cost advantage disappears and a hidden recovery tax emerges that is invisible under single perturbation.

## Addresses

- **RQ3** — How do coherence and entrainment shape the durability or brittleness of adaptation processes? This experiment tests durability under sustained stress, not just acute response. The "hidden tax" is a brittleness mechanism that only manifests temporally.
- **RQ2** — What conditions cultivate or constrain adaptive capacity? Perturbation frequency relative to recovery capacity is itself a condition: systems that cannot stabilise between shocks accumulate invisible debt.

## Technical Uncertainty (R&DTI)

EXP-CM-005a showed similar cumulative costs between modes under single perturbation. Whether this equivalence held under repeated stress was unknown. It was plausible that (a) the cost gap would scale proportionally, (b) entrainment's faster synchronisation would provide per-perturbation efficiency that compounds favourably, or (c) entrainment's incomplete recovery between shocks would create compounding debt. The direction required temporal cost decomposition under periodic perturbation.

## Method

**Design:** 2x2 factorial (coupling mode x perturbation strength), periodic perturbation.

**Variables:**
- IV1: Coupling mode (entrainment vs coherence)
- IV2: Perturbation strength (60)
- DV: Cumulative cost, cost Gini, recovery time, recovery cost, max deviation

**Perturbation schedule:** Periodic (multiple perturbations during run).

**Platform:** NetLogo BehaviorSpace (`coherence_model_simple.nlogox`) with full cost metrics.

## Instruments

- INST-coherence-model — NetLogo entrainment-coherence ABM with cost tracking v1

## Kill Criteria

- If periodic stress produced the same cost profile as single perturbation scaled linearly, the temporal dimension would add no new information.

## Observations

| Mode | Strength | Cumulative Cost | Cost Gini | Recovery Time | Recovery Cost |
|------|----------|-----------------|-----------|---------------|---------------|
| Coherence | 60 | 3650 | 0.025 | 0 | 5.6 |
| Entrainment | 60 | 3883 | 0.024 | 132 | 168.2 |

Max deviation: entrainment 44 degrees vs coherence 3.4 degrees (13x ratio).

## Findings

**HYP-CM-003: Confirmed.**

Three findings that could not have been observed under single perturbation:

1. **Gini gap collapsed** (1.08x vs 1.8x in single perturbation). Repeated stress homogenises cost distribution — even coherence mode's diversity carriers get ground down.

2. **Recovery cost explodes.** Entrainment pays 30x more in recovery cost (168.2 vs 5.6). The "hidden tax": constant recovery effort that never fully succeeds between perturbations.

3. **Max deviation compounds.** Entrainment reaches 44 degrees vs 3.4 degrees (13x ratio) because it cannot stabilise between shocks. Each perturbation compounds on incomplete recovery from the last.

Under single perturbation, cumulative costs look similar. Under repeated stress, entrainment's hidden tax emerges: constant recovery effort that never fully succeeds.

## Triples Produced

```spl
; --- Entities ---
(entity EXP-CM-005b experiment "Cost distribution under periodic perturbation")
(entity HYP-CM-003 hypothesis "Repeated stress reveals entrainment hidden recovery tax")

; --- Epistemic ---
(evidence EXP-CM-005b addresses RQ3)
(evidence EXP-CM-005b addresses RQ2)
(evidence EXP-CM-005b confirms HYP-CM-003)
(evidence EXP-CM-005b strengthens EXP-CM-003)
(evidence EXP-CM-005b at-scale dyad)

; --- Generative ---
(evidence EXP-CM-005a triggered EXP-CM-005b)

; --- Instrumentation ---
(evidence EXP-CM-005b uses-instrument INST-coherence-model)

; --- R&DTI ---
(rdti EXP-CM-005b classification core)
```

## Revisions

| Date | What Changed | Trigger | Prior Position |
|------|-------------|---------|----------------|

## New Knowledge (R&DTI)

1. **Temporal stress decomposition reveals hidden costs.** Single-perturbation analysis fundamentally understates entrainment's vulnerability. The 30x recovery cost ratio is invisible under single perturbation designs. This methodological insight — that perturbation frequency is a critical design variable — could not have been determined without the periodic condition.

2. **Cost distribution equalises under sustained stress.** The Gini gap collapse means coherence's distributional disadvantage (from EXP-CM-005a) is transient — it applies to acute shocks but not chronic stress. Under sustained pressure, both modes distribute cost similarly, but entrainment pays far more total recovery cost.

3. **Compounding deviation is a cascade precursor.** The 13x max deviation ratio under periodic stress (vs 4x under single perturbation) demonstrates that entrainment's inability to fully recover between shocks creates compounding instability. This connects to I002's spiral trigger finding: compounding deviation is the mechanism that eventually triggers the fatigue cascade.

---

*Data: `exports/` (E005b BehaviorSpace export)*
*Analysis: `notebooks/behaviorspace_analysis.ipynb`*
