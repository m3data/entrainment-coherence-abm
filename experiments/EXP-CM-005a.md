---
id: EXP-CM-005a
type: experiment
status: complete
scale: dyad
registered: 2025-12-27
completed: 2025-12-27
---

## Hypothesis

HYP-CM-002: Coherence mode distributes adaptation costs less equally than entrainment because identity preservation concentrates cost on agents whose preferred-headings diverge most from the group mean.

## Addresses

- **RQ3** — How do coherence and entrainment shape the durability or brittleness of adaptation processes? This experiment tests the cost distribution dimension: durability has a price, and who pays it differs by regime.
- **RQ2** — What conditions cultivate or constrain adaptive capacity? The finding that diversity carriers bear disproportionate cost is itself a condition — systems must account for this or risk losing their diverse members.

## Technical Uncertainty (R&DTI)

Whether coherence mode's structural advantage in recovery (established by EXP-CM-003) came with a hidden distributional cost was unknown. It was plausible that identity-preserving coupling would either (a) distribute costs more equally because each agent maintains its own attractor, or (b) concentrate costs on divergent agents who must resist stronger social pull. The direction could not be determined without measuring per-agent cost distribution.

## Method

**Design:** 2x4 factorial (coupling mode x perturbation strength), single perturbation.

**Variables:**
- IV1: Coupling mode (entrainment vs coherence)
- IV2: Perturbation strength (10, 20, 40, 80)
- DV: Cumulative cost (metabolic + shock + recovery), cost Gini coefficient, recovery time

**Cost metrics:** Per-agent metabolic cost, shock cost, recovery cost, cumulative cost, cost variance, cost Gini coefficient.

**Platform:** NetLogo BehaviorSpace (`coherence_model_simple.nlogox`) with cost tracking v1.

## Instruments

- INST-coherence-model — NetLogo entrainment-coherence ABM with cost tracking v1

## Kill Criteria

- If cost distribution showed no significant difference between modes, the cost dimension of the coherence/entrainment distinction would be uninformative.

## Observations

| Mode | Strength | Cumulative Cost | Cost Gini | Recovery Time |
|------|----------|-----------------|-----------|---------------|
| Coherence | 10 | 1127 | 0.039 | 0 |
| Coherence | 80 | 1567 | 0.032 | 22 |
| Entrainment | 10 | 1070 | 0.017 | 0 |
| Entrainment | 80 | 1584 | 0.025 | 183 |

Average Gini: coherence 0.035, entrainment 0.020 (~2x higher in coherence).

## Findings

**HYP-CM-002: Confirmed.**

Coherence has approximately 2x higher Gini coefficient than entrainment (0.035 vs 0.020 average). Agents with preferred-headings far from the group mean pay more to maintain their identity. Entrainment homogenizes everyone toward the same attractor, spreading cost evenly.

The cost of diversity falls on those who hold it. This is not a flaw of coherence — it is the structural price of maintained difference. Systems that want the benefits of diversity (better recovery, linear scaling under stress) must account for the distributional burden on their most different members.

## Triples Produced

```spl
; --- Entities ---
(entity EXP-CM-005a experiment "Cost distribution under single perturbation")
(entity HYP-CM-002 hypothesis "Coherence concentrates cost on divergent agents")

; --- Epistemic ---
(evidence EXP-CM-005a addresses RQ3)
(evidence EXP-CM-005a addresses RQ2)
(evidence EXP-CM-005a confirms HYP-CM-002)
(evidence EXP-CM-005a at-scale dyad)

; --- Generative ---
(evidence EXP-CM-003 triggered EXP-CM-005a)

; --- Instrumentation ---
(evidence EXP-CM-005a uses-instrument INST-coherence-model)

; --- R&DTI ---
(rdti EXP-CM-005a classification core)
```

## Revisions

| Date | What Changed | Trigger | Prior Position |
|------|-------------|---------|----------------|

## New Knowledge (R&DTI)

1. **Diversity has a measurable distributional cost.** The Gini difference (0.035 vs 0.020) quantifies what was previously only a theoretical intuition: identity preservation creates unequal burden. This could not have been determined without per-agent cost tracking.

2. **Cumulative costs are similar across modes; distribution differs.** Total system cost is comparable — the difference is who pays, not how much. This reframes the coherence advantage: it is not cheaper, it is more durable, and the price falls on the agents whose difference makes the system resilient.

---

*Data: `exports/` (E005a BehaviorSpace export)*
*Analysis: `notebooks/behaviorspace_analysis.ipynb`*
