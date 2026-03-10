---
id: EXP-CM-C001
type: experiment
status: complete
scale: dyad
registered: 2026-01-11
completed: 2026-01-11
---

## Hypothesis

HYP-CM-005: Providing identity-pull (relief pathway) selectively to load-bearing agents is the causal mechanism preventing cascade in coherence mode.

**Result: Falsified.** The prediction was inverted. Selective protection of load-bearers increases cascade risk. Universal access is required.

## Addresses

- **RQ2** — What conditions cultivate or constrain adaptive capacity? This experiment discovered a design invariant: relief pathways must be universally available. Selective access is itself a constraint on adaptive capacity.
- **RQ3** — How do coherence and entrainment shape durability or brittleness? The phase misalignment cascade is a new brittleness mechanism: high-influence agents exiting while others remain coupled creates structural fragmentation.

## Technical Uncertainty (R&DTI)

I002 established that coherence prevents cascade via identity-pull. But whether identity-pull for load-bearers specifically was sufficient (targeted intervention) or whether universal access was required (structural property) was unknown. The distinction matters for translation: targeted interventions are cheaper but may not work; structural properties are more expensive but reliable. The direction could not be predicted from the I002 findings alone.

## Method

**Design:** 4 conditions x 50 replications.

**Conditions:**
- A: Load-bearers protected (identity-pull for agents with coupling-bias > 0.7 AND initial distance > 45 degrees)
- B: Non-load-bearers protected (identity-pull for remaining agents)
- C: All protected (full coherence mode)
- D: None protected (full entrainment mode)

**Variables:**
- IV: Selective identity-pull condition (A/B/C/D)
- DV: Spiral rate (% of runs), recovery time, max fatigue

**Load-bearer definition:** `coupling-bias > 0.7 AND initial-distance-from-mean > 45 degrees` — agents with high social sensitivity positioned far from consensus.

**Platform:** NetLogo BehaviorSpace (`coherence_model_simple.nlogox`) with `selective-identity-pull` chooser (C001 extension).

## Instruments

- INST-coherence-model — NetLogo entrainment-coherence ABM with selective identity-pull (C001 extension)

## Kill Criteria

- If all conditions showed equivalent cascade rates, the identity-pull mechanism identified in I002 would not be the causal factor.

## Observations

| Condition | Description | Predicted | Observed |
|-----------|-------------|-----------|----------|
| A | Load-bearers protected | ~0% spirals | **10% spirals** |
| B | Non-load-bearers protected | ~10% spirals | **0% spirals** |
| C | All protected | 0% spirals | 0% spirals |
| D | Entrainment (none) | ~10% spirals | 0% spirals |

Predictions were inverted for conditions A and B.

## Findings

**HYP-CM-005: Falsified.** The prediction was exactly inverted.

**Mechanism discovered — Phase misalignment cascade:**

When high-influence agents escape (condition A) while low-influence agents remain coupled:
1. Social field fragments between identity and entrainment attractors
2. Non-protected agents oscillate between conflicting signals
3. Oscillation cost cascades through high-coupling agents → fatigue → spiral

**Three key findings:**

1. **Protecting load-bearers INCREASES spiral rate** (10% vs 0% for all other conditions). The intervention designed to help made things worse.

2. **Universal access prevents cascade.** Condition C (all protected) has 0% spirals. The relief pathway works — but only when everyone has access.

3. **Selective escape destabilises.** High-influence agents exiting while others remain coupled creates phase misalignment. The "protection" of key nodes externalises collapse risk onto those who remain coupled.

**Design invariant (from falsified hypothesis):**
> Relief pathways must be universally available. Selective escape for high-influence agents destabilises rather than protects.

## Triples Produced

```spl
; --- Entities ---
(entity EXP-CM-C001 experiment "Selective escape valve: targeted vs universal relief")
(entity HYP-CM-005 hypothesis "Selective identity-pull for load-bearers prevents cascade")
(entity COND-001 condition "Universal relief pathway access")

; --- Epistemic ---
(evidence EXP-CM-C001 addresses RQ2)
(evidence EXP-CM-C001 addresses RQ3)
(evidence EXP-CM-C001 falsifies HYP-CM-005)
(evidence EXP-CM-C001 at-scale dyad)

; --- Conditions ---
(evidence COND-001 instantiated-by EXP-CM-C001)

; --- Generative ---
(evidence EXP-CM-I002 triggered EXP-CM-C001)

; --- Instrumentation ---
(evidence EXP-CM-C001 uses-instrument INST-coherence-model)

; --- R&DTI ---
(rdti EXP-CM-C001 classification core)
```

## Revisions

| Date | What Changed | Trigger | Prior Position |
|------|-------------|---------|----------------|

## New Knowledge (R&DTI)

1. **Selective intervention can invert its intended effect.** The experiment designed to confirm a targeted protection mechanism instead demonstrated that the targeting itself causes harm. This is a category of finding that cannot be reached by incremental reasoning from prior results — it required the experimental test.

2. **Phase misalignment cascade is a new failure mode.** When agents with different coupling regimes coexist in the same social field, the field itself becomes incoherent. Agents coupled to a fragmenting field pay higher oscillation costs than agents in a uniformly entrained or uniformly coherent field. This mechanism was not anticipated.

3. **The design invariant is counter-intuitive and load-bearing.** "Protect the most vulnerable" is the intuitive intervention. The finding says the opposite: protect everyone or protect no one, but do not protect selectively by influence. This has direct translation implications for organisational governance, crisis response, and platform design.

---

*Data: `exports/` (C001 BehaviorSpace export)*
*Analysis: `notebooks/behaviorspace_analysis.ipynb`*
*Trace: `dev-updates/DEV_UPDATE_2026-01-11_c001-selective-escape-valve.md`*
