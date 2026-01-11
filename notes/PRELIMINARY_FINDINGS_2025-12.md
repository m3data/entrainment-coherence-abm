# Coherence vs Entrainment Model: Preliminary Findings

**December 2025 – January 2026**
**Status:** Working synthesis of experimental program (E001–I004, C001)
**Document type:** Exploratory findings for mechanism discovery

---

## Purpose and Scope

This document synthesizes findings from 14+ experiments conducted with the Coherence vs Entrainment NetLogo model between December 26–27, 2025. The model is a **conceptual instrument** — an exploratory sandbox for testing ideas about collective dynamics, not a predictive simulator of any specific real-world system.

### The Coherence Theorem

The model tests the following proposition:

> *In agent-based systems with heterogeneous identity and bounded coupling, regimes that preserve internal diversity will exhibit lower peak disruption and faster recovery under repeated perturbation than regimes optimised for phase alignment.*

This theorem is presented as a **testable claim within model scope**. It is not a universal law; its applicability to real-world systems requires further empirical validation.

### How to Read This Document

Readers interested primarily in mechanisms may focus on: **Stress-Scaling Results**, **Spiral Investigation**, and **Mechanism Synthesis**. The Executive Summary provides effect sizes; the Experimental Program Overview traces each claim to its source experiment.

### Key Contribution

This work identifies a previously under-modeled failure mechanism: **load concentration without relief pathways**. The distinction between synchrony-based coordination (entrainment) and identity-preserving coordination (coherence) has design-relevant implications for systems that must absorb stress without cascading failure.

**C001 Update:** Causal testing revealed that relief pathways must be *universally* available to prevent cascade. Selective escape valves for high-influence agents (load-bearers) *increase* spiral risk by creating phase misalignment cascades. This inverts the intuitive prediction and has implications for differential resilience strategies in heterogeneous systems.

---

## Glossary

| Term | Definition |
|------|------------|
| **Alignment work** | Cumulative heading adjustment toward the social field; a measure of coordination effort |
| **Cascade failure** | Runaway amplification of recovery cost driven by exhausted load-bearing agents |
| **Coherence** | Dynamic capacity to maintain integrity while absorbing perturbation; operationalized via identity-preserving coupling with dual attractors |
| **Entrainment** | Phase-locking toward collective alignment; operationalized via strong coupling to population mean |
| **Load-bearing agents** | Agents with high social sensitivity and positions far from consensus who bear disproportionate alignment work (also: sacrificial stabilizers) |
| **Load concentration** | Emergent burden asymmetry where specific agents bear disproportionate stabilization cost |
| **Relief pathway** | Independent internal reference point (e.g., preferred-heading) usable for recovery under fatigue; functions as escape valve under stress |
| **Phase misalignment cascade** | Destabilization caused by mixed coupling regimes where some agents escape to identity while others remain entrained, creating oscillation between conflicting attractors |
| **Selective escape valve** | Relief pathway granted to a subset of agents; C001 shows this can destabilize rather than protect |
| **Stress transition region** | Perturbation level where regime differences become pronounced (empirically ~30-50 under current parameterization) |

---

## Assumptions and Scope

This model makes several simplifying assumptions that bound interpretation:

1. **Identity-pull is restorative, not factional.** Agents return toward their preferred-heading, which is set at random during setup. The model does not explore identity-pull toward polarized clusters or adversarial factions.

2. **Costs are internal and additive.** Costs accumulate within agents; there is no cost externalization to neighbors or environment. This may underestimate real-world cascade risks.

3. **Global variance is a proxy, not a full coherence measure.** Heading variance captures collective dispersion but not cluster structure, local coherence, or pattern persistence.

4. **Single-scale dynamics.** The model assumes one population scale; cross-scale coherence/incoherence interactions are not modeled.

5. **Homogeneous perturbation.** Perturbations affect all agents equally; spatially local or targeted perturbations are not tested.

### What Would Weaken or Falsify the Theorem

- Finding conditions where entrainment outperforms coherence at high stress
- Discovering that identity-pull creates pathological fragmentation under certain configurations
- Evidence that load concentration operates differently (or doesn't occur) in more complex topologies
- Parameter regimes where the stress transition region disappears or inverts

---

## Executive Summary

### Core Finding

**Load concentration without relief pathways is sufficient to explain cascade failure within this model.**

When systems coordinate through phase-locking (entrainment), certain agents — those most socially sensitive and most different from consensus — bear disproportionate cost to stabilize the collective. If these load-bearing agents lack independent recovery pathways, they exhaust themselves and trigger cascade failure.

Systems that preserve individual reference points (coherence) provide relief pathways that prevent this cascade, even when load concentration is higher.

### Key Results (Effect Sizes)

| Comparison | Effect Size | Source |
|------------|-------------|--------|
| Recovery time ratio at high stress | **12× longer** (entrainment vs coherence) | E003 |
| Peak deviation ratio at high stress | **4× higher** (entrainment vs coherence) | E003 |
| Recovery cost under periodic stress | **30× higher** (entrainment vs coherence) | E005b |
| Fatigue amplification of recovery | **11×** vs 7× (entrainment vs coherence) | F001 |
| Alignment work ratio (load-bearing vs rest) | 1.6–2× | I004 |
| Spiral rate: load-bearers-only protected | **10%** (vs 0% for all-protected) | C001 |
| Spiral rate: non-load-bearers-only protected | **0%** (same as all-protected) | C001 |

### Theorem Refinements

The original theorem is supported, with these refinements:

1. **Threshold-dependence:** The distinction only matters above a stress transition region (~30-50 under current parameters). Below this, both regimes perform equivalently.

2. **Configuration-level causality:** Population-level variance is a proxy signal. The causal mechanism operates at the agent level: which agents bear load, how many share it, and whether they have relief pathways.

3. **Relief pathway as key mechanism:** The theorem's "preserved diversity" operates specifically through the availability of internal reference points (escape valves), not diversity per se.

4. **Universal access requirement (C001):** Relief pathways must be universally available to prevent cascade. Selective access for high-influence agents creates phase misalignment that *increases* instability. This inverts the intuitive prediction that protecting load-bearers would be sufficient.

---

## The Regime Distinction

### Entrainment Mode

- **Attractor structure:** Single shared attractor (collective mean heading)
- **Coupling:** `coupling-strength × coupling-bias × (avg-heading - heading)`
- **Baseline:** Low variance (~21-36°)
- **Under stress:** Shared attractor destabilizes → feedback amplification
- **Failure mode:** Cascade via exhausted load-bearers

### Coherence Mode

- **Attractor structure:** Dual attractors (social field + preferred-heading)
- **Coupling:** Attenuated social pull + identity-pull toward preferred-heading
- **Baseline:** Moderate variance (~80-103°)
- **Under stress:** Local basins remain stable → parallel recovery
- **Resilience mechanism:** Overloaded agents can fall back to individual attractor

---

## Stress-Scaling Results

**Source:** E003, E004a, E004b

### The Stress Transition Region

At low stress (perturbation strength 10–20), both regimes perform equivalently — near-instant recovery, minimal disruption. Both are viable under favorable conditions.

Above a transition region (~30-50 under current parameters), the regimes diverge:

| Perturbation Strength | Recovery Ratio (Ent/Coh) | Max Deviation Ratio |
|-----------------------|--------------------------|---------------------|
| 10 | 1× | 1× |
| 20 | 1× | 0.8× |
| 40 | **6×** | **4×** |
| 80 | **12×** | **4×** |

Coherence maintains approximately linear scaling. Entrainment shows superlinear degradation.

> **Interpretive note:**
> The stress transition region reflects a *loss of control authority* in entrainment rather than a failure of coordination per se. Below this region, both regimes have sufficient restoring capacity; above it, entrainment's single shared attractor becomes dynamically unreliable. This is a regime boundary, not a coordination threshold.

### Parameter Dependence

The transition region location depends on:
- **coupling-strength:** Higher coupling → lower threshold (E004a)
- **identity-pull-weight:** Stronger identity-pull → higher stress tolerance (E004b)
- **noise-level, population size:** Not yet systematically explored

The ~30-50 region is an **empirical observation under current parameterization**, not a universal constant. Change-point analysis or piecewise regression could formalize this boundary with uncertainty estimates.

### Mechanism

Entrainment's single shared attractor becomes unreliable when perturbation magnitude exceeds the system's restoring force. Agents chase a wandering, noisy target — each agent's disorientation contributes to every other's.

Coherence agents have local basins (preferred-heading) that remain stable regardless of global state. Recovery is parallel and independent.

---

## Cost and Fatigue Dynamics

### Cost Distribution (E005a, E005b)

**Finding:** Coherence has ~1.8× higher cost Gini (more unequal) under single perturbation.

| Mode | Average Gini |
|------|-------------|
| Coherence | 0.035 |
| Entrainment | 0.020 |

**Interpretation:** Agents whose preferred-headings differ from consensus pay more to maintain distinctiveness. The cost of diversity falls on those who hold it.

**However:** Under periodic stress, the Gini gap collapses (1.08×) as repeated perturbation homogenizes everyone — and entrainment pays **30× more in recovery cost** because it never stabilizes between shocks.

### Fatigue Feedback (F001)

When accumulated costs affect dynamics (fatigue-enabled=true):

| Mode | Recovery (fatigue OFF→ON) | Peak Fatigue |
|------|---------------------------|--------------|
| Coherence | 7→50 ticks (7×) | 0.04 |
| Entrainment | 119→1330 ticks (**11×**) | **0.65** |

**Coherence + Fatigue = Metabolic Governor**
- Identity-pull provides recovery pathway independent of social field
- Costs rise → sluggishness → system slows → costs decay → equilibrium
- Peak fatigue: barely crosses threshold

**Entrainment + Fatigue = Depletion Spiral**
- No independent recovery pathway
- Costs rise → sluggishness → can't align → more cost → SPIRAL
- Peak fatigue: deep into saturation (0.65)

---

## Spiral Investigation: The Falsification Arc

This sequence (I001–I004) is presented as an example of **iterative mechanism falsification**, moving from population-level hypotheses to agent-level causal structure. The investigation into what causes entrainment spirals illustrates mechanism discovery through progressive refinement.

### I001: Initial Heading Variance (NULL)

**Hypothesis:** More initial disorder → more alignment work → more cost → spiral

**Result:** Not supported (r = 0.163). Spiral and non-spiral runs started with essentially identical variance (~89-91°).

### I002: Coupling-Bias Variance (PROXY SIGNAL)

**Hypothesis:** Heterogeneous social sensitivity predicts spiral

**Result:** Statistically associated (t = 3.777, p = 0.011). Spiral runs had higher coupling-bias variance (0.096 vs 0.084).

**Note on statistical approach:** This p-value comes from exploratory hypothesis testing across multiple predictors. It should be treated as **evidence for mechanism investigation**, not confirmatory inference. Effect size (Cohen's d ≈ 2.0) is large.

### I003: Controlled Variance Sweep (NULL DOSE-RESPONSE)

**Hypothesis:** If variance causes spirals, higher variance should produce higher spiral rate.

**Result:** No dose-response relationship (r = 0.000). Variance levels 0.02 and 0.10 both showed ~10% spiral rates.

**Interpretation:** I002 detected a proxy signal, not a cause. Population-level variance correlates with configuration-level vulnerability but doesn't directly produce it.

**Key distinction:** Population variance *flags* vulnerability; configuration *produces* it.

### I004: Configuration Tracking (MECHANISM IDENTIFIED)

**Discovery:** Load concentration + absence of relief pathways → cascade risk.

**Load-Bearing Agents** (hereafter used instead of "at-risk agents"):
- ~22% of population do 1.6–2× the alignment work
- Definition: coupling-bias > 0.7 AND initial-distance-from-mean > 45°

**Concentration Effect:**
- Spirals occurred when there were FEWER load-bearing agents (p = 0.027, exploratory)
- More load-bearers → burden distributed → each survives
- Fewer load-bearers → each overloaded → exhaustion → cascade

**The Coherence Paradox:**
- Coherence has HIGHER work inequality (Gini 0.447 vs 0.335)
- Yet ZERO spirals across 50 runs
- Resolution: Relief pathway (identity-pull) allows high-work agents to recover

---

## C001: Selective Escape Valve (Causal Test)

**Source:** C001_selective_escape_valve (January 2026, 200 runs)

### Purpose

Test whether relief pathway availability is the causal mechanism preventing cascade by selectively granting identity-pull to subgroups.

### Design

| Condition | entrainment-mode? | selective-identity-pull | Description |
|-----------|-------------------|-------------------------|-------------|
| A | false | "load-bearers-only" | Only load-bearers protected |
| B | false | "non-load-bearers-only" | Load-bearers unprotected |
| C | false | "all" | Baseline coherence |
| D | true | (ignored) | Baseline entrainment |

50 repetitions per condition. Parameters matched I004 (perturbation-strength=60, fatigue-enabled=true, 5000 ticks).

### Pre-Registered Predictions

1. Condition A (load-bearers-only) → near-zero spiral rate
2. Condition B (non-load-bearers-only) → spiral rate ≈ Condition D
3. Condition C → zero spirals (baseline coherence)
4. Condition D → ~10% spiral rate (baseline entrainment)

**Falsification criterion:** If Condition A shows high spiral rate, relief pathway for load-bearers is not the key mechanism.

### Results: PREDICTIONS INVERTED

| Condition | Description | Predicted | Observed | Result |
|-----------|-------------|-----------|----------|--------|
| **A** | Load-bearers protected | ~0% spirals | **10% spirals** | FALSIFIED |
| **B** | Load-bearers exposed | ~10% spirals | **0% spirals** | FALSIFIED |
| **C** | All protected | 0% spirals | 0% spirals | Supported |
| **D** | Entrainment baseline | ~10% spirals | 0% spirals | Inconclusive* |

*Entrainment baseline lower than expected from I001; likely due to coupling-bias variance in this sample.

**Recovery Times (mean ± SD):**
- **A:** 85 ± 118 ticks (median: 52)
- **B:** 34 ± 64 ticks (median: 8)
- **C:** 12 ± 27 ticks (median: 0)
- **D:** 103 ± 118 ticks (median: 80)

**Effect sizes:**
- A vs B: Cohen's d = 0.54 (medium) — protecting load-bearers *increases* recovery time
- B vs D: Cohen's d = -0.73 (medium-large) — partial coherence outperforms entrainment

### Mechanism: Phase Misalignment Cascade

**Why did protecting ONLY load-bearers cause spirals?**

The mixed coupling regime creates destabilization:

1. Load-bearers (high coupling-bias) escape toward preferred-heading
2. Non-load-bearers (low coupling-bias) remain in pure entrainment mode
3. Social field fragments between identity and entrainment attractors
4. Non-load-bearers oscillate between conflicting signals (the group is now split)
5. High-coupling agents feel this oscillation as cost → fatigue → spiral

**The paradox:** Giving the strongest agents an escape valve destabilizes the system *unless everyone has it*.

**Why did protecting ONLY non-load-bearers prevent spirals?**

- Load-bearers (high responsiveness) quickly realign to social field
- Non-load-bearers drift to identity but have weak influence on group heading
- Social field remains coherent (dominated by high-coupling agents following it)
- Low cost accumulation → no fatigue → no spiral

**Interpretation:** Letting weak-influence agents opt out is harmless. Letting strong-influence agents opt out is catastrophic.

### Implications

1. **Relief pathways must be universal:** Selective access for high-influence nodes creates phase misalignment that destabilizes the field
2. **Load-bearers need protection least:** High-coupling agents are adapted to social constraint; escape options create dissonance with their structural role
3. **Design principle:** Avoid partial autonomy grants in heterogeneous-influence systems

### Confidence Assessment

**Level: MEDIUM**

Supporting evidence:
- Sample size adequate (N=50 per condition)
- Effect direction clear and large (10% vs 0%)
- Effect sizes medium (Cohen's d ~ 0.5-0.7)
- Mechanistic story coherent with I002 findings

Limitations:
- Entrainment baseline discrepancy (0% observed vs 10% expected)
- Single perturbation regime — may not generalize
- Non-normal distributions preclude parametric significance tests

---

## Mechanism Synthesis

### Unified Causal Frame (Revised after C001)

**Load Concentration + Absence of *Universal* Relief Pathways → Cascade Risk**

```
Agent configuration
├── Some agents are load-bearing (high coupling + far from consensus)
├── Load-bearing agents bear disproportionate stabilization work (1.6–2×)
│
├── If MANY load-bearers: burden distributed → each survives
├── If FEW load-bearers: each overloaded → fatigue accumulates
│
├── [ENTRAINMENT] No relief pathway → exhaustion → cascade failure
├── [COHERENCE]   Universal relief pathway → recovery → stability
│
└── [SELECTIVE ESCAPE - C001 finding]
    ├── Load-bearers-only protected → INCREASES cascade risk (10% spirals)
    │   └── Mechanism: Phase misalignment cascade
    │       └── High-influence agents exit → field fragments → oscillation → cost
    └── Non-load-bearers-only protected → PREVENTS cascade (0% spirals)
        └── Mechanism: Weak agents exit harmlessly; strong agents stabilize field
```

**Key insight:** Relief pathways must be universal. Selective access for high-influence agents creates phase misalignment that destabilizes the entire field. The "protection" of key nodes externalizes collapse risk onto those who remain coupled.

### Levels of Causality

| Level | Variable | Role |
|-------|----------|------|
| Population | Coupling-bias variance | Proxy signal (correlates with configuration risk) |
| Configuration | Who bears load, how many | Determines concentration risk |
| Agent | Relief pathway availability | Determines whether concentration cascades |

Population-level variance does not cause spirals. It correlates with configurations more likely to produce load concentration. The causal mechanism operates at the agent level.

---

## Theoretical Resonance

The following theoretical connections illuminate the findings without claiming equivalence:

### Ashby: Requisite Variety

Entrainment suppresses internal variance to achieve order. Under stress, this reduces adaptive capacity — the system lacks degrees of freedom for reorganization. Coherence preserves variance, maintaining requisite variety for adaptation.

### Allostasis vs Homeostasis

- **Allostasis (coherence):** Stability through continuous reorganization; variable patterns enable stress absorption
- **Homeostasis (entrainment):** Stability through set-point maintenance; rigid patterns become brittle under sustained stress

### Resilience Engineering: Graceful Extensibility

Coherence exhibits graceful extensibility — the ability to extend adaptive capacity as stress increases. Entrainment optimizes for efficiency at baseline but lacks extensibility under duress.

### Design Invariants

> **Original (I004):** Systems that rely on continuous synchronization must provide independent recovery pathways for high-load agents, or they will externalize collapse risk onto a shrinking subset of stabilizers.

> **Revised (C001):** Recovery pathways must be *universally* available. Selective escape valves for high-influence agents do not protect the system — they destabilize it by creating phase misalignment between those who exit and those who remain coupled.

> **Corollary:** In heterogeneous-influence systems, differential autonomy grants load collapse risk onto the less-autonomous. The "protected" nodes' exit is the mechanism of cascade, not its prevention.

This is the actionable synthesis: the mechanism identified here is not merely descriptive but suggests a design constraint for systems that must absorb stress without cascading. The C001 finding adds a crucial refinement: partial solutions that protect only key nodes may be worse than no solution at all.

---

## Metric Interpretation Notes

### Limitations of Global Averages

Mean values can mask critical dynamics:
- **Mean cost** obscures burden concentration on load-bearers
- **Mean fatigue** obscures the few agents driving cascade risk
- **Global variance** obscures local coherence/fragmentation

For cascade risk assessment, prioritize:
- Max-agent cost, max-agent fatigue
- Top-quantile work share (e.g., top 10% share of total alignment work)
- Recovery lag distributions (not just means)

### Future Metric Development

Planned extensions:
- **Cluster persistence:** Do stable subgroups form and persist?
- **Local variance:** Disagreement within neighborhoods vs global
- **Neighbor agreement stability:** Temporal autocorrelation of local alignment

---

## Priority Next Experiments

### C002: Coupling-Bias Variance Sweep in Selective Conditions

**Purpose:** Investigate why entrainment baseline in C001 showed 0% spirals (vs 10% expected). Confirm whether coupling-bias variance interacts with selective escape.

**Design:** Sweep target-coupling-bias-variance (0.06, 0.08, 0.10) × selective-identity-pull conditions

### C003: Graded Identity-Pull Access

**Purpose:** Test whether there is a threshold fraction of agents needing access to prevent cascade.

**Design:** Grant identity-pull to random subsets: 0%, 25%, 50%, 75%, 100% of agents

### T001/T002: Agent-Level Temporal Analysis

**Purpose:** Identify early-warning indicators before cascade onset.

**Design:** Track per-agent time series with high temporal resolution:
- Alignment work increments per tick
- Fatigue trajectory
- Heading autocorrelation (persistence)

**Indicators to test:**
- Rising autocorrelation in turning behavior
- Skew in cost increment distribution
- Top-quantile work share acceleration
- Divergence between load-bearing and non-load-bearing agent trajectories

**Goal:** Identify a "point of no return" in fatigue accumulation, enabling early intervention.

---

## Experimental Program Overview

| Code | Name | Design | Key Finding |
|------|------|--------|-------------|
| **E003** | Stress scaling | 2×4 factorial | Stress transition region; 12× recovery ratio |
| **E004a** | Coupling × perturbation | 2×3×6 factorial | Higher coupling → earlier transition |
| **E004b** | Identity-pull × perturbation | 4×5 factorial | Stronger identity-pull → higher stress tolerance |
| **E005a** | Cost mode contrast | 2×4 factorial, single | Coherence ~1.8× higher Gini |
| **E005b** | Cost distribution | 2×2 factorial, periodic | Entrainment 30× higher recovery cost |
| **F001** | Fatigue validation | 2×2 factorial | Governor vs spiral mechanism |
| **S001a-rep** | Scale sensitivity | 2×3×5 factorial (30 runs) | Coherence stable; entrainment stochastic |
| **I001** | Initial variance | 60 runs | NULL — initial variance ≠ spiral trigger |
| **I002** | Parameter heterogeneity | 60 runs | Coupling-bias variance as proxy (p=0.011) |
| **I003** | Controlled variance | 300 runs | NULL — no dose-response |
| **I004** | Configuration tracking | 100 runs | Load concentration mechanism identified |
| **C001** | Selective escape valve | 4 conditions × 50 runs | **FALSIFIED:** Protecting load-bearers increases spirals; universal access required |

### Statistical Notes

- Multiple hypothesis tests were conducted for **mechanism discovery**, not confirmatory inference
- Effect sizes and qualitative regime differences are primary evidence
- p-values are supporting evidence; no multiple comparison correction applied
- Future work should use logistic regression for binary outcomes (spiral/no-spiral) and survival analysis for recovery time

Because the objective of this experimental program is mechanism discovery rather than parameter estimation, we prioritise effect size, regime separation, and causal plausibility over hypothesis confirmation. Where inferential statistics are reported, they serve as signals for further investigation rather than claims of population inference.

---

## Experiment Registry

### Completed (13 experiments, ~800 runs)

| Code | Exports |
|------|---------|
| E002 | `coherence_model_simple_E002-spreadsheet.csv` |
| E003 | `coherence_model_simple_E003-spreadsheet.csv` |
| E004a | `E004a_coupling_x_perturbation-spreadsheet.csv` |
| E004b | `E004b_identity_pull_x_perturbation-spreadsheet.csv` |
| E005a | `E005a_cost_mode_contrast-spreadsheet.csv` |
| E005b | `E005b_cost_distribution-spreadsheet.csv` |
| F001 | `F001_fatigue_mechanism_validation-spreadsheet.csv` |
| S001a-rep | `S001a_rep_scale_sensitivity_5000-spreadsheet.csv` |
| I001 | `I001_spiral_trigger_investigation-spreadsheet.csv` |
| I002 | `I002_agent_parameter_heterogeneity-spreadsheet.csv` |
| I003 | `I003_controlled_variance_sweep-spreadsheet.csv` |
| I004 | `I004_configuration_tracking-spreadsheet.csv` |
| **C001** | `coherence_model_simple C001_selective_escape_valve-spreadsheet.csv` |

### Ready (Not Yet Run)

| Code | Design |
|------|--------|
| E004c | Social-pull × perturbation (4×5 factorial) |
| E004d | Identity × social interaction (3×3×3 factorial) |
| S001b-rep | Scale sensitivity 10000 ticks × 5 reps |

### Proposed (Awaiting Design)

| Code | Purpose |
|------|---------|
| **C002** | Coupling-bias variance × selective conditions (baseline investigation) |
| **C003** | Graded identity-pull access (threshold detection) |
| **T001** | Agent-level time series (priority early-warning) |
| T002 | Recovery trajectory analysis |
| A002 | Load-bearing threshold sensitivity sweep |
| F002 | Fatigue threshold sweep |

---

## Visualizations

Key figures in `exports/`:

- **E003:** `E003_theorem_validation.png`, `E003_stress_scaling.png`
- **E005:** `E005a_cost_comparison.png`, `E005b_cost_comparison.png`
- **F001:** `F001_fatigue_dynamics.png`, `F001_key_metrics.png`
- **I001–I004:** `I00*_summary_dashboard.png`, hypothesis testing and correlation figures
- **C001:** `c001_analysis_2026-01-11.png`, `c001_summary_2026-01-11.csv`

---

## Lineage

**Conceptual:** Varela (mutual constraint), Bateson (ecology of mind), Ashby (requisite variety), resilience engineering (graceful extensibility)

**Technical:** NetLogo 7.0.3, BehaviorSpace, Python/Jupyter analysis

---

**Summary principles**

> *"The spiral isn't about who's at risk. It's about whether they have anywhere to go when they're spent."*

> *"Escape valves must be universal. When the strongest agents exit while others remain coupled, their 'protection' becomes the mechanism of collapse."* (C001)

---

**Last updated:** 2026-01-11
**Session name:** Kairos
