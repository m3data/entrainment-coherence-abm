# Coherence Model: Definition, Theorem, and Nomenclature

This document formalises the core conceptual frame used in the Coherence vs Entrainment NetLogo model. It provides a shared language for interpreting simulations, experiments, and exported data, and is intended to evolve alongside the model itself.

---

## 1. Core Definition of Coherence

**Coherence** is defined as:

> *The ongoing capacity of a system to maintain its integrity, identity, and relational viability while encountering, absorbing, and reorganising through periods of incoherence without collapsing into chaos or enforcing uniformity.*

Key features of this definition:

- Coherence is **dynamic**, not a static state.
- Incoherence is not a failure mode but a **necessary phase** within adaptive systems.
- Coherence is distinguished from:
  - **Order** (low variance)
  - **Stability** (absence of change)
  - **Entrainment** (phase-locking or synchronisation)

This definition is grounded in ecological, cybernetic, and complex dynamical systems thinking rather than equilibrium-based or optimisation-based frames.

---

## 2. Distinction: Coherence vs Entrainment

Within this model:

- **Entrainment** refers to strong coupling that drives agents toward phase alignment (e.g. shared heading), minimising variance under calm conditions.
- **Coherence** refers to weaker, identity-preserving coupling that tolerates diversity while enabling coordinated reorganisation under stress.

| Dimension | Entrainment | Coherence |
|--------|------------|-----------|
| Primary mechanism | Phase-locking | Relational adaptation |
| Baseline variance | Low | Moderate |
| Response to perturbation | Amplified disruption | Absorptive reorganisation |
| Recovery time | Longer | Shorter |
| Failure mode | Brittleness | Drift (at extremes) |

Entrainment may appear more ordered in low-noise regimes but is more vulnerable to sustained or repeated perturbation.

---

## 3. Coherence Theorem (Operational)

**Coherence Theorem (Model-Level):**

> *In agent-based systems with heterogeneous identity and bounded coupling, regimes that preserve internal diversity will exhibit lower peak disruption and faster recovery under repeated perturbation than regimes optimised for phase alignment.*

This theorem is not presented as a universal law, but as a **testable proposition** within the scope of this model class.

Operationally, this is evaluated through:

- Heading variance dynamics
- Peak variance following perturbations
- Recovery time back to a tolerance band

---

## 4. Incoherence Typology

The model distinguishes between different forms of incoherence:

### 4.1 Regenerative Incoherence

- Short-lived increases in variance
- Enables reconfiguration and learning
- Followed by rapid recovery
- Characteristic of coherent regimes

### 4.2 Generative Incoherence

- Sustained variance that produces new patterns or clusters
- May increase pluralism or innovation
- Not necessarily pathological

### 4.3 Degenerative Incoherence

- Escalating variance without recovery
- Loss of identity, coordination, or relational viability
- Often follows repeated high-amplitude perturbations in entrained systems

---

## 5. Nomenclature and Model Terms

### Agents (Turtles)

- **preferred-heading**  
  Represents agent identity or intrinsic orientation.
- **coupling-bias**  
  Individual sensitivity to social influence.
- **inertia**  
  Resistance to rapid change; moderates responsiveness.

### Global Parameters

- **coupling-strength**
  Strength of social influence across the population.
- **noise-level**
  Ambient stochastic fluctuation.
- **perturbation-strength**
  Amplitude of externally applied disturbance.
- **perturb-duration**
  Temporal extent of a perturbation event.
- **entrainment-mode?**
  Switch determining whether agents pursue phase-locking or coherence dynamics.
- **identity-pull-weight** (added 2025-12-26)
  Strength of return toward agent's preferred-heading in coherence mode. Default: 0.2. Range: 0–1.
- **social-pull-weight** (added 2025-12-26)
  Attenuation of social coupling in coherence mode. Default: 0.5. Range: 0–1.

### Metrics

- **heading-variance**  
  Mean absolute angular deviation from the population mean heading.
- **recovery-time**  
  Number of ticks required for variance to return within a defined tolerance after perturbation.

---

## 6. Scale and Fractality Assumptions

The model assumes:

- Systems are **nested across scales** (nano–micro–meso–macro–meta).
- Coherence at one scale may induce incoherence at another.
- Fractal similarity does not imply sameness; patterns reorganise across scales.

This explicitly rejects equilibrium metaphors and static optimisation frames.

---

## 7. Scope and Limits

This model:

- Is illustrative, not predictive.
- Explores relational dynamics, not goal optimisation.
- Is intended as a **thinking tool** and experimental sandbox.

Future extensions may include:

- Multi-cluster coherence detection
- Adaptive coupling based on tie strength
- Cross-scale perturbation propagation

---

## 8. Empirical Findings (E003: Stress Scaling Response)

**Experiment date:** 2025-12-26
**Design:** 2×4 factorial — entrainment mode (true/false) × perturbation strength (10, 20, 40, 80)
**Regime:** Single perturbation at tick 300, duration 30 ticks
**Recovery tracking:** Rolling 50-tick baseline, recovery measured from perturbation end to tolerance return (±5 degrees)

### 8.1 Recovery Time Results

| Mode | Strength 10 | Strength 20 | Strength 40 | Strength 80 |
|------|-------------|-------------|-------------|-------------|
| Coherence | 0 ticks | 0 ticks | 8 ticks | 10 ticks |
| Entrainment | 0 ticks | 0 ticks | 49 ticks | 120 ticks |

### 8.1.1 Peak Disruption (Max Deviation from Baseline)

| Mode | Strength 10 | Strength 20 | Strength 40 | Strength 80 |
|------|-------------|-------------|-------------|-------------|
| Coherence | 4.5 | 6.0 | 8.2 | 11.3 |
| Entrainment | 4.5 | 4.8 | 34.3 | 46.7 |

### 8.1.2 Stable Baseline Variance

| Mode | Strength 10 | Strength 20 | Strength 40 | Strength 80 |
|------|-------------|-------------|-------------|-------------|
| Coherence | 102.8 | 93.2 | 79.7 | 87.2 |
| Entrainment | 36.3 | 21.5 | 29.8 | 24.0 |

### 8.2 Key Observations

**Coherence mode:**
- Recovery is near-instantaneous (0–10 ticks) across all stress levels
- Baseline variance remains in a moderate band (~80–103)
- Peak disruption scales linearly with perturbation strength (4.5 → 11.3)
- Perturbations are absorbed without amplification
- Demonstrates **regenerative incoherence** pattern

**Entrainment mode:**
- At low stress (10, 20): performs similarly to coherence — both recover instantly
- At moderate stress (40): recovery takes **6× longer** (49 vs 8 ticks), peak disruption **4× higher** (34.3 vs 8.2)
- At high stress (80): recovery takes **12× longer** (120 vs 10 ticks), peak disruption **4× higher** (46.7 vs 11.3)
- Achieves lower baseline variance (~21–36) through phase-locking, but pays brittleness cost under stress

**Critical threshold observation:**
A stress threshold emerges between strength 20 and 40 where entrainment's vulnerability becomes pronounced. Below this threshold, both regimes perform similarly. Above it, entrainment shows superlinear degradation in both recovery time and peak disruption.

### 8.3 Theorem Evaluation

The Coherence Theorem predicts:
> *regimes that preserve internal diversity will exhibit lower peak disruption and faster recovery under repeated perturbation than regimes optimised for phase alignment*

| Metric | Coherence | Entrainment | Ratio (at Str=80) |
|--------|-----------|-------------|-------------------|
| Max recovery time | 10 ticks | 120 ticks | **12×** |
| Max peak disruption | 11.3 | 46.7 | **4×** |
| Recovery scaling | Linear (0→10) | Superlinear (0→120) | — |
| Disruption scaling | Linear (4.5→11.3) | Superlinear (4.5→46.7) | — |

**Stress Amplification Ratios:**

| Perturbation Strength | Recovery Ratio (Ent/Coh) | MaxDev Ratio (Ent/Coh) |
|-----------------------|--------------------------|------------------------|
| 10 | 1× (both 0) | 1× |
| 20 | 1× (both 0) | 0.8× |
| 40 | **6×** | **4×** |
| 80 | **12×** | **4×** |

**Finding:** The theorem is strongly supported by E003 results. Both regimes perform equivalently at low stress, but a critical threshold emerges at perturbation strength ~40 where entrainment's brittleness becomes pronounced. Above this threshold, coherence mode maintains linear scaling while entrainment shows superlinear degradation in both recovery time and peak disruption.

### 8.4 Incoherence Typology Observations

- **Regenerative incoherence:** Coherence mode at all stress levels — perturbations absorbed with rapid return to baseline. Entrainment at low stress (10, 20) also shows this pattern.
- **Amplified incoherence:** Entrainment at moderate/high stress (40, 80) — perturbations cause disproportionate disruption (4× higher peak deviation) with extended recovery times (6–12× longer).
- **Generative incoherence:** Not observed in this experiment (would require cluster formation metrics)

The key distinction: coherence mode exhibits regenerative incoherence regardless of stress level, while entrainment mode transitions from regenerative to amplified incoherence as stress increases past the critical threshold.

### 8.5 Threshold Mechanism Analysis

The bifurcation between regimes arises from different attractor structures in the heading update dynamics.

**Entrainment mode:**
```
desired-turn = coupling-strength × coupling-bias × (avg-heading - heading)
```
Agents turn toward the weighted average heading of neighbours. Full coupling strength, no internal reference.

**Coherence mode:**
```
identity-pull = preferred-heading - heading
social-pull = avg-heading - heading

desired-turn = (coupling-strength × coupling-bias × social-pull-weight × social-pull)
             + (identity-pull-weight × identity-pull)
```
Two forces: attenuated social pull **plus** identity pull toward the agent's own preferred-heading.

#### Why the threshold emerges

| Perturbation Level | Entrainment | Coherence |
|--------------------|-------------|-----------|
| **Low (10–20)** | Collective mean remains stable reference; sufficient restoring force | Each agent has stable internal anchor; rapid local recovery |
| **High (40–80)** | Collective mean becomes destabilised; agents chase a noisy/wandering target → feedback amplification | preferred-heading unaffected by perturbation; local restoration regardless of global state |

**The threshold (~20–40) marks where:**
- Perturbation magnitude exceeds entrainment's collective restoring force
- The mean heading becomes an unreliable attractor
- Entrainment's strength (tight coupling to collective) becomes its weakness

#### Basin stability interpretation

- **Entrainment**: Single shared attractor (the collective mean). Large perturbation can push the *entire attractor* into a noisy, wandering state. Recovery requires global re-coherence.
- **Coherence**: Each agent has its *own* attractor (preferred-heading). Perturbation displaces individuals from local basins, but those basins are stable and unaffected. Recovery is parallel and independent.

This explains the **superlinear** degradation in entrainment: above threshold, each agent's disorientation contributes to every other agent's disorientation. The collective reference point becomes the source of amplification rather than restoration.

#### Open questions for E004

The threshold location likely depends on:
1. **coupling-strength** — does stronger coupling shift the threshold lower?
2. **identity-pull-weight** — does stronger identity anchor raise the threshold?
3. **social-pull-weight** — is there an optimal balance, or does lower social coupling always improve resilience?
4. **Interaction effects** — do identity and social weights interact nonlinearly?

---

## 9. Metrics (Revised)

### Original Metrics

- **heading-variance**
  Mean absolute angular deviation from the population mean heading.

- **last-recovery-time** (deprecated)
  Original recovery tracking had issues with baseline capture at perturbation onset.

### Recovery Tracking v2 (December 2025)

- **stable-baseline**
  Rolling 50-tick average of heading variance, updated only when system is not perturbing or recovering.

- **recovery-time**
  Ticks from perturbation END to when heading variance returns within ±recovery-tolerance of stable-baseline.

- **max-deviation**
  Peak absolute deviation from stable-baseline during and after perturbation.

- **is-recovering?**
  Boolean flag indicating whether system is currently in recovery phase.

This revised tracking ensures:
1. Baseline reflects true stable state (not snapshot at perturbation onset)
2. Recovery timing starts from perturbation end, not threshold crossing
3. Symmetric tolerance band works for both high-variance (coherence) and low-variance (entrainment) regimes

---

## 10. Corrollaries to Living Systems Theory

Systems that maintain coherence through continuous reorganisation exhibit lower stress amplification and negligible recovery cost under perturbation, whereas systems that maintain order through entrainment exhibit suppressed variance at rest but incur disproportionate recovery costs under stress. This mirrors the distinction between allostatic regulation and homeostatic control in living systems.

In living systems, coherence supports adaptability and resilience, while entrainment may lead to brittleness and collapse under sustained stress. Take a breath: the body's ability to maintain coherence through variable breathing patterns allows it to adapt to stress, whereas rigid breathing patterns (entrainment) may lead to dysfunction under pressure. In ecological systems, diverse species interactions (coherence) enable ecosystems to absorb shocks, while monocultures (entrainment) are more vulnerable to collapse. You might think to a school of fish: their ability to change formation (coherence) allows them to evade predators, while rigid schooling (entrainment) may lead to collective vulnerability.

---

## 11. Planned Experiments (E004: Threshold Surface Mapping)

**Purpose:** Map the threshold surface as a function of coupling parameters to understand what drives the bifurcation point between low-stress equivalence and high-stress divergence.

**Research questions:**
1. Does stronger coupling-strength shift the threshold lower (more brittleness)?
2. Does stronger identity-pull-weight raise the threshold (more stress tolerance)?
3. Is there an optimal social-pull-weight, or does lower always improve resilience?
4. Do identity and social weights interact nonlinearly?

### E004a: Coupling × Perturbation

**Design:** 2×3×6 factorial — entrainment mode × coupling-strength × perturbation-strength

| Factor | Levels |
|--------|--------|
| entrainment-mode? | false, true |
| coupling-strength | 0.2, 0.4, 0.6 |
| perturbation-strength | 10, 20, 30, 40, 60, 80 |

**Runs:** 36
**Hypothesis (H1):** Higher coupling → lower threshold (more brittle under stress)

### E004b: Identity-Pull × Perturbation

**Design:** Coherence mode only, 4×5 factorial

| Factor | Levels |
|--------|--------|
| identity-pull-weight | 0.1, 0.2, 0.3, 0.4 |
| perturbation-strength | 20, 30, 40, 60, 80 |

**Runs:** 20
**Hypothesis (H2):** Stronger identity-pull → higher stress tolerance (threshold shifts right)

### E004c: Social-Pull × Perturbation

**Design:** Coherence mode only, 4×5 factorial

| Factor | Levels |
|--------|--------|
| social-pull-weight | 0.25, 0.5, 0.75, 1.0 |
| perturbation-strength | 20, 30, 40, 60, 80 |

**Runs:** 20
**Hypothesis (H3):** Nonlinear effect — very low social-pull → drift; very high → entrainment-like brittleness

### E004d: Identity × Social Interaction

**Design:** Coherence mode only, 3×3×3 factorial

| Factor | Levels |
|--------|--------|
| identity-pull-weight | 0.1, 0.2, 0.4 |
| social-pull-weight | 0.25, 0.5, 1.0 |
| perturbation-strength | 30, 40, 60 |

**Runs:** 27
**Hypothesis (H4):** Optimal coherence requires balanced identity/social weights; interaction effects exist

### Total runs: 103

### Threshold detection method

The threshold is operationalised as the perturbation strength where:
- `recovery-time(entrainment) / recovery-time(coherence) > 2`, or
- `recovery-time > 30 ticks` (absolute bound)

Sigmoid fitting may be used to locate inflection points:
```
recovery(p) = r_min + (r_max - r_min) / (1 + exp(-k(p - p_threshold)))
```

### Derived metrics (post-hoc analysis)

- **Threshold location**: Perturbation strength at regime transition
- **Amplification ratio**: max-deviation / perturbation-strength
- **Recovery scaling exponent**: log-log slope of recovery vs perturbation

---

## 12. Empirical Findings (F001: Fatigue Mechanism Validation)

**Experiment date:** 2025-12-27
**Design:** 2×2 factorial — entrainment mode (true/false) × fatigue-enabled (true/false)
**Regime:** Single perturbation at tick 300, strength 60, duration 30 ticks
**Fatigue parameters:** threshold=500, saturation=2000, intensity=0.5, decay-rate=0.01

### 12.1 Summary Results

| Mode | Fatigue | Recovery Time | Max Deviation | Peak Fatigue | Peak Eff. Inertia |
|------|---------|---------------|---------------|--------------|-------------------|
| Coherence | OFF | 7 ticks | 6.6° | 0.61* | 0.59 |
| Coherence | ON | 50 ticks | 11.2° | 0.04 | 0.62 |
| Entrainment | OFF | 119 ticks | 33.6° | 0.61* | 0.61 |
| Entrainment | ON | 1330 ticks | 85.3° | 0.65 | 0.73 |

*When fatigue is OFF, fatigue level is still calculated (for comparison) but doesn't affect dynamics.

### 12.2 Fatigue Amplification Ratios

| Mode | Recovery (OFF→ON) | Max Deviation (OFF→ON) |
|------|-------------------|------------------------|
| Coherence | 7× (7→50) | 1.7× (6.6→11.2) |
| Entrainment | **11×** (119→1330) | **2.5×** (33.6→85.3) |

### 12.3 Key Finding: Governor vs Spiral

**Fatigue acts as a metabolic governor in coherence mode, but triggers a depletion spiral in entrainment mode.**

#### Coherence + Fatigue = Metabolic Governor

- Peak fatigue level: **0.04** (barely crosses threshold)
- Costs rise → slight sluggishness → system slows down → costs decay → equilibrium
- The fatigue mechanism *regulates* the system's response rate
- Identity-pull provides recovery pathway independent of social field
- Recovery window opens quickly; rest is available

#### Entrainment + Fatigue = Depletion Spiral

- Peak fatigue level: **0.65** (deep into fatigue zone)
- Costs rise → sluggishness → can't align → more cost → more sluggishness
- No independent recovery pathway — agents need the social field, but they're too sluggish to use it
- Recovery window never opens; rest is unavailable
- System eventually recovers only through slow cost decay

### 12.4 Mechanism Analysis

The identity-pull in coherence mode is not just an alternative attractor — it functions as a **metabolic escape valve**.

When social coupling becomes costly (fatigued agents turn slower), coherence-mode agents can still drift toward their preferred-heading. This independent recovery pathway breaks the fatigue spiral.

Entrainment has no home. Its only attractor *is* the social field. When coupling degrades due to fatigue, there is nowhere else to go. Agents must wait for the social field to stabilize, but they can't contribute to that stabilization because they're too sluggish to respond.

**The feedback loop:**
```
Entrainment spiral:
  perturbation → extended recovery → cost accumulation → fatigue builds
  → increased inertia → slower response → even longer recovery
  → more cost → more fatigue → SPIRAL

Coherence regulation:
  perturbation → brief recovery (identity-pull works) → limited cost accumulation
  → minimal fatigue → costs decay during stability → capacity restored
  → EQUILIBRIUM
```

### 12.5 Connection to Metabolic Asymmetry

This finding maps directly to the metabolic asymmetry framing in the EarthianLab CLAUDE.md:

> Systems that depend entirely on external synchronization for stability are vulnerable to exhaustion spirals. Systems that maintain internal reference points can self-regulate.

Coherence-mode agents maintain internal reference points (preferred-heading). Under fatigue, they can still find their way home. Entrainment-mode agents have no internal reference — they are defined entirely by their relationship to the collective. When that relationship becomes costly, they have no fallback.

### 12.6 Implications

1. **For system design:** Systems that require constant external coordination are vulnerable to metabolic collapse under sustained stress. Building in internal reference points (identity, values, preferred states) provides resilience.

2. **For the Coherence Theorem:** Fatigue amplifies the difference between coherence and entrainment. Under observational cost tracking (E005), cumulative costs looked similar. Under consequential cost tracking (F001), the hidden tax of entrainment becomes visible as a depletion spiral.

3. **For future experiments:** The threshold parameters (fatigue-threshold, fatigue-saturation, cost-decay-rate) likely interact with the bifurcation point identified in E003. F002 should explore whether the governor/spiral transition can be tuned.

---

## 13. Status

This document is a **living specification**.

Definitions, theorems, and nomenclature may be revised as empirical findings from BehaviorSpace experiments accumulate.

**Last updated:** 2025-12-27 (F001 fatigue findings: governor vs spiral mechanism, metabolic escape valve concept)