# TENCON 2026 — Experiment Protocol

**Paper:** Coherence vs Entrainment in Human–AI Agentic Systems
**Venue:** IEEE TENCON 2026 (Bali, 10-13 Oct)
**Deadline:** 17 March 2026
**Model:** `netlogo/coherence_model_tencon.nlogox`

---

## Research Question

Under what conditions do human–AI agentic systems remain resilient and adaptive, and when do they collapse into brittle entrainment regimes?

## Core Argument

AI agents — with faster update rates, lower noise, larger influence radii, and amplified coordination signals — act as **entrainment accelerators**. Even in systems designed for coherence, increasing AI proportion or reach can tip the whole system toward brittle dynamics.

This connects to existing findings:
- **C001:** Differential agent properties → cascade (selective escape destabilizes)
- **E003:** Stress amplification above threshold (12× recovery ratio)
- **I002:** Heterogeneous coupling → spiral trigger (p=0.011)

---

## Shared Constants

All experiments use these baseline values (aligned with existing model).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `population` | 100 | Standard |
| `coupling-strength` | 0.4 | Standard |
| `noise-level` | 0.35 | Standard |
| `perturbation-strength` | 60 | Above E003 threshold (20-40) |
| `perturb-duration` | 144 | Standard |
| `recovery-tolerance` | 5 | Standard |
| `identity-pull-weight` | 0.2 | Standard coherence mode |
| `social-pull-weight` | 0.5 | Standard coherence mode |
| `fatigue-enabled?` | true | Needed for spiral detection |
| `fatigue-threshold` | 500 | Standard |
| `fatigue-saturation` | 2000 | Standard |
| `fatigue-intensity` | 0.5 | Standard |
| `cost-decay-rate` | 0.01 | Standard |
| `metabolic-rate` | 0.1 | Standard |
| `shock-multiplier` | 1.0 | Standard |
| `recovery-rate` | 0.05 | Standard |
| `controlled-variance?` | false | Random coupling-bias |
| `selective-identity-pull` | "all" | Universal access (C001 design invariant) |
| `human-update-rate` | 5 | Humans 5× slower than AI |
| `ai-influence-radius` | 8 | ~2.7× human reach (human = 3) |
| `ai-tie-strength-multiplier` | 1.5 | AI amplifies coordination signals |
| `ai-noise-multiplier` | 0.3 | AI is algorithmically precise |
| `human-noise-multiplier` | 1.0 | Human biological baseline |
| Ticks | 3000 | Enough for periodic stress + recovery |

**Design choice:** Global mode only (all agents share entrainment-mode? setting). No per-agent mode switching — cleaner for 3-5 page paper.

---

## Experiments

### H001: AI Proportion Sweep (core result)

**Question:** At what proportion of AI agents does the system tip from resilient to brittle?

**Design:** 2-factor (ai-proportion × mode), single perturbation

| Factor | Values |
|--------|--------|
| `ai-proportion` | 0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9 |
| `entrainment-mode?` | true, false |
| `perturbation-regime` | "single" |
| Repetitions | 30 |
| `runMetricsEveryStep` | false (final-state only) |
| **Total runs** | **7 × 2 × 30 = 420** |

**Metrics captured:**
- Recovery: `recovery-time`, `max-deviation`
- Variance: `heading-variance`, `diversity-index`, `human-heading-variance`, `ai-heading-variance`, `human-diversity-index`, `ai-diversity-index`
- Cost: `mean-cumulative-cost`, `human-mean-cost`, `ai-mean-cost`, `human-ai-cost-ratio`, `cost-gini`
- Work: `mean-alignment-work`, `human-mean-alignment-work`, `ai-mean-alignment-work`, `human-ai-work-ratio`
- Fatigue: `max-fatigue-level`, `human-max-fatigue`, `ai-max-fatigue`, `mean-fatigue-level`, `human-mean-fatigue`, `ai-mean-fatigue`, `agents-fatigued`
- Validation: `initial-coupling-bias-variance`, `num-humans`, `num-ais`

**Expected results:**
- ai-proportion = 0 matches base model (backward compatibility validation)
- Recovery time increases with AI proportion (especially entrainment mode)
- Diversity index drops with higher AI proportion
- human-ai-cost-ratio rises — humans bear disproportionate burden
- Coherence mode shows more resilience across all proportions
- Possible tipping point in the 0.3-0.5 range

---

### H002: Repeated Stress Response

**Question:** Do mixed human-AI systems fail differently under periodic perturbation?

**Design:** 2-factor (ai-proportion × mode), periodic perturbation

| Factor | Values |
|--------|--------|
| `ai-proportion` | 0, 0.2, 0.5 |
| `entrainment-mode?` | true, false |
| `perturbation-regime` | "periodic" |
| Repetitions | 30 |
| `runMetricsEveryStep` | false |
| **Total runs** | **3 × 2 × 30 = 180** |

**Metrics:** Same as H001 + time-series awareness (may switch to `runMetricsEveryStep = true` for select conditions to capture progressive degradation).

**Expected results:**
- Entrainment + high AI proportion shows compounding failure under repeated stress
- Coherence mode maintains stability even with periodic perturbation
- Fatigue spirals emerge faster with AI proportion (AI agents accelerate cost concentration)
- Extends E005b finding (30× recovery cost under periodic stress) to mixed systems

---

### H003: Influence Radius Sweep

**Question:** Is AI reach or AI proportion the stronger driver of regime shift?

**Design:** 2-factor (ai-influence-radius × mode), fixed proportion

| Factor | Values |
|--------|--------|
| `ai-influence-radius` | 3, 5, 8, 12 |
| `ai-proportion` | 0.2 (fixed) |
| `entrainment-mode?` | true, false |
| `perturbation-regime` | "single" |
| Repetitions | 30 |
| `runMetricsEveryStep` | false |
| **Total runs** | **4 × 2 × 30 = 240** |

**Metrics:** Same as H001.

**Expected results:**
- Influence radius is a stronger driver than proportion alone
- At radius = 3 (same as human), AI agents are indistinguishable from fast humans
- At radius = 12, even 20% AI agents can dominate system dynamics
- Isolates connectivity from speed/precision effects

---

## Batching Strategy

Pilot before scale. Review results between batches.

| Batch | Experiment | Design | Runs | Purpose |
|-------|-----------|--------|------|---------|
| **Batch 1** | H001 exploratory | 4 proportions (0, 0.2, 0.5, 0.9) × 2 modes × 10 reps | **80** | Confirm signal exists, validate fork |
| **Batch 2** | H001 full | 7 proportions × 2 modes × 30 reps | **420** | Publication-quality data |
| **Batch 3** | H002 | 3 proportions × 2 modes × 30 reps, periodic | **180** | Repeated stress story |
| **Batch 4** | H003 | 4 radii × 2 modes × 30 reps | **240** | Mechanism isolation (if needed) |

**Total if all batches run:** 840 runs (realistically may stop at Batch 2 or 3 for a 3-5 page paper).

**Batch 1 is embedded** in the model file as BehaviorSpace experiment `H001_batch1_proportion_sweep_exploratory`. Batches 2-4 will be written after Batch 1 analysis.

---

## Paper Figure Plan

Target: 3-5 pages IEEE double-column, ~4 figures.

| Figure | Source | Content |
|--------|--------|---------|
| **Fig 1** | Model description | Schematic: human vs AI agent properties (update rate, noise, radius, influence) |
| **Fig 2** | H001 | Recovery time vs AI proportion — two curves (entrainment/coherence). The tipping point. |
| **Fig 3** | H001 | Dual panel: (a) diversity collapse vs proportion, (b) cost asymmetry (human-ai-cost-ratio) vs proportion |
| **Fig 4** | H002 | Time series: heading variance under periodic stress at 0%, 20%, 50% AI. Shows progressive degradation. |

**Optional Fig 5** (if space): H003 — recovery time vs influence radius. Mechanism isolation.

---

## Analysis Plan

Export CSVs to `exports/` with naming: `H001_batch1_YYYYMMDD.csv`, etc.

Analysis in Python (Jupyter notebook or script in `notebooks/`):
- Load with pandas
- Group by (ai-proportion, entrainment-mode?)
- Summary statistics: mean, sd, median per condition
- Plots: matplotlib/seaborn
- Statistical tests: Mann-Whitney U or Kruskal-Wallis for between-condition comparisons
- Effect sizes: Cohen's d or rank-biserial correlation

---

## Contingencies

- **If Batch 1 shows no signal:** Check backward compatibility first (ai-proportion=0). If fork is broken, debug. If fork works but no proportion effect, increase perturbation-strength to 90 or increase ai-tie-strength-multiplier.
- **If results are noisy:** Increase repetitions in Batch 2 from 30 to 50.
- **If H002 time-series needed:** Re-run select conditions with `runMetricsEveryStep = true` (much larger CSV but captures dynamics).
- **If paper needs more figures:** H003 provides a clean mechanism isolation plot.
- **If time runs out:** H001 alone can carry a 3-page paper. H002 strengthens it to 4-5 pages.

---

*Created: 2026-02-07. Updated as experiments run.*
