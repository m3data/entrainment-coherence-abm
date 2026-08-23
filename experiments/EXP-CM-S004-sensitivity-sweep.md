---
id: EXP-CM-S004 / EXP-CM-S005
type: experiment
status: specified
scale: dyad
registered: 2026-06-28
completed: —
purpose: TENCON 2026 camera-ready — reviewer robustness requirement (R1, R2)
model: netlogo/coherence_model_tencon.nlogox
---

## ▶ Resume here (next session)

**Status (2026-07-03): OAT built + run at n=30, then reproducibility issue found → seeds pinned, re-running seeded at n=100.**

**What happened this session:**
- Built `S-H004` (AI-differential OAT, 4 sub-sweeps) + `S-H005` (fatigue OAT, 2 sub-sweeps) into the model, mirroring `H001_batch2`/`H003_batch4`. NetLogo 7 external `--setup-file` **requires the `<subExperiments>` wrapper** (flat enumeratedValueSets → `head of empty list`). Headless needs **JDK 17** (`brew install openjdk@17`; bundled runtime has no callable `java`; the `netlogo-headless.sh` launcher has a bracketed-arg bug → invoke `java` directly, see `run_seeded_deposit.sh`).
- First run (n=30) verdicts: **F1 diversity-collapse ROBUST**, **F3 threshold ROBUST**, **F2 inverted-U noise-fragile at n=30** (peak-AI wanders 0/0.1/0.2; pooled n=120 shows a modest ~7pp minority peak, inside the sampling band).
- **Reproducibility bug found:** BehaviorSpace seeds were never pinned. A fresh rerun of the *canonical* H001 gives 23% cascade at 0%-AI vs the deposited 10% — the deposit was one noisy n=30 draw. **Fix:** `setup` now calls `random-seed behaviorspace-run-number` (run-number 0 in GUI = unseeded). **Determinism proven** (two invocations byte-identical).
- **Decision (Mat):** bump sweeps to **n=100 + pin seeds + re-deposit**; report findings as **threshold bands**, not sharp point-claims.

**Running now (bg, ~75 min):** `run_seeded_deposit.sh` → re-runs H001/H002/H003 (seeded, published reps) + S-H004/S-H005 (seeded, n=100) → `exports/seeded/*-seeded.csv`.

**Next (after run):**
1. Rebuild `notebooks/validate_baseline.py` anchors from the **seeded** core deposit (old anchors were the unpinned draw).
2. Robustness table with **bands** (F1/F2/F3) + 1 faceted figure → new §Robustness subsection; wire `\ref{sec:robustness}` in the paper (B's forward-ref).
3. Update paper §5 numbers to seeded values (band framing); redeposit to Zenodo.
4. **Clean-context adversarial review before ship** (USDD).

Env ready: `.venv` (pinned deps). Run: `source .venv/bin/activate && MPLBACKEND=Agg python notebooks/<script>.py`. Backups: `netlogo/*.bak`, `publications/.../tencon2026.tex.bak`.

---

## Hypothesis

**HYP-CM-S004: The three headline findings of the TENCON paper are robust to the choice of the AI-differential parameters and the fatigue threshold — they reflect the entrainment/coherence structure, not a single tuning.**

The three findings under test:

1. **Diversity collapse under entrainment** — `diversity-index` falls monotonically with AI proportion in entrainment mode (0.29 → 0.07) but stays flat (~0.48) in coherence mode.
2. **Inverted-U cascade** — entrainment cascade-failure rate peaks at a *minority* AI proportion (10–20%), not a majority.
3. **Regime-bias threshold** — in mixed populations, cascade rate stays near zero below human regime-bias ≈ 0.25 and accelerates beyond it.

This is a **robustness claim, framed for refutation**: the experiment is designed to *break* the findings by moving the parameters the reviewers named. A finding that survives is reported as robust; a finding that flips or drifts is reported honestly as bounded.

## Addresses

- **Reviewer 1** — "Provide a thorough sensitivity analysis showing how key findings (cascade rates, diversity index, regime-bias threshold) vary with different agent parameter settings." Directly.
- **Reviewer 2** — "The fatigue mechanism... provide the mathematical definition, justify the parameter choices, and discuss the sensitivity of the results to these assumptions." The fatigue-threshold/intensity sweep (S005) is the sensitivity half of that answer; the math goes in §3.3 (plan item B).
- **RQ3** — How do coherence and entrainment shape durability/brittleness? A robustness result strengthens the claim that the brittleness is regime-structural, not parameter-contingent.
- **RQ2** — What conditions cultivate/constrain adaptive capacity? Locates *how wide* the identity-preservation safe-zone is across parameter space.

## Technical Uncertainty (R&DTI)

The published findings were established at a single point in parameter space (update-rate differential 5×, AI influence radius 8 vs human 3, AI noise multiplier 0.3, AI tie-strength ×1.5, fatigue-threshold 500). It is genuinely unknown whether:
- the inverted-U **peak location** is stable or migrates with the AI influence radius / speed differential;
- the diversity-collapse **direction** holds when the AI noise advantage shrinks (AI noise multiplier → 1.0 removes the AI's precision edge);
- the **0.25 threshold** is a fixed property or moves with the fatigue threshold (a higher threshold may delay cascade onset and shift the safe-zone right).

The direction of these cannot be predicted from the existing runs — that is the uncertainty the sweep resolves.

## Method

**Platform:** NetLogo BehaviorSpace on `coherence_model_tencon.nlogox`. Analysis via the existing `notebooks/h001_batch1_analysis.py` + `h003_analysis.py` pipeline (parsers already handle these metric columns).

**Design discipline: One-At-A-Time (OAT).** Each swept parameter moves across its levels while all others hold at the published defaults. This isolates each parameter's effect and keeps the run budget proportionate to a ~6pp conference paper. (Full factorial is the release valve *only* if the deadline and page budget allow — see Compute Budget.)

### Fixed baseline (published defaults — all sub-experiments)

| Parameter | Value |
|-----------|-------|
| population | 100 |
| coupling-strength | 0.4 |
| noise-level | 0.35 |
| perturbation-strength | 60 |
| perturb-duration | 144 |
| recovery-tolerance | 5 |
| identity-pull-weight | 0.2 |
| social-pull-weight | 0.5 |
| ai-regime-bias-mean | 0.8 |
| fatigue-saturation | 2000 |

### S-H004 — AI-differential parameter robustness (findings 1 & 2)

**Swept (OAT), each holding the other three at default:**

| Parameter | Default | Levels | Paper claim it tests |
|-----------|---------|--------|----------------------|
| `human-update-rate` | 5 | {2, 3, 5, 8} | AI 5× speed advantage (AI=1 tick; human slower) |
| `ai-influence-radius` | 8 | {5, 8, 11} | AI reach 8 vs human 3 |
| `ai-noise-multiplier` | 0.3 | {0.1, 0.3, 0.5, 1.0} | AI precision advantage (1.0 = no advantage) |
| `ai-tie-strength-multiplier` | 1.5 | {1.0, 1.5, 2.0} | AI influence amplification |

**Crossed with:** `ai-proportion` ∈ {0, 0.1, 0.2, 0.5, 0.9} × `coordination-regime` ∈ {entrainment, coherence}
**Reps:** 30 · **Perturbation:** single · **Exit:** ticks ≥ 3000
**Primary DVs:** `diversity-index` (finding 1), `recovery-time` → cascade rate (finding 2). Supporting: `cost-gini`, `max-fatigue-level`, `human-ai-cost-ratio`.

### S-H005 — Fatigue & threshold robustness (finding 3 + R2 fatigue sensitivity)

**Swept (OAT):**

| Parameter | Default | Levels |
|-----------|---------|--------|
| `fatigue-threshold` | 500 | {300, 500, 800} |
| `fatigue-intensity` | 0.5 | {0.3, 0.5, 0.7} |

**Crossed with:** `human-regime-bias-mean` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `ai-proportion` ∈ {0, 0.2} (the peak-risk zone)
**Mode:** mixed (per-agent regime-bias) · **Perturbation:** periodic (where the threshold is sharpest — H003 periodic 20% AI: 0% → 60%) · **Reps:** 30
**Primary DV:** cascade rate vs human regime-bias → does the ≈0.25 threshold hold/move. Supporting: `diversity-index`, `mean-fatigue-level`, `agents-fatigued`.

## Validation gate (Test-First)

**Before trusting any swept result, the analysis pipeline must reproduce the published baseline at default parameters.** Re-run the default cell of each sweep (the level that equals the published config) and confirm it matches the paper's numbers within Monte-Carlo error:
- entrainment diversity-index ≈ 0.29 (0% AI) → 0.07 (90% AI)
- entrainment cascade peak at 10–20% AI ≈ 27–30%
- coherence diversity ≈ 0.48, ~0 cascades through 50% AI
- H003 periodic threshold near bias 0.25

If the default re-run does **not** reproduce the published numbers, stop — the discrepancy is a pipeline/seed/model-version problem to resolve before any sensitivity claim. This gate doubles as the **reproducibility evidence for Reviewer E** (the stats regenerate from the deposited model + scripts).

### Gate result — PASSED 2026-06-28 (`notebooks/validate_baseline.py`, 13/13 anchors)

The deposited H001 (batch2, 420 runs) and H003 (batch4, 900 runs) exports reproduce the published headline numbers through the existing parsers — several exactly (entrainment cascade 10/30/27%, coherence 0% through 50% AI, periodic threshold 3%→67% and 0%→60%). **Cleared for the sensitivity sweeps.** The script is the standing reproducibility artifact for the response letter (Reviewer E).

**Two paper-text accuracy issues surfaced by the gate (for the moderation pass, plan item D):**
1. **§5.1 entrainment 0%-AI diversity stated as 0.29 — data say 0.22.** The 0.29 is the *10%-AI* value; §5.3 of the same paper already reports 0.22. Internal inconsistency → correct §5.1 to 0.22. Also "monotonic collapse" is slightly imprecise: diversity bumps up at 10% AI (0.22→0.29) before declining monotonically — reword to "declines from 10% AI onward" or "near-monotonic."
2. **H003 "cascade near zero across all AI proportions at bias 0.25" is generous under periodic stress** (reproduced: 0% / 17% / 23% at AI 0 / 0.2 / 0.5). Tighten to "at low AI proportions," or report the periodic-stress qualifier explicitly.

## Kill / decision criteria

Per finding, declare **robust** vs **bounded** vs **fragile**:

- **Finding 1 (diversity collapse):** robust if the entrainment-vs-coherence direction holds at every parameter level. Bounded if the *magnitude* attenuates (e.g. at ai-noise-multiplier 1.0) but direction holds — report the attenuation. Fragile if direction flips.
- **Finding 2 (inverted-U):** robust if the cascade peak stays within AI-proportion ∈ [0.1, 0.3] across levels. Bounded if the peak migrates but remains a minority-AI peak. Fragile if the peak moves to majority AI or vanishes.
- **Finding 3 (threshold):** robust if the safe-zone edge stays within human-bias ∈ [0.15, 0.40]. Bounded if it shifts monotonically with fatigue-threshold (an interesting, reportable result). Fragile if cascade rate decouples from regime-bias.

A *bounded* or *fragile* result is **not a failure** — it is the honest robustness statement the reviewers asked for, and a stronger paper than a fragile finding presented as universal.

## Instruments

- INST-coherence-model-tencon — NetLogo entrainment-coherence ABM, TENCON fork (AI agent types, mixed-regime, fatigue v1)
- `notebooks/h001_batch1_analysis.py`, `notebooks/h003_analysis.py` — BehaviorSpace parsers + cascade/diversity analysis

## Compute Budget (confirm before launch)

OAT design, 30 reps:

- **S-H004:** ~13 parameter-levels (across 4 sweeps, default shared) × 5 AI-proportions × 2 modes × 30 reps ≈ **3,600–3,900 runs**, single perturbation (≤3000 ticks each).
- **S-H005:** ~5 parameter-levels × 5 bias levels × 2 AI-proportions × 30 reps ≈ **1,500 runs**, periodic perturbation (longer per run).

**Release valves if the budget or deadline tightens** (in order): drop reps 30→20; thin the AI-proportion axis to {0, 0.2, 0.5}; cut `fatigue-intensity` and `ai-tie-strength-multiplier` (the two least load-bearing sweeps). The validation gate and the three core sweeps (update-rate, influence-radius, noise, fatigue-threshold) are the floor.

## Reporting

One **Robustness** subsection in the camera-ready (end of §5 Results or top of §6 Discussion): one figure (cascade/diversity vs parameter, faceted) + one compact table of robust/bounded/fragile verdicts per finding, plus ≤1 paragraph. Full per-run data deposited to the Zenodo record alongside the existing H001–H003 exports.

## Output locations

- BehaviorSpace CSVs → `exports/S-H004_*.csv`, `exports/S-H005_*.csv`
- Analysis notebook → `notebooks/sensitivity_analysis.py`
- Figure → `paper/figures/` and `publications/papers/tencon2026/figures/`
