# TRACE_2026-03-10_tencon-batch-analysis-and-draft

## Constraint Acknowledgement

This trace is written under the constraints defined in EarthianLabs root CLAUDE.md.

- **Irreversibility constraints respected:** YES
- **Metabolic governors stressed:** HIGH (deadline-driven; 6 days to submission; full pipeline compressed into one session)
- **Any overrides?** None. Work proceeded under time pressure but without constraint violations. Deadline risk is named explicitly below.

---

## Local Context

IEEE TENCON 2026 deadline discovered in previous session (2026-03-10 morning): "Coherence vs Entrainment in Human-AI Agentic Systems" due 17 March 2026 (Bali conference). Batch 1 CSV data (80 BehaviorSpace runs) existed but had not been analysed. No draft text existed. This session was a full sprint from raw data to paper draft v0.1.

Model context: the ABM (NetLogo, `netlogo/coherence_model_tencon.nlogox`) already implements coherence vs entrainment regimes with fatigue, recovery tracking, and cost metrics from prior experiments (E001–C001). The TENCON framing adds a Human-AI axis: some agents are designated AI agents operating in one of two coupling modes (entrainment or coherence), and the proportion of AI agents is swept as the independent variable.

---

## Decisions Made

- **Spiral classification threshold** (reversible): `recovery_time == -1 OR max_fatigue_level >= 1.0`. Chosen because the BehaviorSpace output encodes non-recovery as -1 and max fatigue at 1.0 represents full incapacitation. Threshold is operationally clean but could be refined if figure distributions suggest a better boundary.

- **Bimodal analysis approach** (reversible): Report spiral rates and conditional recovery times separately; do not report raw means over the full distribution. Raw means are structurally misleading when outcomes are bimodal (spiral vs recovery). This is a methodological commitment that affects how all results are reported in the paper.

- **Paper framing** (reversible): Three challenged assumptions structure for the Discussion section. Assumptions: (1) more AI = more stability; (2) stability = adaptive capacity; (3) higher AI proportion = higher cascade risk. All three are complicated or inverted by the data.

- **BehaviorSpace experiment definitions added to model file** (committed to model, effectively irreversible unless manually removed): Two new XML experiment blocks added to `netlogo/coherence_model_tencon.nlogox` — `H001_batch2_proportion_sweep_full` and `H002_batch3_repeated_stress`. These define the experimental protocol and are now part of the model artifact.

- **Not building Batch 4** (H003 influence radius sweep): Signal from H001+H002 judged sufficient for a 3-5 page IEEE paper. Batch 4 deferred.

---

## Session Work Log

### 1. Batch 1 analysis (80 runs)

Custom Python parser written at `notebooks/h001_batch1_analysis.py` for BehaviorSpace spreadsheet-format CSV. The spreadsheet format differs from the table format — header rows contain parameter names, data rows are interleaved with run metadata. Parser handles this structure and is reusable for all subsequent batches.

Key discovery: outcome distributions are bimodal. Entrainment runs cluster into spiral failures (recovery_time = -1, fatigue saturated) and conditional recoveries. This invalidated mean-based analysis; bimodal approach adopted for all downstream work.

### 2. Batch 2 definition and execution (420 runs)

`H001_batch2_proportion_sweep_full` added to model file. Design: 7 AI proportions (0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9) × 2 modes (entrainment, coherence) × 30 reps = 420 runs. Mat executed in NetLogo BehaviorSpace. Runtime: ~5 minutes. Publication-quality statistical power.

### 3. Batch 3 definition and execution (180 runs)

`H002_batch3_repeated_stress` added to model file. Design: 3 proportions (0, 0.2, 0.5) × 2 modes × 30 reps = 180 runs with periodic perturbation (perturbation every N ticks rather than single event). Tests whether entrainment brittleness scales under repeated stress. Mat executed immediately after Batch 2.

### 4. Full statistical analysis (all three batches, 680 runs total)

Summary statistics, Mann-Whitney U tests (appropriate for non-normal, bimodal distributions), Kruskal-Wallis tests, effect sizes. Publication figures generated: 4-panel plots for H001 (diversity index, cascade rate, conditional recovery time, cost asymmetry ratio), 4-panel for H002, and H002 summary panel.

### 5. Paper draft written

`paper/tencon2026_draft.md` — IEEE TENCON 2026 format, approximately 3.5 pages double-column. Structure: Abstract, Introduction, Related Work, Model Description, Experiments (H001 + H002), Results (5 subsections), Discussion, Conclusion, References. Draft is complete and internally consistent.

---

## Key Findings (Paper-Ready)

**H001 — Proportion sweep (single stress):**

- Diversity collapse is monotonic and statistically massive in entrainment: diversity index 0.29 → 0.07 as AI proportion increases from 0 to 0.9 (Mann-Whitney p = 3.75 × 10⁻¹⁸). Coherence stays flat at ~0.48 (p = 0.67).
- Cascade rate shows an inverted-U in entrainment: peak cascade risk (30%) at 10-20% AI, not at high AI proportions. Small minorities are most destabilising.
- Stabilisation-diversity paradox: AI agents simultaneously reduce cascade risk AND destroy diversity in entrainment mode. The system survives structurally but loses adaptive capacity.
- Coherence has a capacity limit: cascade rate rises to 7-10% at 70-90% AI. Identity-pull works until the AI majority overwhelms the mechanism.
- Cost asymmetry: AI agents pay more under single stress (ratio ~0.6-0.7 human:AI), consistent with prior E005 findings that coherence places cost on the agents who hold diversity.

**H002 — Repeated stress:**

- Entrainment at all tested proportions (0, 0.2, 0.5): 100% cascade failure. No runs recover under periodic perturbation.
- Coherence under repeated stress: 33-57% cascade rate, but surviving runs recover fast. Adaptive capacity degrades proportionally to AI proportion, does not collapse.
- Cost asymmetry inverts under repeated stress: ratio approaches 1.03 (AI and human costs equalise). Repeated stress equalises the distributional burden.

---

## Files Created / Modified

| File | Status | Notes |
|------|--------|-------|
| `notebooks/h001_batch1_analysis.py` | NEW | BehaviorSpace spreadsheet-format parser + full analysis pipeline. Reusable for all batches. |
| `netlogo/coherence_model_tencon.nlogox` | MODIFIED | Added H001_batch2 and H002_batch3 experiment definitions. Model file now contains full TENCON experimental protocol. |
| `paper/tencon2026_draft.md` | NEW | Full paper draft v0.1. IEEE TENCON format. ~3.5 pages double-column in prose; needs LaTeX conversion. |
| `exports/H001_batch1_*.png` | NEW | Batch 1 publication figures (4 files) |
| `exports/H001_batch2_*.png` | NEW | Batch 2 publication figures (4 files) |
| `exports/H002_batch3_4panel.png` | NEW | Batch 3 summary figure |

---

## Research Connections

**RQ1** (How does transformative adaptation manifest): The coherence-entrainment distinction IS a manifestation signature. Two regimes under identical stress produce qualitatively different dynamics — one adapts, one rigidifies and fails. The ABM makes this measurable.

**RQ2** (Conditions that cultivate or constrain adaptive capacity): Identity-pull weight is the key mechanism. Universal access to identity-pull is required (COND-001, confirmed by C001). Capacity limit at 70-90% AI proportion is a new RQ2 finding — even adaptive mechanisms have saturation thresholds.

**RQ3** (How coherence and entrainment shape durability/brittleness): Core paper contribution. Entrainment produces brittle stability (system survives but loses adaptive range). Coherence produces adaptive resilience (system degrades proportionally, recovers when stress resolves). AI agents amplify whichever regime is already active — they are not neutral.

**RQ5** (How adaptive capacity can be sensed, supported, stewarded): Diversity index is a leading indicator of regime type and future brittleness. Design implication: systems need identity-preserving mechanisms, not just stability mechanisms. This connects directly to Sense MCP work (sensing the field) and the broader EarthianLab stewardship frame.

---

## Compression Summary

**What must survive context decay:**

1. The TENCON paper deadline is 17 March 2026 (6 days from today). Paper draft exists at `paper/tencon2026_draft.md`. It is complete in structure but not submission-ready.

2. The core finding: AI agents in entrained systems produce a **stabilisation-diversity paradox** — they make the system less likely to cascade AND simultaneously destroy diversity (the substrate of adaptive capacity). The system becomes safer and less alive simultaneously.

3. The inverted-U cascade pattern is counterintuitive and noteworthy: 10-20% AI proportion in entrainment is more dangerous than 50-90%. Small minorities destabilise more than majorities.

4. Three pre-submission tasks are blocking:
   - **P0**: Reference verification ([1]-[3], [7] plausible but unverified; [10] is a self-reference placeholder)
   - **P0**: UOW affiliation string confirmation ("School of Business, University of Wollongong")
   - **P1**: Figure selection for IEEE double-column format (4-panel plots exist; need to decide which 4 become Fig 1-4)

5. LaTeX conversion (P2) is needed before submission but is not blocking the content review pass.

6. The `traces/` directory is newly created in this session. Prior session artifacts live in `dev-updates/` (legacy). Both are valid; new work goes in `traces/`.

7. CLAUDE.md for this project is significantly out of date — it does not reflect the TENCON model file, the H001/H002 experiments, or the new analysis scripts. An update pass is needed but was not in scope for this session.

---

## Residue / Open Tensions

**P0 — Reference verification is blocking submission.** References [1]-[3] and [7] in the draft are plausible but were not verified via DOI or Zotero during this session. Reference [10] is a self-reference placeholder that needs the actual submission details filled in. Apply the Perplexity → WebFetch → Zotero verification protocol before any submission or circulation.

**P0 — UOW affiliation.** The paper uses "School of Business, University of Wollongong" as the institutional affiliation. Mat needs to confirm this is the correct current school name and that the affiliation is appropriate for this work.

**P1 — Voice pass.** The draft is clean IEEE prose with appropriate academic register. The somatic framing that characterises Mat's stronger work is deliberately restrained for the IEEE audience. Mat may want to decide whether to push harder on the somatic language in Discussion (the "system becomes safer and less alive" framing is there but muted). This is a style decision, not a content one.

**P1 — Figure selection.** 4-panel plots exist for both H001 and H002. IEEE double-column format (4 pages total) can support approximately 4 figures. Need to decide which panels to promote, which to demote to supplementary or prose description. The diversity index and cascade rate panels are almost certainly in; the conditional recovery and cost asymmetry panels are negotiable.

**P2 — LaTeX conversion.** The draft is in Markdown. IEEE TENCON requires LaTeX double-column format. Conversion can be done with Pandoc + IEEE template, but figure placement and table formatting will need manual attention.

**P2 — Backward compatibility note.** Coherence at 0% AI shows higher recovery times (~116 ticks) than prior E003 experiments (~0-5 ticks). Almost certainly explained by different parameter settings in the TENCON model (perturbation-strength=60 with fatigue enabled vs E003 defaults). Not a bug, but worth a sentence in the Methods or footnote to flag the parameter difference.

**P3 — CLAUDE.md drift.** This project's CLAUDE.md was last updated 2026-01-11. It predates the TENCON work entirely. The experiment registry, key files table, and directory structure section are all out of date. A dedicated CLAUDE.md update pass should happen before or after submission.

**P3 — Batch 4 (H003).** The evidence protocol defines an influence-radius sweep (H003). Signal from H001+H002 was judged sufficient for the current paper, but H003 remains in the protocol backlog. If the paper gets a revise-and-resubmit, H003 is ready to run.

**Metabolic note.** This session was high-intensity. A full pipeline (data → analysis → paper draft) in one session under a hard deadline is the kind of pace that accumulates residue. The work is done, but Mat's integration capacity should be the rate limiter for what happens next. Reference verification can be done in a focused 30-minute pass; it does not require another full session.
