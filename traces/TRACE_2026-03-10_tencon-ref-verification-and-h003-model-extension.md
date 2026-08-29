# TRACE_2026-03-10_tencon-ref-verification-and-h003-model-extension

## Constraint Acknowledgement

This trace is written under the constraints defined in EarthianLabs root CLAUDE.md.

- **Irreversibility constraints respected:** YES
- **Metabolic governors stressed:** HIGH (deadline-driven; model extension + reference verification + paper updates in one session; H003 running overnight)
- **Any overrides?** None. The irreversibility of the per-agent regime-bias architecture is named and accepted below as a deliberate structural choice, not a governance violation.

---

## Local Context

Continuation of TENCON 2026 submission sprint. Deadline: 17 March 2026 (5 days from today). Previous session (same date, earlier) produced paper draft v0.1 from raw BehaviorSpace data. This session addressed the two P0 blockers from that trace (reference verification and UOW affiliation), added two new references, and executed a significant model extension (H003: per-agent continuous regime-bias).

The model extension was not planned at the start of this session. It emerged from the observation that the binary entrainment/coherence switch was a simplification that misrepresents realistic human-AI populations, where agents are distributed across a spectrum rather than sorted into two discrete camps. The extension changes the paper's scope: it now includes a fourth experiment showing what happens in mixed-regime populations.

H003 BehaviorSpace batch (900 runs) is currently executing. Next session is analysis and paper integration.

---

## Decisions Made

- **Per-agent continuous regime-bias architecture** (irreversible for paper structure): Replaced the global `entrainment-mode?` boolean with a per-turtle `regime-bias` variable [0, 1]. This changes the paper from a binary comparison to a spectrum analysis. The model cannot easily be reverted to the binary architecture without rewriting the update rule. The paper's Model Description section must be rewritten to reflect the continuous framing.

- **AI agents structurally biased toward entrainment** (reversible via slider): `ai-regime-bias-mean` defaults to 0.8. This reflects the design judgment that AI agents are closer to the entrainment pole — faster, lower noise, higher social coupling — by structural architecture rather than by choice. The default can be changed in the UI or overridden in BehaviorSpace experiments.

- **Human regime-bias drawn from normal distribution** (reversible): Centred on `human-regime-bias-mean` with SD 0.15, representing natural intra-population variation. The SD is a parameter choice that has not been swept; 0.15 is a reasonable starting estimate.

- **Unified update rule blending** (irreversible for H003 validity): The `update-heading` procedure now blends social-pull and identity-pull proportionally to `regime-bias`. Boundary conditions are verified: regime-bias=0 reproduces old coherence behaviour; regime-bias=1 reproduces old entrainment behaviour. This is a deliberate architectural commitment, not a temporary patch.

- **Regime drift deferred** (explicitly parked): The option to implement regime-bias shifting over time through interaction was discussed and deferred to post-submission. Rationale: it would add another dimension to an already complex experimental design. Candidate for post-publication extension work.

- **H003 BehaviorSpace design** (irreversible as executed): 3 AI proportions × 5 human-regime-bias levels × 2 stress types × 30 reps = 900 runs. Currently running. Design cannot be changed without discarding in-progress runs.

- **Paper copy moved to publications/** (reversible): `publications/papers/tencon2026_draft.md` is a copy of the working draft in `paper/tencon2026_draft.md`. The model repo copy remains the source of truth for paper content.

---

## Session Work Log

### 1. Reference verification

Checked references [1]–[3] and [7] from the draft against actual publications.

**[1] Hauptman et al** — authors were wrong. The draft had "E. de Visser" as an author (hallucinated). Correct authors: Hauptman, Schelble, McNeese, Madathil. Fixed in draft. DOI confirmed.

**[2] Shneiderman 2020** — verified clean. DOI: 10.1080/10447318.2020.1741118.

**[3] Johnson et al 2014** — verified clean. DOI: 10.5898/JHRI.3.1.Johnson.

**[7] Sterling 2012** — verified clean. DOI: 10.1016/j.physbeh.2011.06.004.

### 2. UOW affiliation correction

"School of Business" corrected to "Faculty of Arts, Society, and Business" in paper draft. This was the P0 blocker from the previous trace.

### 3. Novelty scan

Surveyed literature for direct threats to the three core novelty claims: (1) coherence-entrainment distinction in human-AI systems; (2) inverted-U cascade pattern; (3) resilience framing for ABM. No direct threats found. The combination of Kuramoto ABM + mixed human-AI + coherence vs entrainment + resilience under stress remains novel.

### 4. Two new references added and integrated

**[5] Contucci, Kertész & Osabutey (2022)**, "Human-AI ecosystem with abrupt changes as a function of the composition," PLOS ONE. DOI: 10.1371/journal.pone.0267310. Statistical physics model showing phase transitions from small AI proportion changes — convergent evidence for the inverted-U finding. Strengthens the credibility of the cascade rate pattern.

**[8] Mitra (2025)**, "Synchronization dynamics of heterogeneous, collaborative multi-agent AI systems," arXiv:2508.12314. Kuramoto model for heterogeneous AI agents — closest methodological neighbour identified. Important to cite and differentiate. Key distinction: Mitra focuses on synchronisation optimisation; this paper focuses on adaptive capacity and resilience costs under stress.

Reference list renumbered throughout draft: now [1]–[12] (was [1]–[10]).

### 5. AI disclosure section drafted

IEEE TENCON 2026 mandates disclosure of AI-generated content. Acknowledgements section written covering: Claude Code (Anthropic) used for code development and writing assistance; NetLogo model design and statistical analysis are the authors' own work; all claims verified against primary sources.

### 6. Submission logistics confirmed

- EDAS submission system
- PDF eXpress ID: 69637X (already open)
- Format: 3–5 pages A4, double-column IEEE
- Similarity threshold: <30%
- Review: single-blind

### 7. H003 model extension (continuous regime-bias)

Modified `netlogo/coherence_model_tencon.nlogox`:

- New turtle variable: `regime-bias` [0, 1] — position on coherence-entrainment spectrum
- New chooser: `coordination-regime` ("entrainment" / "coherence" / "mixed") — replaces `entrainment-mode?` boolean switch
- New sliders: `human-regime-bias-mean` (default 0.3), `ai-regime-bias-mean` (default 0.8)
- Unified `update-heading` procedure: blends social-pull and identity-pull proportionally to regime-bias
- New reporters: `mean-regime-bias`, `human-mean-regime-bias`, `ai-mean-regime-bias`, `regime-bias-variance`
- H003 BehaviorSpace experiment added: 3 AI proportions (0, 0.2, 0.5) × 5 human-regime-bias levels (0, 0.25, 0.5, 0.75, 1) × 2 stress types (single, periodic) × 30 reps = 900 runs
- All existing H001 and H002 experiments updated from `entrainment-mode?` boolean to `coordination-regime` string values
- Backward compatibility verified manually: boundary conditions reproduce old binary results

Mat tested the model manually and confirmed it works. H003 batch is currently running.

---

## Files Created / Modified

| File | Status | Notes |
|------|--------|-------|
| `paper/tencon2026_draft.md` | MODIFIED | Hauptman author fix; UOW affiliation fix; references [5] and [8] added; numbering updated [1]–[12]; AI disclosure section added |
| `publications/papers/tencon2026_draft.md` | NEW (copy) | Copy of working draft in publications directory |
| `netlogo/coherence_model_tencon.nlogox` | MODIFIED | Per-agent regime-bias architecture; unified update-heading rule; H003 experiment added; all prior experiments updated to coordination-regime string |

---

## Research Connections

**RQ1** (How transformative adaptation manifests): The continuous regime-bias now allows observation of adaptation as a spectrum rather than a binary. Mixed-regime populations exhibit intermediate dynamics — the model can now test whether realistic populations (neither fully entrained nor fully coherent) show qualitatively different resilience than the boundary conditions.

**RQ2** (Conditions that cultivate or constrain adaptive capacity): `human-regime-bias-mean` is the central experimental variable in H003 — directly tests how much identity-preservation is required in a human population to maintain adaptive capacity when AI agents are structurally biased toward entrainment. This is the most direct RQ2 operationalisation in the paper.

**RQ3** (Coherence and entrainment shape durability/brittleness): Extended from binary to continuous. H003 will allow identification of thresholds on the human-regime-bias spectrum where collective behaviour shifts from resilient to brittle. These thresholds are the RQ3 contribution from H003.

**RQ5** (Sensing, supporting, stewarding): The discussion section will connect H003 findings to current agentic AI deployment patterns. If AI systems are structurally biased toward entrainment (as the model assumes), what human-side conditions are sufficient to preserve adaptive capacity? This connects directly to design and stewardship questions.

---

## Compression Summary

**What must survive context decay:**

1. The TENCON paper deadline is 17 March 2026 (5 days from today). Draft exists at `paper/tencon2026_draft.md`. Reference verification is now complete. P0 blockers from the previous trace are resolved.

2. H003 BehaviorSpace batch (900 runs) is running now. It uses the new continuous regime-bias architecture. When it completes, the output file will be in `exports/`. The next session's P0 is parsing and analysing H003 output, then integrating findings into the paper.

3. The model has been structurally changed. `entrainment-mode?` (boolean) is gone. `coordination-regime` (chooser: "entrainment" / "coherence" / "mixed") is the new regime selector. Anything that references the old parameter will break. The CLAUDE.md parameter table is out of date.

4. The key new methodological element: per-agent `regime-bias` [0, 1] blends social-pull and identity-pull continuously. AI agents: regime-bias drawn from Normal(ai-regime-bias-mean=0.8, SD=0.15). Human agents: Normal(human-regime-bias-mean, SD=0.15). At boundary conditions this reproduces the old binary experiments exactly.

5. Two new references are in the draft: Contucci et al (2022) as convergent evidence for the inverted-U; Mitra (2025) as the closest methodological neighbour. Both need Zotero ingest when the research MCP profile is active.

6. The paper now has four experiments (H001 proportion sweep, H002 periodic stress, H003 mixed regime). On a 5-page IEEE limit, figure selection will be constrained. H003 data does not yet exist; figure decisions should wait until analysis is complete.

7. LaTeX conversion (P2) is still pending. IEEE template at `publications/papers/conference-latex-template-A4.zip`.

---

## Residue / Open Tensions

**P0 — H003 analysis** — 900 runs executing. This is the gate for all further paper work. Until the data is available and parsed, the paper cannot be completed. First task of next session.

**P0 — Paper rewrite for H003** — Model Description section needs a new subsection on the continuous regime-bias. Results section needs a new H003 subsection. Discussion needs to address what mixed-regime populations reveal about real-world human-AI systems. This is significant writing work under a tight deadline.

**P1 — Figure selection under page pressure** — Four experiments' worth of data for a 5-page limit. H003 will generate its own candidate figures. The paper needs 4–5 figures total; each existing experiment has 4-panel plots. Some findings will need to move to prose description or supplementary. This decision cannot be made until H003 data exists.

**P1 — Editorial/voice pass** — Mat wants to push harder on real-world implications for the TENCON theme ("Intelligent Systems for a Resilient and Sustainable Society"). The 10-20% AI proportion as peak risk zone and identity-preserving mechanisms as design requirement are strong practical claims that should drive the Discussion section. H003 findings may sharpen or complicate these claims.

**P2 — LaTeX conversion** — Still pending. Pandoc to IEEE A4 template + manual figure placement. Not blocking content review but is a hard deadline constraint.

**P2 — Zotero ingest** — Contucci et al (2022) and Mitra (2025) are in the draft but not in Zotero. Requires research MCP profile. Low effort, should be done before submission to maintain literature trail.

**P3 — CLAUDE.md parameter table out of date** — The parameter list in this project's CLAUDE.md does not include `regime-bias`, `coordination-regime`, `human-regime-bias-mean`, `ai-regime-bias-mean`, or any of the H003 reporters. The old `entrainment-mode?` parameter is listed but no longer exists in the model. A CLAUDE.md update pass is needed; deferred until after submission.

**Structural tension — scope vs deadline** — The H003 extension was the right decision scientifically (binary regime is a known simplification; continuous spectrum is more realistic). But it adds analysis work, writing work, and figure selection complexity to a five-day deadline. The extension was executed in a single session while H001/H002 analysis was still fresh. This is a calculated risk, not a drift signal. Named explicitly because it should be monitored: if H003 data is noisy or inconclusive, the paper may need to revert to the three-experiment structure.

**Metabolic note** — Two full-sprint sessions in one day (data analysis + draft in the morning; reference verification + model extension in the afternoon/evening). H003 runs overnight. Next session should be bounded: analysis script + statistical summary + draft section, not another architectural extension. The deadline is real but so is integration capacity.
