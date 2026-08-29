# Coherence vs Entrainment in Human–AI Agentic Systems

**Mathew Mytka**
Faculty of Arts, Society, and Business, University of Wollongong, Wollongong, Australia
mmytka@uow.edu.au

---

## Abstract

As AI agents are integrated into socio-technical systems, a critical question emerges: does tighter coordination make these systems more resilient or more brittle? This paper uses an agent-based model to distinguish two coordination regimes — *entrainment* (phase-locking alignment) and *coherence* (identity-preserving coupling) — and investigates how increasing AI agent proportion affects system resilience under perturbation. Across 1,500 simulation runs spanning three experiments, we find four results that challenge conventional assumptions. First, AI agents reduce cascade failure rates in entrained systems while simultaneously collapsing diversity — the system survives but loses adaptive capacity. Second, a small minority of AI agents (10–20%) produces the highest cascade risk, not a majority. Third, under repeated perturbation, entrained systems fail completely regardless of AI proportion, while coherent systems degrade gracefully. Fourth, in mixed-regime populations where agents vary along a continuous coherence–entrainment spectrum, human identity-preservation below a critical threshold is sufficient to maintain system resilience even when AI agents are structurally biased toward entrainment. These findings suggest that optimising human–AI systems for alignment and speed can produce a form of stability that is antithetical to resilience, and that preserving human autonomy is a structural requirement for collective adaptive capacity.

**Keywords:** agent-based modelling, human-AI systems, resilience, entrainment, coherence, socio-technical systems, coordination dynamics

---

## I. Introduction

The deployment of agentic AI systems into organisational and societal coordination is accelerating. These systems typically operate with faster update rates, lower noise, and broader influence than their human counterparts. The prevailing design assumption is that tighter coupling and faster convergence produce better coordination outcomes.

This paper interrogates that assumption. Drawing on dynamical systems theory and cybernetic principles, we distinguish between two coordination regimes observable in coupled agent populations. *Entrainment* describes phase-locking dynamics where agents converge toward shared states, suppressing internal diversity. *Coherence* describes relational coordination where agents maintain distinct identities while adapting to shared perturbations. Both can appear stable under normal conditions. They diverge under stress.

The distinction matters because AI agents, by design, amplify the properties that drive entrainment: precision, speed, reach, and coupling strength. If entrainment carries hidden fragility costs, then the integration of AI agents into human coordination systems may systematically degrade resilience while appearing to enhance it.

We investigate this through an agent-based model (ABM) that introduces heterogeneous AI agents into an existing entrainment-coherence framework. Three experiments test system behaviour: first across AI proportions in discrete regimes, then under periodic stress, and finally in mixed-regime populations where agents vary continuously along the coherence–entrainment spectrum. The results reveal a paradox: AI agents can simultaneously stabilise a system and hollow out the diversity that makes it adaptive — and a design principle: even modest human identity-preservation is sufficient to counteract it.

## II. Related Work

Research on human–AI teaming has focused primarily on task performance, trust calibration, and decision support [1]–[3]. Less attention has been paid to the *dynamical properties* of mixed human–AI populations under stress. The resilience literature distinguishes engineering resilience (return to equilibrium) from ecological resilience (persistence of function through reorganisation) [4], but this distinction is rarely applied to AI systems design. Contucci et al. [5] demonstrated using a statistical physics model that arbitrarily small changes in AI agent proportion can trigger abrupt phase transitions in human–AI ecosystems — a finding our results extend to coupled oscillator dynamics with an identity-preserving mechanism.

Agent-based models of synchronisation and coordination draw on Kuramoto-type coupled oscillator frameworks [6], where agents adjust internal states based on neighbours' states weighted by coupling strength. Phase transitions between synchronised and desynchronised states are well characterised for homogeneous populations. The introduction of heterogeneous agent properties — particularly differential update rates and influence radii — produces richer dynamics including partial synchronisation and chimera states [7]. Mitra [8] recently applied Kuramoto-type dynamics to heterogeneous multi-agent AI systems, demonstrating synchronisation behaviour across agents with varying natural frequencies. Our model extends this approach to mixed human–AI populations under perturbation, with an explicit distinction between entrainment and coherence regimes.

The coherence-entrainment distinction used here builds on prior work in allostatic regulation [9] and living systems theory [10]. Entrainment as used in this paper refers specifically to the suppression of internal degrees of freedom through coupling, distinct from the broader neuroscience usage. Coherence refers to the maintenance of functional coordination without requiring phase-locking — closer to what Bateson [11] described as the pattern which connects.

## III. Model Description

### A. Agent Architecture

The model implements a population of 100 agents on a toroidal grid, each characterised by a heading (orientation proxy for internal state), a preferred heading (identity), coupling bias (sensitivity to social influence, drawn from U(0,1)), inertia, and tie strength.

Two agent types are defined:

| Property | Human Agents | AI Agents |
|----------|-------------|-----------|
| Update rate | 1 tick/step | 5 ticks/step |
| Noise multiplier | 1.0 | 0.3 |
| Influence radius | 3 patches | 8 patches |
| Tie-strength multiplier | 1.0 | 1.5 |

These differentials are motivated by observable properties of deployed AI systems: faster response cycles, lower stochastic variation, broader information access, and amplified influence on connected agents.

### B. Coordination Regimes

Two regimes are implemented as a global mode switch:

**Entrainment mode.** Each agent adjusts heading toward the local mean of neighbours, weighted by coupling strength and tie strength. Social coupling drives convergence. No identity-preserving mechanism is active.

**Coherence mode.** Agents experience the same social coupling, attenuated by a social-pull weight (0.5), plus an identity-pull toward their preferred heading (weight 0.2). This provides a return attractor independent of the social field.

**Mixed mode.** Each agent is assigned a continuous regime-bias ∈ [0, 1] that determines the blend of social-pull and identity-pull in its update rule. At regime-bias = 0, the agent operates as in coherence mode; at 1, as in entrainment mode. AI agents draw regime-bias from N(0.8, 0.15), reflecting the structural properties that bias AI systems toward entrainment. Human agents draw from N(*μ*, 0.15) where *μ* is an experimental parameter. This continuous architecture allows investigation of realistic populations where coordination style varies within and between agent types.

### C. Stress and Fatigue

Perturbation is applied as a heading displacement of magnitude 60 (above the empirically established critical threshold [12]) for 144 ticks. Two perturbation regimes are tested: single (one event) and periodic (repeating).

A fatigue mechanism creates feedback between accumulated cost and agent responsiveness. Agents accumulate metabolic cost proportional to heading adjustment, shock cost during perturbation, and recovery cost during recovery. Once cumulative cost exceeds a fatigue threshold, effective inertia increases (reduced turning capacity) and recovery cost accelerates. This produces cascade dynamics: cost concentration on high-coupling agents triggers fatigue spirals where the most connected agents become the least responsive.

### D. Metrics

Recovery time: ticks from perturbation end to return within tolerance (5°) of baseline variance. A value of −1 indicates non-recovery within 3000 ticks (cascade failure). Diversity index: normalised heading variance. Cumulative cost: mean total regulatory cost per agent. Cost Gini: inequality of cost distribution. Human-AI cost ratio: mean human cost divided by mean AI cost.

## IV. Experimental Design

### A. H001: AI Proportion Sweep

**Design:** 7 AI proportions (0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9) × 2 modes × 30 repetitions = 420 runs. Single perturbation. Final-state metrics at tick 3000.

**Question:** At what proportion of AI agents does system behaviour change, and does the regime (coherence vs entrainment) moderate the effect?

### B. H002: Repeated Stress

**Design:** 3 AI proportions (0, 0.2, 0.5) × 2 modes × 30 repetitions = 180 runs. Periodic perturbation.

**Question:** Does periodic stress amplify or reveal different failure modes in mixed human–AI systems?

### C. H003: Mixed-Regime Populations

**Design:** 3 AI proportions (0, 0.2, 0.5) × 5 human regime-bias levels (0, 0.25, 0.5, 0.75, 1.0) × 2 stress types × 30 repetitions = 900 runs. Mixed mode with AI regime-bias fixed at 0.8.

**Question:** How much human identity-preservation is required to maintain adaptive capacity when AI agents are structurally biased toward entrainment?

### D. Analysis

Given bimodal outcome distributions (cascade vs recovery), we report cascade failure rates alongside conditional recovery statistics. Between-condition comparisons use Mann-Whitney U tests with rank-biserial effect sizes. Proportion effects within modes use Kruskal-Wallis tests. Cascade proportion comparisons use Fisher's exact test. All tests use α = 0.05.

## V. Results

### A. Diversity Collapse Under Entrainment

The strongest effect across the experiments is the monotonic collapse of diversity in entrained systems as AI proportion increases (Fig. 1b). Diversity index drops from 0.29 (0% AI) to 0.07 (90% AI) in entrainment mode (Kruskal-Wallis H = 94.36, p = 3.75 × 10⁻¹⁸). Coherence mode shows no significant diversity change across proportions (H = 4.05, p = 0.67), maintaining diversity at approximately 0.48 throughout.

Mode differences in diversity are significant at every AI proportion tested (Mann-Whitney p < 10⁻⁴ at all levels, rank-biserial r = −0.61 to −1.00).

### B. The Inverted-U Cascade Pattern

Cascade failure rates in entrainment mode exhibit an inverted-U relationship with AI proportion (Fig. 1a). At 0% AI, 10% of runs cascade. This rises to 30% at 10% AI and 27% at 20% AI, then declines to 10% at proportions above 50%. A small minority of AI agents is the most destabilising configuration.

Coherence mode shows zero cascades from 0% to 50% AI, with cascades emerging at 70% (7%) and 90% (10%). The identity-pull mechanism has a capacity limit: when AI agents constitute the overwhelming majority, even coherence mode can be overwhelmed.

### C. The Stabilisation-Diversity Paradox

At high AI proportions in entrainment mode, two things happen simultaneously: cascade risk decreases (the system fails less often) and diversity collapses (the system retains less adaptive capacity). At 90% AI, only 10% of entrained runs cascade — the same rate as 0% AI — but surviving systems have diversity index 0.07 compared to 0.22 at 0% AI.

This is the central finding. AI agents in entrained systems produce stability at the cost of the internal heterogeneity that enables adaptation to novel conditions. The system becomes more predictable and less capable. Coherence mode resolves this paradox: diversity remains near 0.48 regardless of AI proportion, and cascade rates remain near zero through 50% AI.

### D. Complete Failure Under Repeated Stress

Under periodic perturbation (H002), the distinction between regimes becomes absolute (Fig. 2). Entrainment mode produces 100% cascade failure at all AI proportions tested (0%, 20%, 50%). Mean cumulative cost reaches 16,000–18,000, with maximum fatigue levels at 1.0 across all runs.

Coherence mode degrades but survives. Cascade rates increase with AI proportion (33% at 0%, 43% at 20%, 57% at 50%), but surviving runs recover rapidly (median recovery < 13 ticks). Diversity is maintained at 0.47–0.50 even under repeated stress. The identity-pull mechanism provides a recovery attractor that the social field alone cannot.

### E. Cost Asymmetry

Under single perturbation, the human-AI cost ratio is approximately 0.6–0.7 in both modes: AI agents bear more regulatory cost than humans, consistent with their greater influence and faster update rates. Under periodic stress, this ratio shifts to approximately 1.03 — near parity. Repeated stress eliminates the cost differential as both agent types saturate their fatigue capacity.

### F. Human Identity-Preservation as a Threshold Effect

H003 replaces the binary regime switch with a continuous spectrum. The results show that cascade failure rate scales monotonically with human regime-bias (Fig. 3). At 0% AI under periodic stress, cascade rates rise from 3% (bias = 0, coherence pole) to 67% (bias = 1, entrainment pole; Fisher exact p < 0.0001). At 20% AI — the peak-risk zone identified in H001 — periodic stress produces 0% cascades when human bias = 0 but 60% when bias = 1 (Fisher exact p < 0.0001).

Diversity collapses along the same gradient. At 20% AI under single stress, diversity drops from 0.48 (bias = 0) to 0.17 (bias = 1; Kruskal-Wallis H = 75.85, p < 10⁻⁶). The effect of human regime-bias on diversity is significant at all AI proportions tested (all p < 10⁻⁵).

A critical threshold emerges near bias = 0.25. At this level, cascade rates remain near zero across all AI proportions and stress types, while diversity stays above 0.43. Beyond bias = 0.5, cascade rates accelerate and diversity degrades. The transition is not linear — the system tolerates moderate entrainment bias in its human population but deteriorates rapidly once identity-preservation falls below a threshold.

Notably, mixed-regime populations at the coherence pole (bias = 0) outperform the binary coherence mode from H002 at 20% AI under periodic stress: 0% cascade versus 43%. The within-population heterogeneity of regime-bias itself appears to contribute to resilience, providing a diversity of response strategies that a uniform mode cannot.

## VI. Discussion

### A. Implications for Human–AI System Design

The results challenge three common assumptions in AI systems design.

*More AI agents improve coordination.* True for cascade avoidance in entrainment mode, false for adaptive capacity. The diversity collapse observed in entrained systems with high AI proportions represents a form of coordinated fragility.

*Stability equals resilience.* Entrained systems at 90% AI appear highly stable under single perturbation (low cascade rate, fast convergence). Under repeated stress, they fail completely. Stability and resilience are not the same property.

*The most dangerous configuration is total AI dominance.* The inverted-U pattern in cascade rates suggests the opposite. A small minority of AI agents (10–20%) introduces enough dynamical heterogeneity to trigger cascades without sufficient coordination to suppress them. This has implications for real-world deployment: early-stage AI integration may carry higher systemic risk than full automation.

### B. The Coherence Design Principle

Coherence mode's resilience advantage derives from a single mechanism: identity-pull. Agents maintain a return attractor independent of the social field. This prevents the coupling-fatigue spirals that drive cascade failure in entrainment mode, while preserving the diversity that enables adaptive response.

H003 extends this from a binary design choice to a quantitative requirement. The threshold near regime-bias = 0.25 means that human agents do not need to be fully identity-preserving — they need only maintain modest autonomy from the social field. This is a weaker condition than full coherence, and a more realistic one. The finding that mixed-regime heterogeneity itself confers resilience suggests that systems benefit from a *diversity of coordination strategies*, not just a single optimal mode.

The identity-pull mechanism is an analogue of what living systems theorists describe as autopoietic self-maintenance [10] — the capacity to maintain organisational identity while structurally coupling with the environment. Systems that sacrifice this capacity for tighter coordination gain short-term efficiency at the cost of long-term viability. H003 quantifies the minimum identity-preservation required: enough to anchor, not enough to decouple.

### C. Limitations

The heading-based state space is a simplification of the multidimensional coordination problems found in organisational contexts. The AI agent property differentials, while motivated by observed system properties, are not calibrated to specific deployed systems. The periodic perturbation regime applies uniform stress; real-world stressors are typically heterogeneous and correlated. The regime-bias distribution uses a fixed standard deviation (0.15); real populations may exhibit wider or non-normal variation in coordination style. Regime-bias is static in these experiments; real agents may shift coordination strategy over time through interaction.

## VII. Conclusion

This paper presents an agent-based model investigating the effects of AI agent integration on the resilience of coordinated socio-technical systems. Across 1,500 simulation runs, we find that AI agents can simultaneously reduce cascade risk and destroy the diversity that enables adaptive response. Entrainment-optimised systems appear stable but collapse completely under repeated stress, while coherence-preserving systems degrade gracefully. A small minority of AI agents (10–20%) produces the highest cascade risk. In mixed-regime populations, human identity-preservation below a critical threshold is sufficient to maintain resilience even when AI agents are structurally biased toward entrainment — and the heterogeneity of coordination strategies itself confers protective benefits beyond any single regime.

These results suggest that the design of resilient human–AI systems requires mechanisms that preserve identity and heterogeneity under coupling pressure. The practical implication is specific: human participants in AI-integrated systems need not resist coordination entirely, but they must retain sufficient autonomy to return to their own position when coupling pressure becomes destructive. Optimising for alignment, speed, and convergence produces coordination that is antithetical to the adaptive capacity it claims to support.

## Acknowledgements

The agent-based model was developed in NetLogo. Statistical analysis of BehaviorSpace experiment outputs was conducted using Python scripts. Claude (Anthropic) was used as a research assistant during the analysis phase: specifically for data parsing, statistical computation, literature discovery, and manuscript drafting. All model design, experimental protocol, interpretation of results, and theoretical framing are the author's own. The source code and experiment data are available at [12].

## References

[1] A. I. Hauptman, B. G. Schelble, N. J. McNeese, and K. C. Madathil, "Adapt and overcome: Perceptions of adaptive autonomous agents for human–AI teaming," *Comput. Hum. Behav.*, vol. 138, Art. no. 107451, Jan. 2023.

[2] B. Shneiderman, "Human-centered artificial intelligence: Reliable, safe & trustworthy," *Int. J. Hum.–Comput. Interact.*, vol. 36, no. 6, pp. 495–504, 2020.

[3] M. Johnson et al., "Coactive design: Designing support for interdependence in joint activity," *J. Hum.–Robot Interact.*, vol. 3, no. 1, pp. 43–69, 2014.

[4] C. S. Holling, "Resilience and stability of ecological systems," *Annu. Rev. Ecol. Syst.*, vol. 4, pp. 1–23, 1973.

[5] P. Contucci, J. Kertész, and G. Osabutey, "Human-AI ecosystem with abrupt changes as a function of the composition," *PLOS ONE*, vol. 17, no. 5, Art. no. e0267310, May 2022.

[6] Y. Kuramoto, *Chemical Oscillations, Waves, and Turbulence.* Berlin, Germany: Springer, 1984.

[7] D. M. Abrams and S. H. Strogatz, "Chimera states for coupled oscillators," *Phys. Rev. Lett.*, vol. 93, no. 17, 2004, Art. no. 174102.

[8] C. Mitra, "Synchronization dynamics of heterogeneous, collaborative multi-agent AI systems," arXiv:2508.12314, Aug. 2025.

[9] P. Sterling, "Allostasis: A model of predictive regulation," *Physiol. Behav.*, vol. 106, no. 1, pp. 5–15, 2012.

[10] H. R. Maturana and F. J. Varela, *Autopoiesis and Cognition: The Realization of the Living.* Dordrecht, The Netherlands: Reidel, 1980.

[11] G. Bateson, *Mind and Nature: A Necessary Unity.* New York, NY, USA: Dutton, 1979.

[12] M. Mytka, "Coherence vs entrainment in human-AI agentic systems: Agent-based model and experiment data," Zenodo, 2026, doi: 10.5281/zenodo.19017873.

---

*Draft v0.3 — 14 March 2026. Editorial pass complete. References verified. Pending: Zenodo DOI for [12], LaTeX conversion, figure selection.*
