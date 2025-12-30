# Coherence vs Entrainment Model

A NetLogo agent-based model exploring the distinction between **coherence** (identity-preserving coordination) and **entrainment** (phase-locking alignment) in dynamical systems.

## The Coherence Theorem

> *In agent-based systems with heterogeneous identity and bounded coupling, regimes that preserve internal diversity will exhibit lower peak disruption and faster recovery under repeated perturbation than regimes optimised for phase alignment.*

This model tests this proposition through controlled experiments, comparing two coupling regimes under stress.

## Key Findings

**Load concentration without relief pathways is sufficient to explain cascade failure.**

When systems coordinate through phase-locking (entrainment), certain agents — those most socially sensitive and most different from consensus — bear disproportionate cost to stabilize the collective. If these load-bearing agents lack independent recovery pathways, they exhaust themselves and trigger cascade failure.

Systems that preserve individual reference points (coherence) provide relief pathways that prevent this cascade.

### Effect Sizes

| Comparison | Effect Size | Source |
|------------|-------------|--------|
| Recovery time ratio at high stress | **12x longer** (entrainment vs coherence) | E003 |
| Peak deviation ratio at high stress | **4x higher** (entrainment vs coherence) | E003 |
| Recovery cost under periodic stress | **30x higher** (entrainment vs coherence) | E005b |
| Fatigue amplification of recovery | **11x** vs 7x (entrainment vs coherence) | F001 |

## Quick Start

### Requirements

- [NetLogo 7.0.3](https://ccl.northwestern.edu/netlogo/download.shtml) or later
- Python 3.10+ (for analysis notebooks)

### Running the Model

1. Open `netlogo/coherence_model_simple.nlogox` in NetLogo
2. Click **Setup** to initialize agents
3. Toggle `entrainment-mode?` to switch between regimes
4. Click **Go** to run the simulation
5. Click **Perturb** to apply a disturbance and observe recovery dynamics

### Running Experiments

The model includes pre-configured BehaviorSpace experiments. To run:

1. Open the model in NetLogo
2. Go to **Tools > BehaviorSpace**
3. Select an experiment (e.g., E003, I004)
4. Click **Run** and export results as CSV

### Analyzing Results

```bash
cd coherence-model
pip install -r requirements.txt
jupyter notebook notebooks/behaviorspace_analysis.ipynb
```

## Project Structure

```
coherence-model/
├── netlogo/                    # NetLogo model files
│   └── coherence_model_simple.nlogox  # Main model
├── exports/                    # BehaviorSpace experiment outputs
├── notebooks/                  # Python analysis notebooks
│   ├── behaviorspace_analysis.ipynb   # Primary analysis
│   └── i00*_analysis.py               # Investigation scripts
├── notes/                      # Documentation and findings
│   ├── coherence-model.md             # Theoretical framework
│   └── PRELIMINARY_FINDINGS_2025-12.md # Synthesis document
├── LICENSE                     # Earthian Stewardship License
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## The Two Regimes

### Entrainment Mode
- **Attractor structure:** Single shared attractor (collective mean heading)
- **Baseline:** Low variance (~21-36 degrees)
- **Under stress:** Shared attractor destabilizes, feedback amplification
- **Failure mode:** Cascade via exhausted load-bearers

### Coherence Mode
- **Attractor structure:** Dual attractors (social field + preferred-heading)
- **Baseline:** Moderate variance (~80-103 degrees)
- **Under stress:** Local basins remain stable, parallel recovery
- **Resilience mechanism:** Overloaded agents can fall back to individual attractor

## Experiment Registry

| Code | Name | Design |
|------|------|--------|
| E003 | Stress scaling | 2x4 factorial (mode x strength) |
| E005a/b | Cost distribution | Single and periodic perturbation |
| F001 | Fatigue validation | 2x2 factorial (mode x fatigue-enabled) |
| I001-I004 | Spiral investigation | Mechanism discovery arc |
| S001a-rep | Scale sensitivity | 2x3x5 factorial (30 runs) |

See `notes/PRELIMINARY_FINDINGS_2025-12.md` for detailed analysis of all experiments.

## Mechanism Synthesis

**Load Concentration + Absence of Relief Pathways = Cascade Risk**

```
Agent configuration
├── Some agents are load-bearing (high coupling + far from consensus)
├── Load-bearing agents bear disproportionate stabilization work (1.6-2x)
│
├── If MANY load-bearers: burden distributed, each survives
├── If FEW load-bearers: each overloaded, fatigue accumulates
│
├── [ENTRAINMENT] No relief pathway -> exhaustion -> cascade failure
└── [COHERENCE]   Relief pathway (identity-pull) -> recovery -> stability
```

## Theoretical Context

This work connects to:

- **Ashby:** Requisite variety — entrainment suppresses adaptive capacity
- **Allostasis vs Homeostasis:** Stability through reorganization vs set-point maintenance
- **Resilience Engineering:** Graceful extensibility under stress

### Design Invariant

> Systems that rely on continuous synchronization must provide independent recovery pathways for high-load agents, or they will externalize collapse risk onto a shrinking subset of stabilizers.

## Glossary

| Term | Definition |
|------|------------|
| **Coherence** | Dynamic capacity to maintain integrity while absorbing perturbation |
| **Entrainment** | Phase-locking toward collective alignment |
| **Load-bearing agents** | Agents with high social sensitivity and positions far from consensus |
| **Relief pathway** | Independent internal reference point usable for recovery under fatigue |
| **Stress transition region** | Perturbation level where regime differences become pronounced (~30-50) |

## Citation

If you use this model in your research, please cite:

```
Mytka, M. M. (2025). Coherence vs Entrainment Model: Agent-based exploration of
coordination regimes under stress. https://github.com/[your-repo]
```

## License

This project is released under the [Earthian Stewardship License (ESL-A v0.1)](LICENSE).

**Key terms:**
- Free for non-commercial research, education, and community use
- Commercial use requires permission
- Must respect somatic sovereignty — no manipulation or entrainment without consent
- Improvements to safety/ethics must be shared back

## Contact

**Mat Mytka** — m3untold@gmail.com

Part of the [EarthianLab](https://github.com/[your-org]) ecosystem.

---

*"The spiral isn't about who's at risk. It's about whether they have anywhere to go when they're spent."*
