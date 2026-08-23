"""
validate_baseline.py — reproducibility gate for the TENCON 2026 seeded deposit.

Checks that the SEEDED, reproducible deposit (exports/seeded/, run-number-pinned RNG)
regenerates the headline numbers now stated in the paper, within Monte-Carlo bands.
Two purposes:

  1. GATE — before trusting any downstream analysis, the pipeline must reproduce the
     paper's baseline from the seeded deposit. If this fails, stop and fix the
     discrepancy (parser / model version / re-run) before claiming any result.
  2. EVIDENCE — answers Reviewer 1's concern about AI-assisted statistics: every run is
     seeded by run number, so these numbers regenerate identically from the public
     model + scripts. (Determinism itself is proven separately: two headless invocations
     of a seeded slice produce byte-identical output.)

History:
  2026-06-28 — original gate, against the UNPINNED spreadsheet deposit (13 anchors).
  2026-07-03 — REBUILT against the seeded table-format deposit after finding seeds were
     never pinned; the old deposit was one noisy n=30 draw. Anchors updated to seeded
     values; the refuted inverted-U "peak" anchors removed (see EXP-CM-S004 resume note).

Run:  source .venv/bin/activate && MPLBACKEND=Agg python notebooks/validate_baseline.py
Exit: 0 if all anchors pass, 1 otherwise.
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(__file__)
SEEDED = os.path.join(HERE, '..', 'exports', 'seeded')


def load(name):
    d = pd.read_csv(os.path.join(SEEDED, name), skiprows=6)
    d.columns = [c.strip('"') for c in d.columns]
    return d


def cascade(g):
    return 100.0 * (g['recovery-time'] == -1).mean() if len(g) else None


def diversity(g):
    return g['diversity-index'].mean() if len(g) else None


def sel(d, **kw):
    m = pd.Series(True, index=d.index)
    for k, v in kw.items():
        col = k.replace('_', '-')
        if isinstance(v, str):
            m &= (d[col] == v)
        else:
            m &= (d[col] - v).abs() < 1e-9
    return d[m]


def check(label, got, expected, tol):
    ok = got is not None and abs(got - expected) <= tol
    mark = "PASS" if ok else "FAIL"
    got_s = f"{got:.3f}" if got is not None else "  n/a"
    print(f"  [{mark}] {label:<54} got {got_s}  expected {expected:.3f} ±{tol}")
    return ok


def main():
    results = []

    # ── H001 (single perturbation): diversity collapse + no minority peak ─────
    print("\nH001 seeded (420 runs) — diversity collapse & stabilisation")
    h1 = load('H001_batch2_proportion_sweep_full-seeded.csv')
    ent = lambda ap: sel(h1, coordination_regime='entrainment', ai_proportion=ap)
    coh = lambda ap: sel(h1, coordination_regime='coherence', ai_proportion=ap)
    # Finding 1 — entrainment diversity collapses; coherence holds ~0.48
    results.append(check("entrainment diversity ~0.28 (0% AI)", diversity(ent(0.0)), 0.28, 0.04))
    results.append(check("entrainment diversity ~0.11 (90% AI)", diversity(ent(0.9)), 0.11, 0.04))
    results.append(check("coherence diversity ~0.48 (0% AI)", diversity(coh(0.0)), 0.48, 0.04))
    results.append(check("coherence diversity ~0.48 (90% AI)", diversity(coh(0.9)), 0.48, 0.04))
    # Stabilisation — cascade does NOT peak at a minority; low at high AI; coherence ~0
    results.append(check("entrainment cascade low-AI band ~20% (0% AI)", cascade(ent(0.0)), 20, 12))
    results.append(check("entrainment cascade declines by 90% AI (<=18%)", cascade(ent(0.9)), 8, 12))
    results.append(check("coherence cascade ~0% (50% AI)", cascade(coh(0.5)), 0, 5))

    # ── H002 (repeated stress): regime divergence ────────────────────────────
    print("\nH002 seeded (180 runs) — regime divergence under repeated stress")
    h2 = load('H002_batch3_repeated_stress-seeded.csv')
    e2 = lambda ap: sel(h2, coordination_regime='entrainment', ai_proportion=ap)
    c2 = lambda ap: sel(h2, coordination_regime='coherence', ai_proportion=ap)
    results.append(check("entrainment periodic cascade high (0% AI, ~57%)", cascade(e2(0.0)), 57, 15))
    results.append(check("entrainment periodic cascade high (20% AI, ~80%)", cascade(e2(0.2)), 80, 15))
    results.append(check("coherence periodic cascade ~0% (all AI tested)",
                         max(cascade(c2(0.0)), cascade(c2(0.2)), cascade(c2(0.5))), 0, 5))

    # ── H003 (mixed, periodic): identity-preservation band ───────────────────
    print("\nH003 seeded (900 runs) — regime-bias band (periodic stress)")
    h3 = load('H003_batch4_mixed_regime-seeded.csv')
    per = h3[h3['perturbation-regime'] == 'periodic']
    pb = lambda ap, b: cascade(sel(per, ai_proportion=ap, human_regime_bias_mean=b))
    results.append(check("0% AI periodic: cascade low at bias 0 (~3%)", pb(0.0, 0.0), 3, 7))
    results.append(check("0% AI periodic: cascade high at bias 1 (~57%)", pb(0.0, 1.0), 57, 12))
    results.append(check("20% AI periodic: cascade low at bias 0 (~3%)", pb(0.2, 0.0), 3, 7))
    results.append(check("20% AI periodic: cascade high at bias 1 (~50%)", pb(0.2, 1.0), 50, 14))

    n_pass = sum(results)
    print(f"\n{'='*62}\nGATE: {n_pass}/{len(results)} anchors reproduced from the SEEDED deposit.")
    if n_pass == len(results):
        print("PASS — seeded deposit reproduces the paper baseline. Reproducible by construction.")
        return 0
    print("FAIL — discrepancy vs the paper baseline. Resolve before any downstream claim.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
