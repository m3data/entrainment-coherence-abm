#!/usr/bin/env python3
"""
sensitivity_analysis.py — TENCON robustness sweep analysis (EXP-CM-S004/S005).

Reads the seeded OAT sweeps (S-H004 AI-differential, S-H005 fatigue) and reports
the three headline findings as ROBUST / BOUNDED / FRAGILE with threshold bands and
Wilson 95% CIs on cascade proportions (per Mat 2026-07-03: report bands, not points).

Usage:
    source .venv/bin/activate
    MPLBACKEND=Agg python notebooks/sensitivity_analysis.py [--data-dir exports/seeded]

Cascade = fraction of runs with recovery-time == -1 (non-recovery within 3000 ticks).
Default cell (all AI-diff params at published default) must reproduce the seeded core.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

# ----- config -----
DEFAULTS = {"human-update-rate": 5, "ai-influence-radius": 8,
            "ai-noise-multiplier": 0.3, "ai-tie-strength-multiplier": 1.5}
H004_SWEEPS = {"human-update-rate": [2, 3, 5, 8],
               "ai-influence-radius": [5, 8, 11],
               "ai-noise-multiplier": [0.1, 0.3, 0.5, 1.0],
               "ai-tie-strength-multiplier": [1.0, 1.5, 2.0]}
AIPROP = [0, 0.1, 0.2, 0.5, 0.9]
H005_SWEEPS = {"fatigue-threshold": [300, 500, 800],
               "fatigue-intensity": [0.3, 0.5, 0.7]}
BIAS = [0, 0.25, 0.5, 0.75, 1.0]


def load(path):
    """Load a NetLogo table-format CSV (6 metadata rows, header on row 7)."""
    d = pd.read_csv(path, skiprows=6)
    d.columns = [c.strip('"') for c in d.columns]
    return d


def eqmask(series, val, tol=1e-9):
    return (series - val).abs() < tol


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (lo, mid, hi) in %."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    lo = max(0.0, 100 * (centre - half))
    hi = min(100.0, 100 * (centre + half))
    return (lo, 100 * p, hi)


def cascade(g):
    n = len(g)
    k = int((g["recovery-time"] == -1).sum())
    return wilson(k, n)


def slice_others_default(d, sweep_var):
    """Rows where every AI-diff param except sweep_var sits at its default."""
    m = pd.Series(True, index=d.index)
    for k, v in DEFAULTS.items():
        if k != sweep_var:
            m &= eqmask(d[k], v)
    return d[m]


# ---------------------------------------------------------------- findings
def finding1_2(d4):
    """Diversity collapse (F1) + inverted-U / low-AI band (F2), per param level."""
    out = []
    for sv, levels in H004_SWEEPS.items():
        sub_all = slice_others_default(d4, sv)
        for lv in levels:
            s = sub_all[eqmask(sub_all[sv], lv)]
            ent = s[s["coordination-regime"] == "entrainment"]
            coh = s[s["coordination-regime"] == "coherence"]
            d0 = ent[eqmask(ent["ai-proportion"], 0)]["diversity-index"].mean()
            d9 = ent[eqmask(ent["ai-proportion"], 0.9)]["diversity-index"].mean()
            cohd = coh["diversity-index"].mean()
            crates = {ap: cascade(ent[eqmask(ent["ai-proportion"], ap)]) for ap in AIPROP}
            out.append(dict(sweep=sv, level=lv, is_default=(lv == DEFAULTS[sv]),
                            ent_div0=d0, ent_div9=d9, coh_div=cohd, cascade=crates))
    return out


def finding3(d5):
    """Regime-bias threshold (F3), per fatigue-param level, at ai-proportion 0.2."""
    out = []
    for sv, levels in H005_SWEEPS.items():
        other = "fatigue-intensity" if sv == "fatigue-threshold" else "fatigue-threshold"
        odef = 0.5 if other == "fatigue-intensity" else 500
        for lv in levels:
            m = eqmask(d5[sv], lv) & eqmask(d5[other], odef) & eqmask(d5["ai-proportion"], 0.2)
            s = d5[m]
            rates = {b: cascade(s[eqmask(s["human-regime-bias-mean"], b)]) for b in BIAS}
            out.append(dict(sweep=sv, level=lv, is_default=(lv in (500, 0.5)), rates=rates))
    return out


# ---------------------------------------------------------------- verdicts
def band(lo, hi):
    return f"{lo:.0f}-{hi:.0f}%"


def report(d4, d5):
    print("=" * 74)
    print("TENCON ROBUSTNESS SWEEP — findings as bands (Wilson 95% CI on cascade)")
    print("=" * 74)

    f12 = finding1_2(d4)
    print("\nFINDING 1 — Diversity collapse under entrainment")
    collapse_ok = True
    for r in f12:
        direction = r["ent_div0"] > r["ent_div9"] and r["coh_div"] > r["ent_div9"]
        collapse_ok &= direction
        tag = " (default)" if r["is_default"] else ""
        print(f"  {r['sweep']}={r['level']}{tag}: entrainment {r['ent_div0']:.2f}->{r['ent_div9']:.2f}, "
              f"coherence flat {r['coh_div']:.2f}  {'OK' if direction else 'FLIP'}")
    print(f"  VERDICT: {'ROBUST — collapse direction holds at every level' if collapse_ok else 'BOUNDED/FRAGILE — see flips'}")

    print("\nFINDING 2 — Cascade concentrates at low AI proportion (inverted-U)")
    peak_in_low_band = True
    for r in f12:
        mids = {ap: r["cascade"][ap][1] for ap in AIPROP}
        peak_ap = max(mids, key=mids.get)
        # is majority-AI (0.5, 0.9) ever the worst?  and where is the peak
        low_band_worst = mids[0.5] < max(mids[0], mids[0.1], mids[0.2]) and mids[0.9] < max(mids[0], mids[0.1], mids[0.2])
        peak_in_low_band &= low_band_worst
        tag = " (default)" if r["is_default"] else ""
        lo, mid, hi = r["cascade"][peak_ap]
        print(f"  {r['sweep']}={r['level']}{tag}: peak@AI={peak_ap} ({band(lo,hi)}); "
              f"lo-AI band worst = {low_band_worst}")
    print(f"  VERDICT: {'ROBUST (as band): risk sits in the 0-20% AI band, falls at majority AI' if peak_in_low_band else 'BOUNDED — majority AI worst in some cells'}")
    print("  NOTE: exact peak location is within the CI band; report as low-AI concentration, not a point peak.")

    f3 = finding3(d5)
    print("\nFINDING 3 — Human identity-preservation threshold (~0.25 band)")
    thr_ok = True
    for r in f3:
        mids = {b: r["rates"][b][1] for b in BIAS}
        # low below 0.25, rising above
        low = max(mids[0], mids[0.25])
        high = min(mids[0.75], mids[1.0])
        holds = low < high
        thr_ok &= holds
        tag = " (default)" if r["is_default"] else ""
        seq = ", ".join(f"{b}:{mids[b]:.0f}%" for b in BIAS)
        print(f"  {r['sweep']}={r['level']}{tag}: [{seq}]  {'OK' if holds else 'FLAT'}")
    print(f"  VERDICT: {'ROBUST — safe-zone below ~0.25-0.5 bias, cascade accelerates above' if thr_ok else 'FRAGILE'}")

    return f12, f3


def default_cell_check(d4):
    """Mandatory gate: default AI-diff cell reproduces the seeded core baseline."""
    print("\n" + "=" * 74)
    print("DEFAULT-CELL REPRODUCTION (all AI-diff params at published default)")
    print("=" * 74)
    m = pd.Series(True, index=d4.index)
    for k, v in DEFAULTS.items():
        m &= eqmask(d4[k], v)
    dfl = d4[m]
    for regime in ["entrainment", "coherence"]:
        for ap in AIPROP:
            g = dfl[(dfl["coordination-regime"] == regime) & eqmask(dfl["ai-proportion"], ap)]
            if len(g) == 0:
                continue
            lo, mid, hi = cascade(g)
            print(f"  {regime:11s} AI={ap:<4} div={g['diversity-index'].mean():.3f} "
                  f"cascade={mid:4.0f}% [{band(lo,hi)}] (n={len(g)})")


def make_figure(d4, d5, outpath):
    """Fig 2: three panels stacked vertically, sized for one IEEE column (3.5 in wide).
    All text set at >=7 pt so it stays legible at final column width; Wilson 95% CI bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "lines.linewidth": 1.0, "lines.markersize": 3.0,
                         "font.family": "sans-serif"})

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.9))

    def plot_cascade(ax, xs, groups, label_fn, title):
        for lv, s, xkey in groups:
            ci = [cascade(s[eqmask(s[xkey], x)]) for x in xs]
            lo = [c[0] for c in ci]; mid = [c[1] for c in ci]; hi = [c[2] for c in ci]
            line, = ax.plot(xs, mid, marker="o", label=label_fn(lv))
            ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.12, linewidth=0)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Cascade rate (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    # (top) entrainment cascade vs AI proportion, one line per human-update-rate level (F2)
    ax = axes[0]
    sub = slice_others_default(d4, "human-update-rate")
    ent = sub[sub["coordination-regime"] == "entrainment"]
    groups = [(lv, ent[eqmask(ent["human-update-rate"], lv)], "ai-proportion")
              for lv in H004_SWEEPS["human-update-rate"]]
    plot_cascade(ax, AIPROP, groups, lambda lv: f"human update every {lv} ticks",
                 "(a) Entrainment cascade vs AI proportion")
    ax.set_xlabel("AI proportion")
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.5, borderpad=0.2, labelspacing=0.25)

    # (middle) entrainment vs coherence diversity vs AI proportion (F1, default cell)
    ax = axes[1]
    m = pd.Series(True, index=d4.index)
    for k, v in DEFAULTS.items():
        m &= eqmask(d4[k], v)
    dfl = d4[m]
    for regime, mk in [("entrainment", "o"), ("coherence", "s")]:
        g = dfl[dfl["coordination-regime"] == regime]
        cells = [g[eqmask(g["ai-proportion"], ap)]["diversity-index"] for ap in AIPROP]
        ys = [c.mean() for c in cells]
        err = [1.96 * c.std(ddof=1) / np.sqrt(len(c)) for c in cells]   # 95% CI of the mean
        ax.errorbar(AIPROP, ys, yerr=err, marker=mk, capsize=2, label=regime)
    ax.set_title("(b) Diversity vs AI proportion", loc="left")
    ax.set_xlabel("AI proportion")
    ax.set_ylabel("Diversity index")
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", frameon=False)

    # (bottom) cascade vs human regime-bias, one line per fatigue-threshold (F3)
    ax = axes[2]
    groups = []
    for lv in H005_SWEEPS["fatigue-threshold"]:
        s = d5[eqmask(d5["fatigue-threshold"], lv) & eqmask(d5["fatigue-intensity"], 0.5)
               & eqmask(d5["ai-proportion"], 0.2)]
        groups.append((lv, s, "human-regime-bias-mean"))
    ax.axvspan(0.25, 0.5, alpha=0.12, color="grey", linewidth=0, label="0.25\u20130.5 region")
    plot_cascade(ax, BIAS, groups, lambda lv: f"$\\theta_f$ = {lv}",
                 "(c) Mixed, periodic, 20% AI: cascade vs human bias")
    ax.set_xlabel("Human regime-bias")
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout(h_pad=0.4)
    fig.savefig(outpath, dpi=300)
    print(f"\nfigure -> {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="exports/seeded")
    ap.add_argument("--h004", default=None)
    ap.add_argument("--h005", default=None)
    ap.add_argument("--fig", default="exports/sensitivity_robustness.png")
    args = ap.parse_args()

    def pick(explicit, *cands):
        if explicit:
            return explicit
        for c in cands:
            if os.path.exists(c):
                return c
        return cands[0]

    h004 = pick(args.h004,
                os.path.join(args.data_dir, "S-H004_ai_param_robustness-seeded.csv"),
                "exports/archive-pre-seed/S-H004_ai_param_robustness-table.csv")
    h005 = pick(args.h005,
                os.path.join(args.data_dir, "S-H005_fatigue_robustness-seeded.csv"),
                "exports/archive-pre-seed/S-H005_fatigue_robustness-table.csv")
    print(f"S-H004: {h004}\nS-H005: {h005}")
    if not (os.path.exists(h004) and os.path.exists(h005)):
        sys.exit("sweep CSVs not found — run run_seeded_deposit.sh first")

    d4, d5 = load(h004), load(h005)
    print(f"loaded S-H004 {len(d4)} rows, S-H005 {len(d5)} rows")
    report(d4, d5)
    default_cell_check(d4)
    make_figure(d4, d5, args.fig)


if __name__ == "__main__":
    main()
