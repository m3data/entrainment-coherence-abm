#!/usr/bin/env python3
"""
paper_stats_seeded.py — single source of truth for every statistic cited in
TENCON §5 (Results), recomputed on the SEEDED, reproducible deposit.

Replaces numbers from the old unpinned draw. Report magnitudes as bands per
Mat 2026-07-03. Run: source .venv/bin/activate && python notebooks/paper_stats_seeded.py
"""
import numpy as np, pandas as pd
from scipy.stats import kruskal, mannwhitneyu, fisher_exact

SEED = "exports/seeded"
def load(name):
    d = pd.read_csv(f"{SEED}/{name}", skiprows=6); d.columns=[c.strip('"') for c in d.columns]; return d
def casc(g): return 100.0*(g['recovery-time']==-1).mean()
def n_casc(g): return int((g['recovery-time']==-1).sum()), len(g)
def rbc(u, n1, n2): return 2*u/(n1*n2) - 1  # rank-biserial from Mann-Whitney U

h1 = load("H001_batch2_proportion_sweep_full-seeded.csv")
h2 = load("H002_batch3_repeated_stress-seeded.csv")
h3 = load("H003_batch4_mixed_regime-seeded.csv")
APS = [0,0.1,0.2,0.3,0.5,0.7,0.9]   # all 7 H001 design levels (0.3/0.7 were previously dropped)
# Per-AI reporters (human-ai-cost-ratio, human-ai-work-ratio, ai-mean-*) sentinel to 0 when
# there are no AI agents; 0%-AI cells must be EXCLUDED from any statistic over them.
PER_AI = ['human-ai-cost-ratio','human-ai-work-ratio','ai-mean-cost','ai-mean-alignment-work','ai-max-fatigue','ai-mean-fatigue','ai-diversity-index']
def with_ai(df): return df[df['ai-proportion']>1e-9]
def per_ai_mean(df,col):
    assert col in PER_AI; return with_ai(df)[col].mean()


print("="*70); print("§5.1  DIVERSITY COLLAPSE UNDER ENTRAINMENT (H001)"); print("="*70)
ent = h1[h1['coordination-regime']=='entrainment']; coh = h1[h1['coordination-regime']=='coherence']
ent_groups = [ent[abs(ent['ai-proportion']-ap)<1e-9]['diversity-index'] for ap in APS]
coh_groups = [coh[abs(coh['ai-proportion']-ap)<1e-9]['diversity-index'] for ap in APS]
He,pe = kruskal(*ent_groups); Hc,pc = kruskal(*coh_groups)
print(f"entrainment diversity 0%->90% AI: {ent_groups[0].mean():.2f} -> {ent_groups[-1].mean():.2f}")
print(f"coherence diversity (flat): {np.mean([g.mean() for g in coh_groups]):.2f}")
print(f"KW entrainment across AI prop: H={He:.2f}, p={pe:.2e}")
print(f"KW coherence  across AI prop: H={Hc:.2f}, p={pc:.2f}")
rs=[]; ps=[]
for ap in APS:
    a=ent[abs(ent['ai-proportion']-ap)<1e-9]['diversity-index']; b=coh[abs(coh['ai-proportion']-ap)<1e-9]['diversity-index']
    u,p=mannwhitneyu(a,b); rs.append(rbc(u,len(a),len(b))); ps.append(p)
print(f"mode diff (MW) at each AI prop: max p={max(ps):.1e}, rank-biserial r in [{min(rs):.2f},{max(rs):.2f}]")

print("\n"+"="*70); print("§5.3  STABILISATION-DIVERSITY PARADOX (H001 single perturbation)"); print("="*70)
print("entrainment cascade & diversity by AI proportion:")
for ap in APS:
    g=ent[abs(ent['ai-proportion']-ap)<1e-9]
    print(f"  AI={ap:<4} cascade {casc(g):.0f}%  diversity {g['diversity-index'].mean():.2f}")
print("coherence cascade by AI proportion:", {ap: round(casc(coh[abs(coh['ai-proportion']-ap)<1e-9])) for ap in APS})
print("-> cascade roughly flat across low AI (0-20%), declines at majority AI; NO minority peak")

print("\n"+"="*70); print("§5.4  REPEATED STRESS — REGIME DIVERGENCE (H002 periodic)"); print("="*70)
for reg in ['entrainment','coherence']:
    r=h2[h2['coordination-regime']==reg]
    print(f"  {reg}: " + ", ".join(f"AI={ap}:{casc(r[abs(r['ai-proportion']-ap)<1e-9]):.0f}%" for ap in sorted(r['ai-proportion'].unique())))
ent2=h2[h2['coordination-regime']=='entrainment']
print(f"  entrainment mean cumulative cost: {ent2['mean-cumulative-cost'].mean():.0f}  max-fatigue mean: {ent2['max-fatigue-level'].mean():.2f}")
# human-ai cost ratio single (H001) vs periodic (H002) — 0%-AI cells excluded (reporter sentinels to 0)
coh2=h2[h2['coordination-regime']=='coherence']
for lab,df in [('single H001 ent',ent),('periodic H002 ent',ent2),('single H001 coh',coh),('periodic H002 coh',coh2)]:
    w=with_ai(df); per={ap: round(w[abs(w['ai-proportion']-ap)<1e-9]['human-ai-cost-ratio'].mean(),3) for ap in sorted(w['ai-proportion'].unique())}
    print(f"  human-ai-cost-ratio {lab}: per AI prop {per}; mean excl 0%AI = {per_ai_mean(df,'human-ai-cost-ratio'):.3f}")

print("\n"+"="*70); print("§5.5  IDENTITY-PRESERVATION THRESHOLD (H003 mixed)"); print("="*70)
per = h3[h3['perturbation-regime']=='periodic']; sing = h3[h3['perturbation-regime']=='single']
def fisher_bias(df, ap):
    g=df[abs(df['ai-proportion']-ap)<1e-9]
    k0,n0 = n_casc(g[abs(g['human-regime-bias-mean'])<1e-9])
    k1,n1 = n_casc(g[abs(g['human-regime-bias-mean']-1)<1e-9])
    _,p = fisher_exact([[k1,n1-k1],[k0,n0-k0]])
    return k0,n0,k1,n1,p
for ap in [0,0.2]:
    k0,n0,k1,n1,p = fisher_bias(per, ap)
    print(f"  periodic AI={ap}: cascade bias0 {100*k0/n0:.0f}% -> bias1 {100*k1/n1:.0f}%  (Fisher p={p:.1e})")
# diversity gradient at 20% AI single
g20=sing[abs(sing['ai-proportion']-0.2)<1e-9]
bias_groups=[g20[abs(g20['human-regime-bias-mean']-b)<1e-9]['diversity-index'] for b in [0,0.25,0.5,0.75,1.0]]
Hd,pd_=kruskal(*bias_groups)
print(f"  diversity vs bias @20%AI single: {bias_groups[0].mean():.2f}(bias0) -> {bias_groups[-1].mean():.2f}(bias1)  KW H={Hd:.1f}, p={pd_:.1e}")
# threshold: cascade & diversity at bias 0.25 across conditions
print("  cascade at bias=0.25 (periodic):", {ap: round(casc(per[(abs(per['ai-proportion']-ap)<1e-9)&(abs(per['human-regime-bias-mean']-0.25)<1e-9)])) for ap in [0,0.2,0.5]})
b25=pd.concat([sing[abs(sing['human-regime-bias-mean']-0.25)<1e-9],per[abs(per['human-regime-bias-mean']-0.25)<1e-9]])
print(f"  diversity at bias=0.25 (all): min {b25['diversity-index'].min():.2f}, mean {b25['diversity-index'].mean():.2f}")
# mixed bias0 vs binary coherence at 20%AI periodic
mix0 = casc(per[(abs(per['ai-proportion']-0.2)<1e-9)&(abs(per['human-regime-bias-mean'])<1e-9)])
coh2 = casc(h2[(h2['coordination-regime']=='coherence')&(abs(h2['ai-proportion']-0.2)<1e-9)])
print(f"  mixed bias0 @20%AI periodic: {mix0:.0f}% cascade vs binary coherence(H002) {coh2:.0f}%")
