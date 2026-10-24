"""Draw panel a of Figure 6: the Tomov et al. (2020) fMRI two-armed bandit.

From the fit of tomov_iti_glmm.py, whose linear trend runs over the ten trials of each pair of arms:
  left   switching as a function of the inter-trial interval (ITI) before the choice, in quintiles of the ITI
         (mean +/- standard error of the mean, s.e.m., across participants), with the model's population-average
         curve (posterior-mean fixed + participant coefficients, averaged over the trial mix)
  right  switching over the ten trials of a pair of arms (mean +/- s.e.m. across participants) with the model's
         trend curve
The panel is drawn at the pixel width of Figure 6 (7.2 in at 300 dpi); figures/make_generalization_figure.py pastes it
at the top of the figure.

Inputs:  data/tomov2020/model_table.csv, results/tomov2020_iti_glmm_summary.json, results/tomov2020_iti_glmm_subject.csv
Output:  results/fig6a_tomov.png
Run:     python3 plot_fig6a_tomov.py (after tomov_iti_glmm.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; RES = ROOT / "results"
W, H = 2160, 702                                             # panel size in pixels (Figure 6 is 2160 px wide)
# The trial table, the fit summary (R) and the participant coefficients (S).
T = pd.read_csv(ROOT / "data/tomov2020/model_table.csv")
T["tb"] = (T.t - 1) % 10 + 1                                 # trial within the pair of arms (3-10 in the table)
# Only 3-10 occur: trial 1 has no preceding ITI and is dropped in preprocessing, so trial 2
# has no previous trial in the table either.
R = json.load(open(RES / "tomov2020_iti_glmm_summary.json"))
S = pd.read_csv(RES / "tomov2020_iti_glmm_subject.csv").set_index("pid")
b0, bi, bt = (S.loc[T.pid, c].to_numpy() for c in ("Intercept", "iti_c", "trial_c"))   # participant coefficients, per trial
med = T.groupby("pid").iti_pre.transform("median").to_numpy()   # participant median; iti_c = ITI - med
inv = lambda z: 1.0 / (1.0 + np.exp(-z))                     # logistic function: log-odds -> probability
sem = lambda v: v.std(ddof=1) / np.sqrt(len(v))

# Left panel: observed switching by ITI quintile (per participant and bin first, then mean +/- s.e.m. across
# participants; x = mean ITI in the bin), and the model curve. For each ITI value x on a 60-point grid the
# curve is the predicted p(switch) averaged over all trials, each with its participant's coefficients and its
# own trial_c.
edges = np.quantile(T.iti_pre, np.linspace(0, 1, 6)); T["bin"] = pd.cut(T.iti_pre, edges, include_lowest=True, labels=False)
pp = T.groupby(["pid", "bin"]).agg(sw=("is_shift", "mean"), x=("iti_pre", "mean")).reset_index()
g = pp.groupby("bin").agg(m=("sw", "mean"), se=("sw", sem), x=("x", "mean"))
xs = np.linspace(T.iti_pre.min(), T.iti_pre.max(), 60)
curve = np.array([inv(b0 + bi * (x - med) + bt * T.trial_c.to_numpy()).mean() for x in xs])
# Right panel: observed switching by trial within the pair of arms, and the model curve. For each trial t the
# curve averages the prediction over all trials, each with its own ITI and with the trend term set to t.
pt = T.groupby(["pid", "tb"]).is_shift.mean().reset_index()
gt = pt.groupby("tb").is_shift.agg(m="mean", se=sem)
ts = np.arange(3, 11)
curve_t = np.array([inv(b0 + bi * (T.iti_pre.to_numpy() - med) + bt * ((t - 5.5) / 10.0)).mean() for t in ts])

DPI = 200; fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
# Draw the panel: letter, title line with n, and the two axes with data, model curves and estimates.
INK, RED, GREY = "#222222", "#c0392b", "#9a9a9a"
fig.text(0.012, 0.965, "a", fontsize=16, fontweight="bold", color="black", va="top")
fig.text(0.53, 0.965, "Tomov et al. (2020): two-armed bandits with the interval before the choice jittered uniformly between 4.8 and 7.3 s "
         f"(n = {R['n_participants']}, {R['n_trials']:,} choices)", fontsize=10.5, color=INK, ha="center", va="top")
axL = fig.add_axes([0.095, 0.22, 0.36, 0.58]); axR = fig.add_axes([0.595, 0.22, 0.36, 0.58])
axL.errorbar(g.x, g.m, yerr=g.se, fmt="o", color=INK, ms=6, capsize=3, lw=1.5, zorder=3, label="data, quintiles of the ITI (mean ± s.e.m.)")
axL.plot(xs, curve, color=RED, lw=2, label="mixed-effects model")
axL.set_xlabel("prior inter-trial interval (s)", fontsize=10.5, color=INK); axL.set_ylabel("p(switch)", fontsize=10.5, color=INK)
axL.set_xlim(4.6, 7.5); axL.set_ylim(0.0, 0.35); axL.set_xticks([5, 6, 7])
axL.text(0.03, 0.96, f"b = {R['iti_c']['estimate']:+.3f} log-odds per second\n95% CrI [{R['iti_c']['ci_low']:+.3f}, {R['iti_c']['ci_high']:+.3f}], "
         f"posterior tail probability {R['iti_c']['p_direction']:.3f}\nodds ratio {R['iti_c']['or_median']:.2f} for a 3 s difference "
         f"[{R['iti_c']['or_ci_low']:.2f}, {R['iti_c']['or_ci_high']:.2f}]", transform=axL.transAxes, fontsize=8.5, color=INK, va="top")
axL.legend(loc="lower right", fontsize=8, frameon=False)
axL.set_title("Switching by the interval before the choice", fontsize=11, color=INK)
axR.errorbar(gt.index, gt.m, yerr=gt.se, fmt="o", color=INK, ms=6, capsize=3, lw=1.5, zorder=3)
axR.plot(ts, curve_t, color=RED, lw=2)
axR.set_xlabel("trial within a pair of arms", fontsize=10.5, color=INK); axR.set_ylabel("p(switch)", fontsize=10.5, color=INK)
axR.set_xlim(2.5, 10.5); axR.set_xticks(ts); axR.set_ylim(0.0, 0.6)
axR.text(0.97, 0.96, f"linear trend {R['trial_c']['estimate']:+.2f} log-odds per ten trials\n95% CrI [{R['trial_c']['ci_low']:+.2f}, {R['trial_c']['ci_high']:+.2f}]",
         transform=axR.transAxes, fontsize=8.5, color=INK, va="top", ha="right")
axR.set_title("Switching over the ten trials of a pair of arms", fontsize=11, color=INK)
for ax in (axL, axR):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9.5, colors=INK)
fig.savefig(RES / "fig6a_tomov.png", dpi=DPI)
print("wrote", RES / "fig6a_tomov.png")
print("ITI quintiles: " + ", ".join(f"{x:.2f} s: {m:.3f} ± {s:.3f}" for x, m, s in zip(g.x, g.m, g.se)))
print("by trial in the pair: " + ", ".join(f"t{t}: {m:.3f}" for t, m in zip(gt.index, gt.m)))
