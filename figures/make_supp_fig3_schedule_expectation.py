"""Supplementary Fig. 3 (Supplementary Note 2; model-free): a slow expectation of waiting is a function of trial number under the fixed,
front-loaded interval schedule shared by every participant. E_{t+1} = E_t + rho (w_t - E_t), w_t = -1 after a 1 s and +1 after a 4 s interval,
E_1 = 0, for learning rates rho from 0.01 to 1; the schedule is the modal interval after each trial in Study 3 (identical for every
participant, and the same in Studies 1 and 2). Panels: a, E over the session for four rates; b, the correlation of E with trial number
as a function of rho. Writes figures/SupplementaryFig3_schedule_expectation.png / .pdf; the numbers are added to
figures/figure_numbers.json (key "supp_fig3").
Run: python3 figures/make_supp_fig3_schedule_expectation.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300})
t = pd.read_csv(REPO / "data/study_3/data/trials.csv")
sched = t.groupby("trial")["iti"].agg(lambda x: x.value_counts().index[0]).sort_index()
w = np.where(sched.to_numpy() == 4, 1.0, -1.0); trials = sched.index.to_numpy()


def expectation(rho):
    E = np.zeros(len(w)); e = 0.0
    for i in range(len(w)):
        E[i] = e                       # expectation at the choice on trial i + 1 (intervals up to the previous trial)
        e += rho * (w[i] - e)
    return E


rates = np.exp(np.linspace(np.log(0.01), 0, 60))
cor = [float(np.corrcoef(expectation(r)[1:], trials[1:])[0, 1]) for r in rates]
show = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
nums = dict(pct_long_by_block=[round(100 * float((w[i:i + 20] > 0).mean())) for i in range(0, 100, 20)],
            correlation_with_trial={f"{r:g}": float(np.corrcoef(expectation(r)[1:], trials[1:])[0, 1]) for r in show},
            mean_first20_last20={f"{r:g}": [float(expectation(r)[1:21].mean()), float(expectation(r)[-20:].mean())] for r in show})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.8, 1], wspace=0.3))
for i in range(len(w)):
    a.axvspan(trials[i] - 0.5, trials[i] + 0.5, color="#C8433F" if w[i] > 0 else "#3B6FB6", alpha=0.08, lw=0)
shown = [0.02, 0.05, 0.1, 0.2]
cols = plt.cm.viridis(np.linspace(0.05, 0.8, len(shown)))
for r, c in zip(shown, cols):
    rr = nums["correlation_with_trial"][f"{r:g}"]
    a.plot(trials[1:], expectation(r)[1:], color=c, lw=1.6 if r == 0.05 else 1.0, label=f"ρ = {r:g}, r = {rr:.2f}")
a.set_xlim(1, 100); a.set_ylim(-0.75, 0.75); a.set_xlabel("trial"); a.set_ylabel("expected wait E (−1 = 1 s, +1 = 4 s)")
a.legend(frameon=False, fontsize=5.4, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.01), title="learning rate ρ, and the correlation r of E with trial number", title_fontsize=5.6,
         alignment="left", handlelength=1.0, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
b.plot(rates, cor, color="#2B2B2B", lw=1.3); b.set_xscale("log"); b.axhline(0, color="#999999", lw=0.6)
b.set_xlabel("learning rate ρ"); b.set_ylabel("correlation with trial"); b.set_ylim(-1, 0.2)
fig.subplots_adjust(left=0.09, right=0.985, top=0.8, bottom=0.17)
for ax, l in ((a, "a"), (b, "b")):
    fig.text(ax.get_position().x0 - 0.075, a.get_position().y1 + 0.13, l, fontsize=11, fontweight="bold")
fig.savefig(OUT / "SupplementaryFig3_schedule_expectation.png"); fig.savefig(OUT / "SupplementaryFig3_schedule_expectation.pdf"); plt.close(fig)
nf = OUT / "figure_numbers.json"; allnums = json.load(open(nf)) if nf.exists() else {}
allnums["supp_fig3"] = nums; json.dump(allnums, open(nf, "w"), indent=1)
print(json.dumps(nums, indent=1))
