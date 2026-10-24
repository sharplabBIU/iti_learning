"""Supplementary Fig. 2 (Supplementary Note 1): only the last ITI predicts switching, in the two preregistered studies.

Why the models use only the ITI just elapsed and not a running estimate of the expected ITI. For Studies 3 and 4, the
effect on switching of a 4 s against a 1 s ITI by how many trials back it came, each earlier ITI tested with the last ITI
and the linear trial term in the model (the preregistered mixed-effects model of the main text):
  switch ~ ITI k back (4 s vs 1 s) + last ITI (centred, per s) + trial + (1 + ITI k back + trial | participant)
  k = 2 on trials where the option chosen two back was the same as the last one (an expectation tied to the option), and
  k = 2, 3, 4 on trials where it differed (an expectation of the task as a whole).
The last ITI is the per-second coefficient of the k = 2 (other option) model times 3, i.e. 4 s against 1 s.
Points: posterior mean; bars: 95% credible interval (log-odds).
Inputs: data/study_{3,4}/analysis/results/16_position_controlled/glmm_*_fixef.csv and glmm_*.json
  (written by 16_position_controlled_contrasts.py).
Run: python3 figures/make_supp_fig2_lag_profile.py   (output: figures/SupplementaryFig2_lag_profile.png / .pdf; the numbers are
added to figures/figure_numbers.json, key "supp_fig2")
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = lambda s: HERE.parent / f"data/study_{s}/analysis/results/16_position_controlled"
OUT = HERE
LONG, INK, GREY = "#C8433F", "#2B2B2B", "#8C8C8C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 300})

# (label on the x axis, model file, term, scale to 4 s against 1 s)
ROWS = [("last\nITI", "twoback_different", "prev_iti_c", 3.0),
        ("2 back\nsame\noption", "twoback_same", "prev2_long", 1.0),
        ("2 back\nother\noption", "twoback_different", "prev2_long", 1.0),
        ("3 back\nother\noption", "threeback_different", "prev3_long", 1.0),
        ("4 back\nother\noption", "fourback_different", "prev4_long", 1.0)]


def estimates(s):
    out = []
    for label, model, term, scale in ROWS:
        f = pd.read_csv(RES(s) / f"glmm_{model}_fixef.csv").set_index("term").loc[term]
        meta = json.load(open(RES(s) / f"glmm_{model}.json"))
        out.append(dict(label=label, model=model, term=term, mean=scale * f["mean"], lo=scale * f["ci_low"],
                        hi=scale * f["ci_high"], n_obs=meta["n_obs"], n_participants=meta["n_groups"],
                        rhat_max=meta["rhat_max"], divergences=meta["n_divergences"]))
    return out


fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
numbers = {}
for ax, s in zip(axs, (3, 4)):
    est = estimates(s); numbers[f"study_{s}"] = est
    for i, e in enumerate(est):
        col = LONG if i == 0 else INK
        ax.errorbar(i, e["mean"], yerr=[[e["mean"] - e["lo"]], [e["hi"] - e["mean"]]], fmt="o", color=col, ms=4,
                    elinewidth=1.1, capsize=0)
    ax.axhline(0, color=GREY, lw=0.7)
    ax.axvline(0.5, color="#DDDDDD", lw=0.7, ls=":")
    ax.set_xticks(range(len(est))); ax.set_xticklabels([e["label"] for e in est])
    ax.set_xlim(-0.5, len(est) - 0.5)
    ax.set_title(f"Study {s} (n = {est[0]['n_participants']})", pad=4)
axs[0].set_ylabel("effect on switching of a\n4 s vs 1 s ITI (log-odds)")
fig.tight_layout()
for ax, l in zip(axs, "ab"):
    b = ax.get_position(); fig.text(b.x0 - (0.075 if l == "a" else 0.035), b.y1 + 0.02, l, fontsize=11, fontweight="bold")
fig.savefig(OUT / "SupplementaryFig2_lag_profile.png"); fig.savefig(OUT / "SupplementaryFig2_lag_profile.pdf")
nf = OUT / "figure_numbers.json"; allnums = json.load(open(nf)) if nf.exists() else {}
allnums["supp_fig2"] = numbers; json.dump(allnums, open(nf, "w"), indent=1, default=float)
for s, est in numbers.items():
    print(s, "; ".join(f"{e['label'].replace(chr(10), ' ')}: {e['mean']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]" for e in est))
