"""Figure 5: Study 4, the replication with the ITI sequence reversed against the hypothesis (ITI = inter-trial interval).

  a  switching after a 1 s and a 4 s ITI by the previous outcome (panel function of make_manuscript_figures.py)
  b  the same contrast in each half of the session (02_switching/phase_by_outcome.json)
  c  the share of 4 s ITIs in successive blocks of twenty trials under the sequence of Studies 1-3 and under the Study 4
     sequence, with the correlation of a 4 s ITI with trial number (tasks/task_study4_newITIsequence/iti_schedule_study4.csv)
  d  the ITI effect of the mixed-effects model with the linear trial term in every study: the odds ratio of switching for
     4 s against 1 s over all trials, after wins and after losses (data/study_N/analysis/results/02_switching/glmm.json)
Main samples (sessions that reached trial 90). The plotted numbers are added to figures/figure_numbers.json (key "fig5").
Run: python3 figures/make_study4_figure.py   -> figures/Fig5_study4.png / .pdf
"""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import make_manuscript_figures as MF
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter

plt, COL = MF.plt, MF.COL
from analysis_common.data import load_study                     # noqa: E402  (path set up by make_manuscript_figures)
from analysis_common import config as C                          # noqa: E402

fig = plt.figure(figsize=(MF.WIDTH, 5.9))
(r1b, r1t), (r2b, r2t) = MF._rows(fig, [2.05, 1.95], [0.45, 1.0], top_in=0.0)
gs_top = GridSpec(1, 3, figure=fig, wspace=0.6, left=0.085, right=0.985, bottom=r1b, top=r1t)
gs_bot = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.85], wspace=0.42, left=0.085, right=0.985, bottom=r2b, top=r2t)
a = fig.add_subplot(gs_top[0, 0]); MF.panel_by_outcome(a, 4, "By previous outcome")

# ---- d: mixed-model odds ratios, every study ----------------------------------------------------------------------------------
d = fig.add_subplot(gs_bot[0, 1])
STUDIES = [(1, "Study 1\nunfilled"), (2, "Study 2\nfilled"), (3, "Study 3\nunfilled"), (4, "Study 4\nreversed")]
SERIES = [("all", ("trial_controlled",), "all trials", COL["ink"], -0.24), ("win", ("by_outcome_trial_controlled", "after_win"), "after a win", COL["win"], 0.0),
          ("loss", ("by_outcome_trial_controlled", "after_loss"), "after a loss", COL["loss"], 0.24)]
nums = {}
for i, (s, lab) in enumerate(STUDIES):
    gl = MF.J(s, "02_switching/glmm.json")
    for key, path, slab, col, dx in SERIES:
        r = gl[path[0]] if len(path) == 1 else gl[path[0]][path[1]]
        d.errorbar([i + dx], [r["odds_ratio"]], yerr=[[r["odds_ratio"] - r["or_ci_low"]], [r["or_ci_high"] - r["odds_ratio"]]], fmt="o",
                   color=col, ms=4, capsize=2, lw=1.2, label=slab if i == 0 else None)
        nums[f"study{s}_{key}"] = {k: round(float(r[k]), 4) for k in ("odds_ratio", "or_ci_low", "or_ci_high", "p_direction")}
d.axhline(1, color=COL["grey"], ls=":", lw=0.9)
d.set_yscale("log"); d.set_ylim(0.66, 1.9); d.set_yticks([0.8, 1.0, 1.2, 1.5, 1.8]); d.set_yticklabels(["0.8", "1.0", "1.2", "1.5", "1.8"])
d.yaxis.set_minor_formatter(NullFormatter())
d.set_xticks(range(len(STUDIES))); d.set_xticklabels([lab for _, lab in STUDIES])
d.set_xlim(-0.6, len(STUDIES) - 0.4)
d.set_ylabel("odds ratio of switching,\n4 s vs 1 s (mixed model\nwith linear trial term)")
d.legend(frameon=False, loc="lower left", fontsize=6.2, ncol=3, handletextpad=0.2, columnspacing=0.7, borderaxespad=0.2)
d.set_title("Interval effect in every study", pad=4)

# ---- c: the two interval sequences ----------------------------------------------------------------------------------------------
e = fig.add_subplot(gs_bot[0, 0])
BLOCKS = [0, 20, 40, 60, 80, 100]


def share_long(s):
    tr, _, _ = load_study(f"study_{s}", C.EXCLUSION_MAIN)
    sched = tr.groupby("trial")["iti"].agg(lambda x: int(x.mode().iloc[0]))
    return sched.groupby(pd.cut(sched.index, BLOCKS)).apply(lambda x: (x == 4).mean()).to_numpy(), sched


q3, sched3 = share_long(3); q4, sched4 = share_long(4)
sch = pd.read_csv(MF.REPO / "tasks/task_study4_newITIsequence/iti_schedule_study4.csv")     # the designed sequences (interval after trial t)
r3 = np.corrcoef(sch["trial"], (sch["iti_studies_1_3"] == 4).astype(float))[0, 1]
r4 = np.corrcoef(sch["trial"], (sch["iti"] == 4).astype(float))[0, 1]
x = np.arange(5); w = 0.38
e.bar(x - w / 2, q3, w, color=COL["grey"], label=f"Studies 1–3 (r = {r3:+.2f})".replace("-", "−"))
e.bar(x + w / 2, q4, w, color=COL["long"], label=f"Study 4 (r = {r4:+.2f})".replace("-", "−"))
e.axhline(0.5, color=COL["ink"], ls=":", lw=0.9)
e.set_xticks(x); e.set_xticklabels([f"{lo + 1}–\n{hi}" for lo, hi in zip(BLOCKS[:-1], BLOCKS[1:])], fontsize=6.2)
e.set_ylim(0, 1.02); e.set_xlabel("trials"); e.set_ylabel("share of 4 s intervals")
e.legend(frameon=False, loc="upper right", fontsize=6.2, title="r, 4 s interval with trial", title_fontsize=6.0, alignment="right",
         borderaxespad=0.1)
e.set_title("The two interval sequences", pad=4)
nums["share_long_by_block"] = {"studies_1_3": [round(float(v), 3) for v in q3], "study_4": [round(float(v), 3) for v in q4],
                               "corr_trial_long_studies_1_3": round(float(r3), 3), "corr_trial_long_study_4": round(float(r4), 3)}
nums["study4_by_outcome"] = MF.NUMBERS.get("study4_by_outcome")

h = MF.panel_by_half(fig, gs_top[0, 1:3], 4, "By previous outcome, in each half of the session")
nums["study4_by_half"] = MF.NUMBERS.get("study4_by_half")
MF.letters(fig, [(a, "a"), (h, "b"), (e, "c"), (d, "d")], dx=0.045)
fig.savefig(MF.OUT / "Fig5_study4.png"); fig.savefig(MF.OUT / "Fig5_study4.pdf"); plt.close(fig)
nf = MF.OUT / "figure_numbers.json"; allnums = json.load(open(nf)) if nf.exists() else {}
allnums["fig5"] = nums; json.dump(allnums, open(nf, "w"), indent=1, default=float)
print(json.dumps(nums, indent=1, default=float))
print("wrote", MF.OUT / "Fig5_study4.png")
