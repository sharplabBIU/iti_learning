"""02b - The inter-trial interval (ITI) effect on switching after wins and after losses, in the first and second half of the session.

This script runs for any study. The study is chosen by the folder the script is run from: a copy in
data/study_N/analysis/ analyses study N (analysis_common/runner.py reads the folder name).

Descriptive (Figs. 1e, 2c, 3b and 5b; Results, "early and late in the task"). The halves are trials 2-50 and
51-100, using choices on consecutive responded trials. For each half x previous outcome, each participant's
p(switch) after a 1 s and after a 4 s ITI is compared by a paired t test (two-sided). No model is fitted.

Inputs: data/trials.csv, main sample.
Outputs (results/02_switching/): phase_by_outcome.json, fig_phase_by_outcome.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (script sits in data/study_N/analysis/)
from analysis_common.runner import Runner, paired_bar, GREEN, RED
from analysis_common.behavior import switch_table, paired_iti_test

R = Runner(__file__)
R.banner("02b ITI effect after wins and after losses, by half of the session (descriptive)")
out = R.out("02_switching")
# Trial pairs with lagged predictors; keep pairs of directly consecutive responded trials.
trials, P = R.load()
tab = switch_table(trials); tab = tab[tab["consec1"]].copy()
# Paired 1 s vs 4 s test in each of the four cells: half (early/late) x previous outcome (prev_win 1/0).
# The early half starts at trial 2 because trial 1 has no previous choice.
HALVES = [("early", "trials 2-50"), ("late", "trials 51-100")]
OUTC = [(1, "after_win", "after a win"), (0, "after_loss", "after a loss")]
res = {}
for h, hlab in HALVES:
    for w, olab, oname in OUTC:
        r, _ = paired_iti_test(tab[(tab["half"] == h) & (tab["prev_win"] == w)], outcome="switch")
        res[f"{h}_{olab}"] = r
        print(f"  {hlab:<14} {oname:<13}: p(switch) {r['mean_short']:.3f} -> {r['mean_long']:.3f} (Δ {r['diff_mean']:+.3f}; "
              f"t({r['df']}) = {r['t']:.2f}, p = {r['p_two_sided']:.3g})")
R.save_json(out, "phase_by_outcome", res)

# 2 x 2 figure: rows = halves, columns = previous outcome. Dotted line = 2/3, indifference between the two
# other options, shown after losses.
plt = R.plt
fig, axes = plt.subplots(2, 2, figsize=(8, 7.6), sharey=True)
for i, (h, hlab) in enumerate(HALVES):
    for j, (w, olab, oname) in enumerate(OUTC):
        ax = axes[i, j]; col = GREEN if w else RED
        paired_bar(ax, res[f"{h}_{olab}"], colors=(col, col), title=f"{hlab}, {oname}")
        ax.set_ylim(0, 1); ax.set_xlabel("prior ITI")
        if not w:
            ax.axhline(2 / 3, color="grey", ls=":", lw=1)
fig.suptitle(f"{R.label}: ITI effect by previous outcome, in each half of the session"); fig.tight_layout()
fig.savefig(out / "fig_phase_by_outcome.png"); plt.close(fig)
print("wrote", out / "fig_phase_by_outcome.png")
