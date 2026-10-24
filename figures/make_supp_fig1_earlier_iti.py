"""Supplementary Fig. 1: does a longer inter-trial interval (ITI) after the choice k trials back predict more switching, on trials
where the option chosen k back differs from the option chosen last (a(t-k) != a(t-1)), for k = 2, 3 and 4 (one row each)? Every
level of the last ITI included; the k preceding trials responded and consecutive. Main samples (sessions that reached trial 90).

Drawn from the outputs of 16_position_controlled_contrasts.py in each study (results/16_position_controlled/):
  left   raw switching by trial type (ITI after the choice k back 1 s, blue, or 4 s, red), mean +/- s.e.m. across participants,
         per study; the number beside each point is the mean trial number of that trial type (the position confound)
         (raw_contrasts.csv, subset <k>back_different)
  right  the mixed-model odds ratio for 4 s against 1 s k back, posterior mean and 95% credible interval, per study:
         switch ~ prevK_long + prev_iti_c + trial_c + (1 + prevK_long + trial_c | participant), trial_c = (trial - 50.5) / 50
         (summary.json "glmm", keys twoback_different, threeback_different, fourback_different)
Run: python3 figures/make_supp_fig1_earlier_iti.py   (output: figures/SupplementaryFig1_earlier_ITI_other_option.png / .pdf)
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

SHORT, LONG, INK, INK2, GRID = "#3B6FB6", "#C8433F", "#2B2B2B", "#6b6b66", "#E8EAED"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42, "savefig.dpi": 300})
OUT = HERE
NAME = "SupplementaryFig1_earlier_ITI_other_option"
LAGS = [(2, "two", "twoback"), (3, "three", "threeback"), (4, "four", "fourback")]
STUDIES = [("study_1", "Study 1\nunfilled"), ("study_2", "Study 2\nfilled"), ("study_3", "Study 3\nunfilled"),
           ("study_4", "Study 4\nunfilled")]
XS = np.arange(len(STUDIES)); DX = 0.14


def load(key):
    """raw_contrasts.csv and the mixed-model records of script 16 for one study, or None if they are missing."""
    res = REPO / f"data/{key}/analysis/results/16_position_controlled"
    raw, summ = res / "raw_contrasts.csv", res / "summary.json"
    if not raw.exists() or not summ.exists():
        print(f"{key}: no script-16 outputs in {res}; marked 'not run'")
        return None
    gl = json.load(open(summ)).get("glmm")
    if not gl:
        print(f"{key}: {summ} has no 'glmm' key; marked 'not run'")
        return None
    return pd.read_csv(raw).set_index("subset"), gl


DATA = {key: load(key) for key, _ in STUDIES}

fig = plt.figure(figsize=(9.2, 9.4))
gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1.0], wspace=0.32, hspace=0.72, top=0.93, bottom=0.12, left=0.09, right=0.98)
axes = {}
or_all, p_all = [], []
for key, _ in STUDIES:
    if DATA[key] is not None:
        for _, _, word in LAGS:
            g = DATA[key][1].get(f"{word}_different")
            if g:
                or_all += [g["or_ci_low"], g["or_ci_high"]]
or_lim = (min([0.8] + or_all) / 1.04, max([1.25] + or_all) * 1.04)

for r, (k, kw, word) in enumerate(LAGS):
    sub = f"{word}_different"
    la, lb = "abcdef"[2 * r: 2 * r + 2]

    # left: raw switching by trial type, per study; mean trial number of each trial type beside its point
    ax = fig.add_subplot(gs[r, 0], sharey=axes.get("a")); axes.setdefault("a", ax)
    ticks, labelled = [], False
    for j, (key, lab) in enumerate(STUDIES):
        if DATA[key] is None or sub not in DATA[key][0].index:
            ax.text(j, 0.12, "not run", transform=ax.get_xaxis_transform(), ha="center", va="center", color=INK2, fontsize=7)
            ticks.append(lab); continue
        c = DATA[key][0].loc[sub]
        p_all += [c["mean_1s"] - c["sem_1s"], c["mean_1s"] + c["sem_1s"], c["mean_4s"] - c["sem_4s"], c["mean_4s"] + c["sem_4s"]]
        ax.plot([j - DX, j + DX], [c["mean_1s"], c["mean_4s"]], color=INK2, lw=0.8, zorder=1)
        for lv, col, mk, dx, ha in [("1s", SHORT, "o", -DX, "right"), ("4s", LONG, "s", DX, "left")]:
            ax.errorbar(j + dx, c[f"mean_{lv}"], yerr=c[f"sem_{lv}"], color=col, marker=mk, ms=5, lw=1.4, capsize=2.5, zorder=3,
                        label=None if labelled else f"{lv[0]} s")
            ax.text(j + dx + (0.07 if ha == "left" else -0.07), c[f"mean_{lv}"], f"{c[f'mean_trial_{lv}']:.0f}", ha=ha, va="center",
                    fontsize=6.5, color=INK2, style="italic")
        labelled = True
        ticks.append(f"{lab}\nn = {int(c['n'])}\nΔ {c['diff']:+.3f}")
        print(f"lag {k} {key}: raw 1 s {c['mean_1s']:.3f} ({c['sem_1s']:.3f}), 4 s {c['mean_4s']:.3f} ({c['sem_4s']:.3f}), "
              f"diff {c['diff']:+.4f} (s.e.m. {c['diff_sem']:.4f}), t({int(c['df'])}) = {c['t']:.2f}, p = {c['p_two_sided']:.3g}, "
              f"n {int(c['n'])}; mean trial 1 s {c['mean_trial_1s']:.1f}, 4 s {c['mean_trial_4s']:.1f}")
    ax.set_xticks(XS); ax.set_xticklabels(ticks, fontsize=6.8); ax.set_xlim(-0.6, len(STUDIES) - 0.4)
    ax.set_ylabel("p(switch)")
    ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    if labelled:
        ax.legend(frameon=False, ncol=2, loc="upper left", title=f"ITI after the choice {kw} back", title_fontsize=7,
                  handletextpad=0.3, columnspacing=1.0, alignment="left")
    ax.set_title(f"{la}  Switching by trial type (raw; numbers: mean trial)", loc="left")
    row_top = ax.get_position().y1

    # right: GLMM odds ratio, 4 s vs 1 s after the choice k back, with the linear decay over trials
    ax = fig.add_subplot(gs[r, 1])
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--", zorder=1)
    ticks = []
    for j, (key, lab) in enumerate(STUDIES):
        g = DATA[key][1].get(sub) if DATA[key] is not None else None
        if not g:
            ax.text(j, 0.12, "not run", transform=ax.get_xaxis_transform(), ha="center", va="center", color=INK2, fontsize=7)
            ticks.append(lab); continue
        ax.errorbar(j, g["odds_ratio"], yerr=[[g["odds_ratio"] - g["or_ci_low"]], [g["or_ci_high"] - g["odds_ratio"]]], fmt="o", color=INK,
                    ms=5, capsize=2.5, lw=1.4, zorder=3)
        ax.text(j + 0.09, g["odds_ratio"], f"{g['odds_ratio']:.2f}", ha="left", va="center", fontsize=6.5, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=4)
        ticks.append(f"{lab}\n[{g['or_ci_low']:.2f}, {g['or_ci_high']:.2f}]")
        print(f"lag {k} {key}: GLMM {g['formula']}: b {g['estimate']:+.4f} [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}], OR {g['odds_ratio']:.3f} "
              f"[{g['or_ci_low']:.3f}, {g['or_ci_high']:.3f}], rhat_max {g['rhat_max']:.3f}, "
              f"n_obs {g['n_obs']}, participants {g['n_groups']}")
        if g["rhat_max"] > 1.1:
            print(f"  WARNING: {key} {sub} rhat_max {g['rhat_max']:.3f} > 1.1")
    ax.set_yscale("log"); ax.set_ylim(*or_lim)
    yt = [v for v in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0) if or_lim[0] <= v <= or_lim[1]]
    ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.set_xticks(XS); ax.set_xticklabels(ticks, fontsize=6.8); ax.set_xlim(-0.6, len(STUDIES) - 0.4)
    ax.set_ylabel(f"odds ratio, {kw} back 4 s vs 1 s")
    ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_title(f"{lb}  Mixed-effects model with a linear trend over the session", loc="left")
    fig.text(0.02, row_top + 0.033, f"Choice {kw} trials back differs from the last choice: the ITI after it, 4 s against 1 s",
             fontsize=9.5, fontweight="bold", color=INK, ha="left", va="bottom")

if p_all:                                                           # headroom above the data for the legends (y shared across rows)
    lo, hi = min(p_all), max(p_all)
    axes["a"].set_ylim(lo - 0.08 * (hi - lo), hi + 0.45 * (hi - lo))
fig.savefig(OUT / f"{NAME}.png", bbox_inches="tight"); fig.savefig(OUT / f"{NAME}.pdf", bbox_inches="tight")
print("wrote", OUT / f"{NAME}.png")
