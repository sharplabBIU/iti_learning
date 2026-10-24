"""Figures 1-4 of the manuscript, drawn from the pipeline outputs in data/study_N/analysis/results/ (main samples).

  Fig1_study1      task design (with Study 1's reward trajectories), learning, the ITI effect after wins and losses, and in
                   each half of the session
  Fig2_study2      design with the random-dot-motion task filling the ITI, the ITI effect after wins and losses and in each
                   half of the session, and why choices faster than 250 ms were treated as missed trials
                   (make_contamination_figure.py; needs the Study 2 fit of the model without an ITI term, m08)
  Fig3_study3      the ITI effect after wins and losses and in each half of the session, switching over the session with the
                   fixed ITI sequence, and the three accounts of the effect with the signature each predicts
  Fig4_modelling   the model sequence fitted to Study 3 by NUTS (06_model_fits), the checks behind each stage and of the ITI
                   models (10_ppc), parameter recovery of the winning model and model recovery among the three ITI models
                   (07_recovery), and the exploratory-state marker of Studies 3 and 4 (19_exploration_marker_ppc)
ITI = inter-trial interval. Numbers reported in the text that are computed here (for example the posterior predictive values
of Fig. 4b and the response-time shares of Fig. 2d-f) are written to figures/figure_numbers.json.
Run:  python3 figures/make_manuscript_figures.py [--figs 1 2 3 4]    (outputs: figures/Fig*.png at 300 dpi and Fig*.pdf)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE
sys.path.insert(0, str(REPO))
from analysis_common.data import load_study                     # noqa: E402
from analysis_common.behavior import switch_table                # noqa: E402
from analysis_common import config as C                          # noqa: E402
from analysis_common.likelihoods import PARAMS                   # noqa: E402

RES = lambda s: REPO / f"data/study_{s}/analysis/results"
COL = dict(short="#3B6FB6", long="#C8433F", win="#1B9E77", loss="#D95F02", grey="#8C8C8C", ink="#2B2B2B", light="#E8EAED")
KDIR, PPC_DIR, REC_DIR = "06_model_fits", "10_ppc", "07_recovery"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "pdf.fonttype": 42, "savefig.dpi": 300})
WIDTH = 7.2   # 183 mm, Nature double column
NUMBERS = {}  # numbers computed for the figures, written to figure_numbers.json
# Parameter symbols as in the Methods (alpha / alpha_r = learning rates after a win / a loss; decay = decay of unchosen values;
# alpha_c, beta_c = rate and weight of the choice kernel; omega = novelty bonus; beta_slope = exploration trend; kappa = the share of
# the inverse temperature moved to the wait)
PARAM_TXT = {"alpha": "α⁺", "alpha_r": "α⁻", "beta": "β", "decay": "d", "alpha_c": "α_C", "beta_c": "β_C", "omega": "ω",
             "beta_slope": "κ_β", "kappa": "κ", "gamma": "γ", "cost": "c", "stick": "s", "bias_slope": "κ_b"}
# Fig. 4: the stages of the model sequence and the colours of the models
SEQ_STAGES = [("A", "Value learning"), ("B", "Choice bias"), ("B2", "Novelty"), ("C", "Trend over the session"), ("D", "ITI")]
SEQ_DARK, SEQ_LIGHT = "#2B2B2B", "#BDBDBD"
PPC_MODEL_COL = {"data": "#2B2B2B", "m01_intercepts": "#BDBDBD", "m04_rw_2lr_decay": "#8C8C8C", "m05_stick": "#E69F00",
                 "m06_choice_kernel": "#56B4E9", "m07_novelty": "#7570B3", "m08_trend_beta": "#1B9E77", "m09_trend_bias": "#D95F02",
                 "m10_iti_bias": "#5E3C99", "m11_iti_target": "#F4A6A3", "m12_iti_compete": "#E7298A"}
FINAL_MODELS = ["m08_trend_beta", "m10_iti_bias", "m11_iti_target", "m12_iti_compete"]   # the final comparison
REC_GRID = "07_recovery/model_recovery_n50_iti3_grid"                                     # model recovery of the three ITI models
REC_MODELS = ["m10_iti_bias", "m11_iti_target", "m12_iti_compete"]


def J(s, rel):
    return json.load(open(RES(s) / rel))


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "p = n/a"
    if p < 1e-3:
        e = int(np.floor(np.log10(p))); m = p / 10 ** e
        return f"p = {m:.1f}×10$^{{{e}}}$" if m < 9.95 else f"p = 10$^{{{e + 1}}}$"
    return f"p = {p:.3f}"


def fmt_p_short(p):
    if p is None or not np.isfinite(p):
        return "p = n/a"
    if p < 1e-10:
        return "p < 10$^{-10}$"
    return fmt_p(p)


def letters(fig, items, dx=0.035, dy=0.006):
    """Panel letters at the top-left corner of each axes' bounding box (figure coordinates)."""
    for ax, s in items:
        b = ax.get_position()
        fig.text(b.x0 - dx, b.y1 + dy, s, fontsize=11, fontweight="bold", ha="right", va="bottom")


def panel_by_outcome(ax, s, title=None, legend=True):
    o = J(s, "02_switching/switching_by_outcome.json")
    NUMBERS[f"study{s}_by_outcome"] = {k: o[k] for k in ["switch_after_win", "switch_after_loss"]}
    x = np.array([0, 1])
    for key, col, lab, dxx, ty in [("switch_after_win", COL["win"], "after a win", -0.03, -0.09), ("switch_after_loss", COL["loss"], "after a loss", 0.03, 0.075)]:
        r = o[key]
        ax.errorbar(x + dxx, [r["mean_short"], r["mean_long"]], yerr=[r["sem_short"], r["sem_long"]], color=col, marker="o", ms=4, lw=1.6, capsize=2.5)
        ax.text(1.1, r["mean_long"] + ty, f"{lab}\n$d_z$ = {r['dz']:.2f}\n{fmt_p_short(r['p_two_sided'])}", color=col, fontsize=6.2, va="center", ha="left")
    ax.set_xticks(x); ax.set_xticklabels(["1 s", "4 s"]); ax.set_xlim(-0.35, 1.75); ax.set_ylim(0, 1)
    ax.set_xlabel("prior inter-trial interval"); ax.set_ylabel("p(switch)")
    if title:
        ax.set_title(title, pad=4)


def panel_by_half(fig, spec, s, title=None):
    """The by-outcome contrast in the first (trials 2-50) and the second half (51-100) of the session, descriptive
    (02b_iti_switching_phase_by_outcome.py: per participant p(switch) after a 1 s and a 4 s ITI, paired t tests). Two small axes in `spec`;
    returns the first, for the panel letter."""
    o = J(s, "02_switching/phase_by_outcome.json")
    NUMBERS[f"study{s}_by_half"] = o
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.12)
    axs = [fig.add_subplot(inner[0, 0])]; axs.append(fig.add_subplot(inner[0, 1], sharey=axs[0]))
    x = np.array([0, 1])
    for ax, (h, hlab) in zip(axs, (("early", "trials 2–50"), ("late", "trials 51–100"))):
        for key, col, lab, dxx, ty in [("after_win", COL["win"], "win", -0.03, -0.07), ("after_loss", COL["loss"], "loss", 0.03, 0.06)]:
            r = o[f"{h}_{key}"]
            ax.errorbar(x + dxx, [r["mean_short"], r["mean_long"]], yerr=[r["sem_short"], r["sem_long"]], color=col, marker="o", ms=3.5, lw=1.4, capsize=2)
            ax.text(1.1, r["mean_long"] + ty, f"Δ {r['diff_mean']:+.3f}\n{fmt_p_short(r['p_two_sided'])}".replace("-", "−"), color=col, fontsize=5.8,
                    va="center", ha="left")
        ax.set_xticks(x); ax.set_xticklabels(["1 s", "4 s"]); ax.set_xlim(-0.3, 1.85); ax.set_ylim(0, 1)
        ax.set_title(hlab, fontsize=7, pad=3)
    axs[0].set_ylabel("p(switch)"); axs[1].tick_params(labelleft=False)
    axs[0].text(0.03, 0.03, "after a win", color=COL["win"], fontsize=6, transform=axs[0].transAxes, va="bottom")
    axs[0].text(0.03, 0.97, "after a loss", color=COL["loss"], fontsize=6, transform=axs[0].transAxes, va="top")
    fig.text((axs[0].get_position().x0 + axs[1].get_position().x1) / 2, axs[0].get_position().y0 - 0.36 / fig.get_figheight(), "prior inter-trial interval",
             ha="center", va="top", fontsize=7.5)
    if title:
        fig.text((axs[0].get_position().x0 + axs[1].get_position().x1) / 2, axs[0].get_position().y1 + 0.24 / fig.get_figheight(), title, ha="center", va="bottom",
                 fontsize=8)
    return axs[0]


def panel_learning(ax, s, title=None):
    tr, P, _ = load_study(f"study_{s}", C.EXCLUSION_MAIN)
    v = tr[tr["valid"]]
    per = v.groupby(["trial", "subject_id"])["chose_best"].mean().groupby("trial").agg(["mean", "sem"])
    sm = per["mean"].rolling(5, center=True, min_periods=1).mean(); se = per["sem"].rolling(5, center=True, min_periods=1).mean()
    ax.fill_between(per.index, sm - se, sm + se, color=COL["short"], alpha=0.25, lw=0)
    ax.plot(per.index, sm, color=COL["short"], lw=1.4, label="p(chose best option)")
    ax.axhline(1 / 3, color=COL["grey"], ls=":", lw=0.9, label="chance (1/3)")
    L = J(s, "01_learning/learning_stats.json"); NUMBERS[f"study{s}_learning"] = L      # statistics in the text, not in the panel
    ax.set_xlabel("trial"); ax.set_ylabel("p(chose best option)"); ax.set_ylim(0.15, 0.92); ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]); ax.set_xlim(1, 100)
    if title:
        ax.set_title(title, pad=4)


def panel_switch_over_trials(ax, s=3, title=None):
    tr, P, _ = load_study(f"study_{s}", C.EXCLUSION_MAIN)
    tab = switch_table(tr); tab = tab[tab["consec1"]]
    per = tab.groupby(["trial", "subject_id"])["switch"].mean().groupby("trial").agg(["mean", "sem"])
    sched = tr.groupby("trial")["iti"].agg(lambda x: int(x.mode().iloc[0]))          # fixed schedule: interval FOLLOWING trial t
    prior = sched.shift(1).reindex(per.index)                                          # interval PRECEDING trial t
    cols = prior.map({1: COL["short"], 4: COL["long"]}).fillna(COL["grey"])
    ax.plot(per.index, per["mean"], color=COL["ink"], lw=0.8, alpha=0.35, zorder=1)
    ax.errorbar(per.index, per["mean"], yerr=per["sem"], fmt="none", ecolor=cols, elinewidth=0.6, alpha=0.5, zorder=2)
    ax.scatter(per.index, per["mean"], c=cols, s=14, zorder=3, edgecolor="none")
    roll = per["mean"].rolling(9, center=True, min_periods=3).mean()
    ax.plot(per.index, roll, color=COL["ink"], lw=1.8, zorder=4, label="9-trial moving average")
    ax.scatter([], [], c=COL["long"], s=14, label="prior interval 4 s"); ax.scatter([], [], c=COL["short"], s=14, label="prior interval 1 s")
    ax.set_xlim(0.5, 100.5); ax.set_ylim(0.2, 0.85); ax.set_xlabel("trial"); ax.set_ylabel("p(switch)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2, ncol=1)
    # the full interval schedule as a strip (interval that followed each trial)
    strip = ax.inset_axes([0, -0.40, 1, 0.07])
    strip.set_xlim(0.5, 100.5); strip.set_ylim(0, 1); strip.axis("off")
    for t, v in sched.items():
        strip.add_patch(plt.Rectangle((t - 0.5, 0), 1, 1, color=COL["long"] if v == 4 else COL["short"], lw=0))
    ax.text(0, -0.315, "fixed interval schedule (blue 1 s, red 4 s)", transform=ax.transAxes, fontsize=6.2, va="bottom", ha="left")
    q = sched.groupby(pd.cut(sched.index, [0, 20, 40, 60, 80, 100])).apply(lambda x: (x == 4).mean())
    NUMBERS[f"study{s}_schedule_pct_long_by_quintile"] = [round(100 * v) for v in q.values]
    NUMBERS[f"study{s}_switch_first_last_20"] = [float(per["mean"].iloc[:19].mean()), float(per["mean"].iloc[-20:].mean())]
    if title:
        ax.set_title(title, pad=4)


def panel_accounts(ax):
    """Fig. 3d: the three accounts of the effect side by side, each with the model-agnostic signature it predicts. Time cost in value:
    the wait is a cost charged to the value of the option it followed, so it works against what a win had just added, and the effect
    should be larger after wins. Time-scaled choice bias: the wait weakens the pull to repeat the last choice whatever it earned
    (value-independent, so the same effect after wins and losses). Global expectation of waits: carry-over to options other than the
    one the wait followed (tested with the decline in switching modelled; Supplementary Fig. 1). Laid out in points."""
    fig = ax.figure
    bb = ax.get_position(); W, H = bb.width * fig.get_figwidth() * 72, bb.height * fig.get_figheight() * 72
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def box(x, y, w, h, text, fc, tc="white", fs=6.0, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3", fc=fc, ec="none", zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=4,
                linespacing=1.15)

    def arr(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color="#555555", lw=0.8, zorder=2, shrinkA=0, shrinkB=0))

    cols = [
        ("Time cost in value", "(time as a cost, as in foraging)", [("choose A", "#2B2B2B"), ("win or loss", COL["grey"]), ("4 s wait", COL["long"])],
         "the wait is a cost charged to A:\nit works against what a win\nhad just added to A",
         "more leaving after a win\nthan after a loss", COL["long"]),
        ("Time-scaled choice bias", "(a fading trace of the last choice)", [("choose A", "#2B2B2B"), ("win or loss", COL["grey"]), ("4 s wait", COL["long"])],
         "the pull to repeat A weakens,\nwhatever A earned; the value\nof A is unchanged",
         "more leaving after a win\nand after a loss alike", "#7570B3"),
        ("Global expectation of waits", "(not modelled)", [("choose B", "#6B6B6B"), ("4 s wait", COL["long"]), ("choose A", "#2B2B2B")],
         "the expected wait rises,\nwhatever was chosen",
         "leave A, although the\nlong wait followed B", COL["short"]),
    ]
    sep = 14; cw = (W - 2 * sep) / 3
    ws = [0.29, 0.33, 0.29]; gap = cw * (1 - sum(ws)) / 2
    h1, h2, h3 = 18, 34, 26
    for k, (title, sub, seq, state, pred, pc) in enumerate(cols):
        x0 = k * (cw + sep); top = H - 2
        ax.text(x0 + cw / 2, top, title, ha="center", va="top", fontsize=6.9, fontweight="bold", color=COL["ink"])
        ax.text(x0 + cw / 2, top - 10, sub, ha="center", va="top", fontsize=6.0, color=COL["grey"])
        y1 = top - 25 - h1; x = x0
        for i, ((t, c), wf) in enumerate(zip(seq, ws)):
            w = wf * cw
            box(x, y1, w, h1, t, c, fs=5.8)
            if i < len(seq) - 1:
                arr((x + w + 1.0, y1 + h1 / 2), (x + w + gap - 1.0, y1 + h1 / 2))
            x += w + gap
        y2 = y1 - 11 - h2
        arr((x0 + cw / 2, y1 - 1.5), (x0 + cw / 2, y2 + h2 + 1.5))
        box(x0, y2, cw, h2, state, COL["light"], tc=COL["ink"], bold=False, fs=5.7)
        y3 = y2 - 11 - h3
        arr((x0 + cw / 2, y2 - 1.5), (x0 + cw / 2, y3 + h3 + 1.5))
        box(x0, y3, cw, h3, pred, pc, fs=5.9)
        ax.text(x0 + cw / 2, y3 - 4, "signature", ha="center", va="top", fontsize=5.6, color=COL["grey"], style="italic")
        if k < len(cols) - 1:
            ax.plot([x0 + cw + sep / 2] * 2, [y3 - 10, top], color="#DDDDDD", lw=0.6)


def _card(ax, x, y, w, h, color, alpha=1.0, chosen=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=COL["ink"] if chosen else "none",
                                lw=1.6 if chosen else 0, alpha=alpha, zorder=3))
    ax.text(x + w / 2, y + h / 2, "?", ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)


def _stage(ax, x, y, w, h, title, sub, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=fc, ec=COL["ink"], lw=1.0, zorder=2))
    ax.text(x + w / 2, y + h + 0.12, title, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y - 0.1, sub, ha="center", va="top", fontsize=6.3, color="#444444", linespacing=1.15)


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10, color=COL["ink"], lw=1.0, zorder=5))


def panel_task_schematic(ax, s, rdm=False):
    cards = ["#1B9E77", "#E69F00", "#7570B3"]
    y, h = 1.15, 1.25
    if not rdm:
        ax.set_xlim(0, 8.0); ax.set_ylim(0, 3.1); ax.axis("off")
        xs, w = [0.2, 2.2, 4.2], 1.6
    else:
        ax.set_xlim(0, 8.7); ax.set_ylim(0, 3.1); ax.axis("off")
        xs, w = [0.15, 1.7], 1.3
    # choice
    _stage(ax, xs[0], y, w, h, "Choice", "within 2 s\nkeys Q / T / P")
    for i, cc in enumerate(cards):
        _card(ax, xs[0] + 0.18 + i * (w - 0.36) / 3, y + 0.28, (w - 0.36) / 3 - 0.08, 0.7, cc)
    _arrow(ax, xs[0] + w + 0.04, xs[1] - 0.04, y + h / 2)
    # outcome
    _stage(ax, xs[1], y, w, h, "Outcome", "1.25 s\nwin \\$1 or lose \\$1")
    for i, cc in enumerate(cards):
        _card(ax, xs[1] + 0.18 + i * (w - 0.36) / 3, y + 0.28, (w - 0.36) / 3 - 0.08, 0.7, cc, alpha=0.35 if i != 1 else 1.0, chosen=(i == 1))
    ax.text(xs[1] + w / 2, y + 0.145, "+\\$1", ha="center", va="center", fontsize=6.3, color=COL["win"], fontweight="bold", zorder=6)
    if not rdm:
        _arrow(ax, xs[1] + w + 0.04, xs[2] - 0.04, y + h / 2)
        _stage(ax, xs[2], y, w, h, "Inter-trial interval", "fixed sequence\n(same for everyone)", fc="#111111")
        ax.text(xs[2] + w / 2, y + h / 2, "+", ha="center", va="center", fontsize=16, color="white", zorder=6)
        cx = xs[2] + w / 2
        ax.text(cx - 0.3, y - 0.62, "1 s", color=COL["short"], fontsize=8, fontweight="bold", ha="center", va="top")
        ax.text(cx, y - 0.62, "or", color=COL["ink"], fontsize=7.5, ha="center", va="top")
        ax.text(cx + 0.3, y - 0.62, "4 s", color=COL["long"], fontsize=8, fontweight="bold", ha="center", va="top")
        nx, wn = 6.2, 1.6; x_prev = xs[2] + w
    else:
        _arrow(ax, xs[1] + w + 0.04, 3.21, y + h / 2)
        xf, wf = 3.25, 0.7
        _stage(ax, xf, y, wf, h, "Fixation", "0.5 s", fc="#111111")
        ax.text(xf + wf / 2, y + h / 2, "+", ha="center", va="center", fontsize=14, color="white", zorder=6)
        _arrow(ax, xf + wf + 0.04, 4.16, y + h / 2)
        xr, wr = 4.2, 2.65
        _stage(ax, xr, y, wr, h, "Random-dot motion",
               "fills the whole interval: 1 s or 4 s\n500 dots, 7% coherent motion, up or down\ndown: press Q+T+P; up: withhold; no feedback", fc="#111111")
        cx, cy, r = xr + 0.62, y + h / 2, 0.52
        ax.add_patch(Circle((cx, cy), r, fc="#111111", ec="#777777", lw=0.8, zorder=3))
        rng = np.random.default_rng(3)
        ang = rng.uniform(0, 2 * np.pi, 160); rad = r * np.sqrt(rng.uniform(0, 1, 160))
        ax.scatter(cx + rad * np.cos(ang), cy + rad * np.sin(ang), s=1.6, color="white", zorder=4, lw=0)
        for k in range(6):
            px, py = cx + rad[k] * np.cos(ang[k]) * 0.8, cy + rad[k] * np.sin(ang[k]) * 0.8
            ax.add_patch(FancyArrowPatch((px, py + 0.12), (px, py - 0.12), arrowstyle="-|>", mutation_scale=6, color="#F28E2B", lw=0.8, zorder=5))
        ax.text(xr + 1.3, cy + 0.36, "net motion", fontsize=6.3, color="white", zorder=6, ha="left")
        ax.text(xr + 1.3, cy + 0.1, "downward → Q+T+P", fontsize=6.3, color="#F28E2B", zorder=6, ha="left")
        ax.text(xr + 1.3, cy - 0.16, "upward → withhold", fontsize=6.3, color="#9ecae1", zorder=6, ha="left")
        ax.text(xr + 1.3, cy - 0.44, "1 s or 4 s", fontsize=7, fontweight="bold", color="white", zorder=6, ha="left")
        nx, wn = 7.25, 1.3; x_prev = xr + wr
    _arrow(ax, x_prev + 0.04, nx - 0.04, y + h / 2)
    _stage(ax, nx, y, wn, h, "Next trial", "probabilities drift\n(restless bandit)")
    for i, cc in enumerate(cards):
        _card(ax, nx + 0.17 + i * (wn - 0.34) / 3, y + 0.28, (wn - 0.34) / 3 - 0.08, 0.7, cc, alpha=0.5)
    ax.add_patch(FancyArrowPatch((0.2, 0.22), (nx + wn, 0.22), arrowstyle="-|>", mutation_scale=12, color=COL["grey"], lw=1.2))
    ax.text((0.2 + nx + wn) / 2, 0.02, "time", ha="center", va="bottom", fontsize=7, color=COL["grey"], style="italic")


def panel_trajectories(ax, s, title=None):
    tr, P, _ = load_study(f"study_{s}", "none")
    traj = tr.groupby("trial")[["p1", "p2", "p3"]].median()
    NUMBERS[f"study{s}_trajectories_identical_across_participants"] = bool(tr.groupby("trial")["p1"].nunique().max() == 1)
    for i, cc in enumerate(["#1B9E77", "#E69F00", "#7570B3"]):
        ax.plot(traj.index, traj[f"p{i + 1}"], color=cc, lw=1.3, label=f"option {i + 1}")
    ax.set_ylim(0, 1.2); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlim(1, 100); ax.set_xlabel("trial"); ax.set_ylabel("p(win)")
    ax.legend(frameon=False, fontsize=5.7, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=3, columnspacing=0.5, handlelength=0.9,
              handletextpad=0.3, borderaxespad=0.1)
    if title:
        ax.set_title(title, pad=4)


def _rows(fig, heights_in, gaps_in, top_in=0.25, **kw):
    """vertical extents (bottom, top in figure fractions) of stacked rows given their heights and the gaps above each row, in inches"""
    H = fig.get_figheight(); y = H - top_in; out = []
    for h, g in zip(heights_in, gaps_in):
        y -= g; out.append(((y - h) / H, y / H)); y -= h
    return out


def fig1():
    """Fig. 1: a task; b reward probabilities and c learning, side by side; d the effect by previous outcome; e by outcome in each
    half of the session."""
    fig = plt.figure(figsize=(WIDTH, 8.1))
    (r1b, r1t), (r2b, r2t), (r3b, r3t) = _rows(fig, [1.95, 1.85, 2.0], [0.0, 0.55, 0.95], top_in=0.2)
    a = fig.add_axes([0.03, r1b, 0.94, r1t - r1b]); panel_task_schematic(a, 1)
    g2 = GridSpec(1, 2, figure=fig, wspace=0.3, left=0.085, right=0.985, bottom=r2b, top=r2t)
    b = fig.add_subplot(g2[0, 0]); panel_trajectories(b, 1, "Reward probabilities")
    c = fig.add_subplot(g2[0, 1]); panel_learning(c, 1, "Participants learned")
    g3 = GridSpec(1, 3, figure=fig, wspace=0.6, left=0.085, right=0.985, bottom=r3b, top=r3t)
    d = fig.add_subplot(g3[0, 0]); panel_by_outcome(d, 1, "By previous outcome")
    e = panel_by_half(fig, g3[0, 1:3], 1, "By previous outcome, in each half of the session")
    letters(fig, [(b, "b"), (c, "c"), (d, "d"), (e, "e")])
    fig.text(0.012, r1t + 0.004, "a", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.savefig(OUT / "Fig1_study1.png"); fig.savefig(OUT / "Fig1_study1.pdf"); plt.close(fig)


def fig2():
    """Fig. 2: a task; b the effect by previous outcome; c by outcome in each half of the session; d-f why choices faster than 250 ms
    were treated as missed trials (make_contamination_figure.py)."""
    import make_contamination_figure as CF
    fig = plt.figure(figsize=(WIDTH, 8.65))
    (r1b, r1t), (r2b, r2t), (r3b, r3t) = _rows(fig, [1.9, 2.0, 1.75], [0.0, 0.7, 1.05], top_in=0.2)
    a = fig.add_axes([0.03, r1b, 0.94, r1t - r1b]); panel_task_schematic(a, 2, rdm=True)
    g2 = GridSpec(1, 3, figure=fig, wspace=0.6, left=0.085, right=0.985, bottom=r2b, top=r2t)
    b = fig.add_subplot(g2[0, 0]); panel_by_outcome(b, 2, "By previous outcome")
    c = panel_by_half(fig, g2[0, 1:3], 2, "By previous outcome, in each half of the session")
    rdm = J(2, "11_rdm_task/rdm_stats.json")["accuracy_by_iti"]; NUMBERS["study2_rdm_accuracy"] = rdm    # still reported in the text
    g3 = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 1, 1.1], wspace=0.5, left=0.085, right=0.985, bottom=r3b, top=r3t)
    d = fig.add_subplot(g3[0, 0]); e = fig.add_subplot(g3[0, 1]); f = fig.add_subplot(g3[0, 2])
    dd, r1 = CF.compute(); nums = {}; CF.draw(d, e, f, dd, r1, nums); NUMBERS["study2_contamination"] = nums
    e.set_ylabel("model probability\nof the choice made")
    letters(fig, [(b, "b"), (c, "c")]); letters(fig, [(d, "d"), (e, "e"), (f, "f")], dx=0.055)
    fig.text(0.012, r1t + 0.004, "a", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.savefig(OUT / "Fig2_study2.png"); fig.savefig(OUT / "Fig2_study2.pdf"); plt.close(fig)


def fig3():
    """Fig. 3: a the effect by previous outcome and b by outcome in each half of the session, in the top row; c switching over the
    task with the fixed ITI sequence; d the three accounts and their signatures."""
    fig = plt.figure(figsize=(WIDTH, 8.3))
    (r1b, r1t), (r2b, r2t), (r3b, r3t) = _rows(fig, [2.0, 1.75, 2.05], [0.5, 0.85, 1.0], top_in=0.0)
    g1 = GridSpec(1, 3, figure=fig, wspace=0.6, left=0.085, right=0.985, bottom=r1b, top=r1t)
    a = fig.add_subplot(g1[0, 0]); panel_by_outcome(a, 3, "By previous outcome")
    b = panel_by_half(fig, g1[0, 1:3], 3, "By previous outcome, in each half of the session")
    c = fig.add_axes([0.085, r2b, 0.9, r2t - r2b]); panel_switch_over_trials(c, 3, "Switching over the task")
    d1 = fig.add_axes([0.085, r3b, 0.9, r3t - r3b]); panel_accounts(d1)
    letters(fig, [(a, "a"), (b, "b"), (c, "c")])
    fig.text(a.get_position().x0 - 0.035, r3t + 0.006, "d", fontsize=11, fontweight="bold", ha="right", va="bottom")
    fig.savefig(OUT / "Fig3_study3.png"); fig.savefig(OUT / "Fig3_study3.pdf"); plt.close(fig)


def _seq_tables(s=3):
    comp = pd.read_csv(RES(s) / f"{KDIR}/model_comparison_loo.csv")
    comp["stage_key"] = comp["stage"].str.split(".").str[0]
    fin = pd.read_csv(RES(s) / f"{KDIR}/final_comparison.csv")
    cj = J(s, f"{KDIR}/comparison.json")
    return comp, fin, cj


def _ppc_reps(s=3):
    allr = pd.read_csv(RES(s) / f"{PPC_DIR}/ppc_signatures_all_reps.csv")
    data = allr[allr.model == "data"].iloc[0]
    sim = allr[allr.model != "data"]
    summ = sim.groupby("model").agg(["mean", lambda x: np.percentile(x, 2.5), lambda x: np.percentile(x, 97.5)])
    return data, summ


def _ppc_val(summ, m, k):
    r = summ.loc[m, k]
    return float(r.iloc[0]), float(r.iloc[1]), float(r.iloc[2])


def panel_seq_overall(ax, s=3):
    """Fig. 4a, left: every model of the sequence compared overall, as elpd relative to the best model (not to the model it
    builds on); error bars, 2 SE of the paired difference from the best model; dark, the model carried forward at each stage."""
    comp, fin, cj = _seq_tables(s)
    best = str(comp["best_model"].iloc[0])
    chosen = {"m02_rw_1lr", "m03_rw_2lr", "m04_rw_2lr_decay", "m06_choice_kernel", "m07_novelty", "m08_trend_beta", best}
    rows, ypos, heads = [], [], []
    y = 0.0
    for key, name in [("A", "Value learning")] + SEQ_STAGES[1:]:
        sub = comp[comp.stage_key == key]
        if not len(sub):
            continue
        heads.append((y, name)); y += 1.0
        for _, r in sub.iterrows():
            rows.append(r); ypos.append(y); y += 1.0
        y += 0.35
    NUMBERS["study3_seq_overall"] = dict(best=best, rows=[dict(model=r.model, label=C.MODEL_SHORT[r.model], elpd=float(r.elpd_loo),
                                                               d=float(r.elpd_minus_best), se=float(r.se_vs_best)) for r in rows])
    lo = min(float(r.elpd_minus_best) for r in rows)
    for yy, r in zip(ypos, rows):
        dd, se = float(r.elpd_minus_best), float(r.se_vs_best)
        col = (PPC_MODEL_COL[r.model] if r.model in FINAL_MODELS[1:] else (SEQ_DARK if r.model in chosen else SEQ_LIGHT))
        ax.barh(yy, dd, height=0.72, color=col, zorder=2)
        if se > 0:
            ax.errorbar(dd, yy, xerr=2 * se, fmt="none", ecolor="#555555", elinewidth=0.8, capsize=1.8, zorder=3)
        txt = "best" if r.model == best else f"{dd:,.0f} ± {2 * se:.0f}".replace("-", "−")
        ax.text(1.01, yy, txt, va="center", ha="left", fontsize=5.9, color=COL["ink"], transform=ax.get_yaxis_transform(), zorder=5)
        ax.text(lo * 1.04, yy, C.MODEL_SHORT[r.model], va="center", ha="right", fontsize=5.9,
                color=COL["ink"] if r.model in chosen else "#6B6B6B", fontweight="bold" if r.model in chosen else "normal")
    for yy, name in heads:
        ax.text(lo * 1.04, yy, name, va="center", ha="right", fontsize=6.8, fontweight="bold", color=COL["ink"], style="italic")
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_ylim(y - 0.2, -0.8); ax.set_xlim(lo * 1.02, 0)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4000))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace("-", "−")))
    ax.set_xlabel(f"elpd relative to the best model, {C.MODEL_ABBR[best]} (± 2 SE)")


def panel_seq_final(ax, s=3):
    """Fig. 4a, right: the final comparison enlarged, the model without an ITI term and the three ITI models: elpd relative to the best (± 2 SE of the paired difference), with the protected exceedance probability among them and
    the number of participants each model fits best."""
    comp, fin, cj = _seq_tables(s)
    f = fin.set_index("model"); order = [m for m in FINAL_MODELS if m in f.index]
    NUMBERS["study3_seq_final"] = {m: dict(label=C.MODEL_SHORT[m], elpd_minus_base=float(f.loc[m, "elpd_minus_base"]), se_vs_base=float(f.loc[m, "se_vs_base"]),
                                           elpd_minus_best=float(f.loc[m, "elpd_minus_best"]), se_vs_best=float(f.loc[m, "se_vs_best"]),
                                           pxp=float(f.loc[m, "protected_exceedance"]), n_best=int(f.loc[m, "n_subjects_best"])) for m in order}
    NUMBERS["study3_seq_n"] = int(cj["n_subjects"])
    lo = min(float(f.loc[m, "elpd_minus_best"] - 2 * f.loc[m, "se_vs_best"]) for m in order)
    for i, m in enumerate(order):
        g, se = f.loc[m, "elpd_minus_best"], f.loc[m, "se_vs_best"]
        ax.barh(i, g, 0.6, color=PPC_MODEL_COL[m], zorder=2)
        if se > 0:
            ax.errorbar(g, i, xerr=2 * se, fmt="none", ecolor="#555555", elinewidth=0.8, capsize=2, zorder=3)
        if g == 0:
            ax.text(lo * 0.03, i, "best", ha="right", va="center", fontsize=5.8, color=COL["ink"])
        ax.text(1.16, i, f"{f.loc[m, 'protected_exceedance']:.2f}", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=5.8)
        ax.text(1.58, i, f"{int(f.loc[m, 'n_subjects_best'])}", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=5.8)
    ax.text(1.16, -0.62, "PXP", transform=ax.get_yaxis_transform(), ha="center", va="bottom", fontsize=5.8, color="#555555")
    ax.text(1.58, -0.62, "best\nfor n", transform=ax.get_yaxis_transform(), ha="center", va="bottom", fontsize=5.8, color="#555555", linespacing=1.0)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([C.MODEL_ABBR[m] for m in order], fontsize=6.0)
    ax.set_ylim(len(order) - 0.45, -0.55); ax.set_xlim(lo * 1.08, -lo * 0.04)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("elpd relative to the best\n(± 2 SE)", fontsize=6.5)
    ax.set_title("Final comparison, enlarged", fontsize=7.2, pad=22, loc="right")


def _dot(ax, x, m, summ, k, w=0.0):
    mu, lo, hi = _ppc_val(summ, m, k)
    ax.errorbar(x + w, mu, yerr=[[mu - lo], [hi - mu]], fmt="o", ms=3.2, color=PPC_MODEL_COL[m], elinewidth=0.9, capsize=0, zorder=3)
    return mu, lo, hi


def panels_seq_checks(axs, s=3):
    """b: the check that justifies each step (simulated from every participant's posterior means, 20 times)."""
    data, summ = _ppc_reps(s)
    cur = pd.read_csv(RES(s) / f"{PPC_DIR}/ppc_curves.csv")
    obs = cur[cur.model == "data"].set_index("w")
    sim = cur[cur.model != "data"].groupby(["model", "w"])
    ck = {}
    # b1 value learning: share of choices of the currently best option by 10-trial window
    ax = axs[0]; x = np.arange(10) * 10 + 5.5
    for m, ls in (("m01_intercepts", "--"), ("m04_rw_2lr_decay", "-")):
        mu = sim.chose_best.mean().loc[m]; lo = sim.chose_best.quantile(0.025).loc[m]; hi = sim.chose_best.quantile(0.975).loc[m]
        ax.fill_between(x, lo, hi, color=PPC_MODEL_COL[m], alpha=0.25, lw=0); ax.plot(x, mu, ls, color=PPC_MODEL_COL[m], lw=1.2, label=C.MODEL_ABBR[m])
        ck[f"chose_best_rmse_{m}"] = float(np.sqrt(((mu.to_numpy() - obs.chose_best.to_numpy()) ** 2).mean()))
    ax.plot(x, obs.chose_best, "o", color=PPC_MODEL_COL["data"], ms=2.8, label="data")
    ax.axhline(1 / 3, color=COL["grey"], ls=":", lw=0.8)
    ax.set_xlabel("trial"); ax.set_ylabel("p(chose best option)"); ax.set_ylim(0.12, 0.9); ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_title("Value learning", fontsize=7.5, pad=3)
    ax.legend(frameon=False, fontsize=5.4, loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=3, handlelength=0.9, columnspacing=0.5,
              handletextpad=0.25, borderaxespad=0.1)
    # b2 choice bias: switching after a stay and after a switch
    ax = axs[1]; ms = ["m04_rw_2lr_decay", "m05_stick", "m06_choice_kernel"]
    for j, (k, lab) in enumerate((("switch_after_stay", "after a stay"), ("switch_after_switch", "after a switch"))):
        ax.hlines(data[k], j - 0.38, j + 0.38, color=PPC_MODEL_COL["data"], ls="--", lw=0.9)
        for i, m in enumerate(ms):
            mu, lo, hi = _dot(ax, j, m, summ, k, w=(i - (len(ms) - 1) / 2) * 0.16)
            ck[f"{k}_{m}"] = mu
        ck[f"{k}_data"] = float(data[k])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["after\na stay", "after\na switch"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("p(switch)"); ax.set_title("Choice bias", fontsize=7.5, pad=3)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=PPC_MODEL_COL[m], ms=3.2, label=C.MODEL_ABBR[m]) for m in ms],
              frameon=False, fontsize=5.6, loc="upper left", handletextpad=0.2, borderaxespad=0.1)
    ax.set_ylim(0.3, 0.8)
    # b3 novelty: returns to the option chosen two trials back, after a switch
    ax = axs[2]; ms = ["m06_choice_kernel", "m07_novelty"]
    ax.axhline(data["return_two_back"], color=PPC_MODEL_COL["data"], ls="--", lw=0.9)
    for i, m in enumerate(ms):
        mu, lo, hi = _dot(ax, i, m, summ, "return_two_back"); ck[f"return_two_back_{m}"] = (mu, lo, hi)
    ck["return_two_back_data"] = float(data["return_two_back"])
    ax.set_xticks([0, 1]); ax.set_xticklabels([C.MODEL_ABBR[m].replace(" ", "\n") for m in ms]); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0.3, 0.55)
    ax.set_ylabel("p(return to option two back)"); ax.set_title("Novelty", fontsize=7.5, pad=3)
    ax.text(0.5, 0.97, "dashed: data", transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color="#555555")
    # b4 trend: switching by 10-trial window
    ax = axs[3]
    for m in ("m07_novelty", "m08_trend_beta", "m09_trend_bias"):
        mu = sim.p_switch.mean().loc[m]
        ax.plot(x, mu, "-", color=PPC_MODEL_COL[m], lw=1.2, label=C.MODEL_ABBR[m])
        ck[f"p_switch_rmse_{m}"] = float(np.sqrt(((mu.to_numpy() - obs.p_switch.to_numpy()) ** 2).mean()))
    ax.plot(x, obs.p_switch, "o", color=PPC_MODEL_COL["data"], ms=2.8, label="data")
    ck["p_switch_first_last"] = (float(obs.p_switch.iloc[0]), float(obs.p_switch.iloc[-1]))
    ax.set_xlabel("trial"); ax.set_ylabel("p(switch)"); ax.set_ylim(0.3, 0.7); ax.set_title("Trend over the session", fontsize=7.5, pad=3)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right", handlelength=1.4)
    NUMBERS["study3_seq_checks"] = ck


def panel_seq_interval(ax, s=3):
    """c: the effect of the interval on switching (4 s minus 1 s) within 10-trial windows, all trials, after wins, after losses."""
    data, summ = _ppc_reps(s)
    ms = [m for m in FINAL_MODELS if m in summ.index]
    ks = [("glmm_iti", "all trials"), ("glmm_iti_win", "after wins"), ("glmm_iti_loss", "after losses")]
    out = {"data": {k: float(data[k]) for k, _ in ks}}
    for j, (k, _) in enumerate(ks):
        ax.hlines(data[k], j - 0.4, j + 0.4, color=PPC_MODEL_COL["data"], ls="--", lw=0.9)
        for i, m in enumerate(ms):
            mu, lo, hi = _dot(ax, j, m, summ, k, w=(i - (len(ms) - 1) / 2) * 0.2)
            out.setdefault(m, {})[k] = (mu, lo, hi)
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in ks]); ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("ITI slope\n(log-odds per s)")
    ax.set_title("The ITI", fontsize=7.5, pad=3)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=PPC_MODEL_COL[m], ms=3.2, label=C.MODEL_ABBR[m]) for m in ms]
              + [plt.Line2D([], [], ls="--", color=PPC_MODEL_COL["data"], lw=0.9, label="data")],
              frameon=False, fontsize=5.8, loc="upper left", ncol=3, handletextpad=0.2, columnspacing=0.8)
    lo_ = min(min(v[1] for k2, v in out[m].items()) for m in ms if m in out); hi_ = max(max(v[2] for k2, v in out[m].items()) for m in ms if m in out)
    ax.set_ylim(min(lo_, min(out["data"].values())) - 0.02, max(hi_, max(out["data"].values())) + 0.075)
    NUMBERS["study3_seq_interval_ppc"] = out


def panel_seq_param_recovery(fig, spec, s=3, model=None):
    model = model or _seq_tables(s)[2].get("winner_elpd") or "m12_iti_compete"      # the winner of the final comparison
    NUMBERS["study3_seq_param_recovery_model"] = model
    f = RES(s) / f"{REC_DIR}/parameter_recovery_{model}.csv"
    if not f.exists():
        ax = fig.add_subplot(spec); ax.axis("off"); ax.text(0.5, 0.5, f"parameter recovery of {C.MODEL_SHORT[model]}: to be run", ha="center", va="center", fontsize=7, color="#777777")
        return ax
    d = pd.read_csv(f); params = PARAMS[model]
    sm = pd.read_csv(RES(s) / f"{REC_DIR}/parameter_recovery_{model}_summary.csv").set_index("parameter")
    NUMBERS["study3_seq_param_recovery"] = {p: float(sm.loc[p, "pearson_r"]) for p in sm.index}
    NUMBERS["study3_seq_param_recovery_n"] = int(d.subject_id.nunique())
    rr = pd.read_csv(RES(s) / f"{REC_DIR}/parameter_ranges.csv"); rr = rr[rr.model == model].set_index("parameter")
    inr = {}
    for p in params:
        g = d[(d.parameter == p) & d.generating.between(rr.loc[p, "fitted_min"], rr.loc[p, "fitted_max"])]
        if len(g) > 10:
            inr[p] = dict(r=float(np.corrcoef(g.generating, g.recovered)[0, 1]), n=int(len(g)), lo=float(rr.loc[p, "fitted_min"]), hi=float(rr.loc[p, "fitted_max"]))
    NUMBERS["study3_seq_param_recovery_in_fitted_range"] = inr
    ncol = 6; nrow = int(np.ceil(len(params) / ncol))
    sub = GridSpecFromSubplotSpec(nrow, ncol, subplot_spec=spec, hspace=0.75, wspace=0.55)
    first = None
    for i, p in enumerate(params):
        ax = fig.add_subplot(sub[i // ncol, i % ncol]); first = first or ax
        g = d[d.parameter == p]
        ax.axvspan(rr.loc[p, "fitted_min"], rr.loc[p, "fitted_max"], color="#E8E8E8", lw=0, zorder=0)
        ax.scatter(g.generating, g.recovered, s=1.5, alpha=0.35, color=COL["short"], lw=0)
        lo, hi = g.generating.min(), g.generating.max(); ax.plot([lo, hi], [lo, hi], color="#777777", lw=0.7, ls="--")
        ax.set_title(f"{PARAM_TXT.get(p, p)}  r = {sm.loc[p, 'pearson_r']:.2f}", fontsize=6.2, pad=2)
        ax.tick_params(labelsize=5.2, length=2)
    return first


def panel_seq_model_recovery(ax, s=3):
    """Fig. 4e: model recovery among the three ITI models (07_recovery_nuts.py, model_recovery_n50_iti3_grid): synthetic participants with parameters spread evenly over the full range of every parameter; every model fitted to every data set, and a
    data set is won by the model with the highest summed elpd (1 = the generating model wins, 0 otherwise). Points: elpd of the generating
    model minus the best of the others (± 2 SE of the participant-level difference); open points: a competing fit did not converge."""
    import glob
    fs = sorted(glob.glob(str(RES(s) / REC_GRID / "*_rep*.json")))
    if not fs:
        ax.axis("off"); ax.text(0.5, 0.5, "model recovery: to be run", ha="center", va="center", fontsize=7, color="#777777"); return
    recf = RES(s) / REC_GRID / "recovery.json"
    rec = json.load(open(recf)) if recf.exists() else {}
    models = REC_MODELS
    out = {m: [] for m in models}
    for f in fs:
        r = json.load(open(f)); g = r["generating"]
        others = [m for m in models if m != g and m in r["fits"]]
        if g not in models or g not in r["fits"] or not others:
            continue
        el = {m: np.asarray(r["fits"][m]["subject_elpd"]) for m in [g] + others}
        o = max(others, key=lambda m: el[m].sum())
        dd = el[g] - el[o]
        out[g].append(dict(rep=r["rep"], competitor=o, margin=float(dd.sum()), se=float(np.sqrt(len(dd)) * dd.std(ddof=1)),
                           won=bool(dd.sum() > 0), rhat_gen=float(r["fits"][g]["rhat_max"]),
                           rhat_other=float(max(r["fits"][m]["rhat_max"] for m in others))))
    NUMBERS["study3_seq_model_recovery"] = dict(models=models, n_subjects=rec.get("n_subjects", 50), reps=rec.get("reps", 5), datasets=out,
                                               recovery_rate=float(np.mean([d["won"] for v in out.values() for d in v])), theta_source="grid")
    top = 0.0
    for i, m in enumerate(models):
        for k, d in enumerate(out[m]):
            x = i + (k - (len(out[m]) - 1) / 2) * 0.09
            filled = d["rhat_other"] <= 1.1
            ax.errorbar(x, d["margin"], yerr=2 * d["se"], fmt="o", ms=3.2, color=PPC_MODEL_COL[m], mfc=PPC_MODEL_COL[m] if filled else "white",
                        elinewidth=0.8, capsize=0)
            top = max(top, d["margin"] + 2 * d["se"])
    for i, m in enumerate(models):
        if out[m]:
            n_won = sum(d["won"] for d in out[m])
            ax.text(i, top * 1.06, f"{n_won}/{len(out[m])}\nwon", ha="center", va="bottom", fontsize=5.8, linespacing=1.0)
    ax.axhline(0, color="#999999", lw=0.6)
    wrap = lambda t: t.replace(" (", "\n(").replace(" × ", "\n× ").replace("Explore trend", "Explore\ntrend").replace(" → ", "\n→ ").replace(" competes ", "\ncompetes ")
    wrap2 = lambda t: wrap(t).replace("choice bias", "choice\nbias").replace("learning target", "learning\ntarget").replace("with value", "with\nvalue")
    ax.set_xticks(range(len(models))); ax.set_xticklabels([wrap2(C.MODEL_ABBR[m]) for m in models], fontsize=5.4)
    ax.set_xlim(-0.6, len(models) - 0.4); ax.set_ylim(min(-20, min((d["margin"] - 2 * d["se"] for v in out.values() for d in v), default=0) * 1.1), top * 1.32)
    ax.set_ylabel("elpd, generating − other\n(± 2 SE)", fontsize=6.3); ax.set_xlabel("generating model", fontsize=6.5)
    ax.set_title("Model recovery", fontsize=7.5, pad=3)


def panels_seq_exploration(axs):
    """Fig. 4f: the exploratory-state marker, the ITI effect on switching by quintile of u = 1/(1 + n) (n a leaky count of the option's
    earlier choices, leak 0.85), after wins and losses, Studies 3 and 4: data (mean +- s.e.m.) and 20 simulated data sets of every model
    (mean and 95% range) (data/study_{3,4}/analysis/results/19_exploration_marker_ppc/)."""
    out = {}
    for j, s in enumerate((3, 4)):
        d = RES(s) / "19_exploration_marker_ppc"
        D = pd.read_csv(d / "marker.csv"); P = pd.read_csv(d / "ppc.csv")
        for k, o in enumerate(("win", "loss")):
            ax = axs[2 * j + k]; x = np.arange(1, 6)
            for n_, m in enumerate(FINAL_MODELS):
                p_ = P[(P.model == m) & (P.outcome == o)][[f"q{q}" for q in range(1, 6)]]
                if not len(p_):                       # a model not (yet) fitted in this study
                    continue
                xx = x + (n_ - (len(FINAL_MODELS) - 1) / 2) * 0.05
                ax.plot(xx, p_.mean(), color=PPC_MODEL_COL[m], lw=1.1, label=C.MODEL_ABBR[m])
                ax.fill_between(xx, p_.quantile(0.025), p_.quantile(0.975), color=PPC_MODEL_COL[m], alpha=0.12, lw=0)
            dd = D[D.outcome == o].sort_values("quintile")
            ax.errorbar(x, dd.effect, yerr=dd["sem"], fmt="o-", color=PPC_MODEL_COL["data"], ms=2.8, lw=1.3, capsize=1.5, label="data", zorder=5)
            ax.axhline(0, color="#BBBBBB", lw=0.6)
            ax.set_xticks(x); ax.set_xticklabels(["1\nexploit", "2", "3", "4", "5\nexplore"], fontsize=5.8)
            ax.set_title(f"Study {s}, after a {o}", fontsize=7.0, pad=3, color=COL["win" if o == "win" else "loss"])
            ax.set_xlabel("u = 1/(1 + n), quintile", fontsize=6.2)
            out[f"study{s}_{o}"] = dict(data=dd.effect.tolist())
    axs[0].set_ylabel("ITI effect on p(switch)\n(4 s − 1 s)", fontsize=6.5)
    lo = min(a.get_ylim()[0] for a in axs); hi = max(a.get_ylim()[1] for a in axs)
    for a in axs:
        a.set_ylim(lo, hi)
    for a in axs[1:]:
        a.tick_params(labelleft=False)
    NUMBERS["study3_seq_exploration"] = out


def fig4():
    """Fig. 4: a overall comparison and the final comparison; b the check behind each stage; c the ITI checks; d parameter recovery of
    the winning model; e model recovery among the three ITI models; f the exploratory-state marker (Studies 3 and 4).
    Layout in inches from the bottom of a 13.3 in figure."""
    H = 13.3; y = lambda inch: inch / H
    fig = plt.figure(figsize=(WIDTH, H))
    ga = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 0.8], wspace=1.3, left=0.30, right=0.895, top=y(12.96), bottom=y(9.75))
    a = fig.add_subplot(ga[0]); panel_seq_overall(a)
    a2 = fig.add_subplot(GridSpecFromSubplotSpec(2, 1, subplot_spec=ga[1], height_ratios=[1.25, 1.0], hspace=0.0)[1]); panel_seq_final(a2)
    gb = GridSpec(1, 4, figure=fig, wspace=0.55, left=0.075, right=0.985, top=y(8.95), bottom=y(7.6))
    bs = [fig.add_subplot(gb[i]) for i in range(4)]; panels_seq_checks(bs)
    gc = GridSpec(1, 1, figure=fig, left=0.1, right=0.74, top=y(6.76), bottom=y(5.5))
    c = fig.add_subplot(gc[0]); panel_seq_interval(c)
    leg = c.get_legend()
    if leg is not None:
        h, l = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
        leg.remove(); c.legend(handles=h, labels=l, frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(1.03, 1.0), ncol=1)
    ge = GridSpec(1, 2, figure=fig, width_ratios=[2.4, 1.0], wspace=0.28, left=0.075, right=0.985, top=y(4.85), bottom=y(3.05))
    d = panel_seq_param_recovery(fig, ge[0])
    e = fig.add_subplot(ge[1]); panel_seq_model_recovery(e)
    gf = GridSpec(1, 4, figure=fig, wspace=0.12, left=0.115, right=0.985, top=y(2.07), bottom=y(0.61))
    fs = [fig.add_subplot(gf[i]) for i in range(4)]; panels_seq_exploration(fs)
    hf, lf = fs[0].get_legend_handles_labels()
    fig.legend(hf, lf, frameon=False, fontsize=6.0, loc="upper center", ncol=len(lf), bbox_to_anchor=(0.53, y(2.5)))
    fig.text(0.012, y(12.96), "a", fontsize=11, fontweight="bold", va="top")
    for ax, l in ((bs[0], "b"), (c, "c"), (d, "d"), (e, "e"), (fs[0], "f")):
        b = ax.get_position(); fig.text(b.x0 - 0.055, b.y1 + 0.012, l, fontsize=11, fontweight="bold", va="bottom")
    fig.savefig(OUT / "Fig4_modelling.png"); fig.savefig(OUT / "Fig4_modelling.pdf"); plt.close(fig)



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="*", type=int, default=[1, 2, 3, 4], help="figures to draw (5: make_study4_figure.py; "
                    "6: make_generalization_figure.py)")
    args = ap.parse_args()
    fns = {1: fig1, 2: fig2, 3: fig3, 4: fig4}
    for k in args.figs:
        fns[k](); print("wrote", fns[k].__name__)
    nf = OUT / "figure_numbers.json"
    old = json.load(open(nf)) if nf.exists() else {}
    old.update(NUMBERS); json.dump(old, open(nf, "w"), indent=1, default=float)
    print("wrote", nf)
