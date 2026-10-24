"""Fig. 2d-f: in Study 2 some choices were late answers to the dot-motion display, not choices. compute() and draw() are
imported by make_manuscript_figures.fig2.

On a downward display participants answered by pressing Q, T and P together (the choice keys). When the display ended before the
three-key press, the press could land on the next choice screen, where the first of the three keys to go down registered as the choice
(Q = option 1, T = 2, P = 3; tasks/task_v2_rdm_study_2/public/index.html). Such choices are (a) impossibly fast, (b) unrelated to the
values the participant had learned, and (c) spread over the three keys regardless of their values, with switching at the chance level
of 2/3. They are removed from every Study 2 analysis by treating choices faster than 250 ms as missed trials (config rt_floor_ms).
All data here are the raw choices (apply_rt_floor=False), main sample (sessions that reached trial 90), consecutive responded pairs.
  d  response time of the next choice after a 1 s display: Study 2 after a missed downward display, Study 2 after every other display,
     and Study 1 (unfilled 1 s interval) for reference (share of choices per 25 ms bin)
  e  probability of the observed choice under the Study 2 model without an ITI term (m08, fitted with the floor by
     06_fit_models_nuts.py; each participant's posterior means run forward through every raw choice and outcome), by response-time
     bin; dotted line, 1/3
  f  share of choices on each key and p(switch) (diamonds) for fast (< 250 ms) and slower (>= 400 ms) choices
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from analysis_common.data import load_study
from analysis_common import config as C
import analysis_common.likelihoods as L

FIT = REPO / "data/study_2/analysis/results/06_model_fits/m08_trend_beta_subjects.csv"
BINS = [-1, 150, 250, 400, 1e9]; BLAB = ["< 150", "150–250", "250–400", "≥ 400"]


def prepared(key):
    t, _, _ = load_study(key, C.EXCLUSION_MAIN, apply_rt_floor=False)
    t = t.sort_values(["subject_id", "trial"]).copy(); g = t.groupby("subject_id")
    t["prev_trial"] = g["trial"].shift(1); t["prev_iti"] = g["iti"].shift(1); t["prev_choice"] = g["choice"].shift(1)
    t["prev_valid"] = g["valid"].shift(1)
    if "rdm_target_direction" in t:
        t["prev_dir"] = g["rdm_target_direction"].shift(1).astype(str).str.upper()
        t["prev_resp"] = g["rdm_user_response"].shift(1).astype(str).str.upper()
    return t


def compute():
    """The Study 2 choices after a dot-motion display (raw, with response-time bins and model probability), and Study 1 after a 1 s ITI."""
    s2 = prepared("study_2"); s1 = prepared("study_1")
    # model probability of the observed choice (m08 fitted with the floor), run through every raw responded choice
    fit = pd.read_csv(FIT).set_index("subject_id")
    pobs = {}
    for sid, g in s2[s2["valid"]].groupby("subject_id"):
        if sid not in fit.index:
            continue
        th = fit.loc[sid, [f"{p}_mean" for p in L.PARAMS["m08_trend_beta"]]].to_numpy(float)[None, :]
        pr = L.loglik("m08_trend_beta", th, g["choice"].to_numpy(int) - 1, g["reward"].to_numpy(float), g["iti"].to_numpy(float),
                      return_probs=True, trials=g["trial"].to_numpy(int))[0]
        for i, (idx, c) in enumerate(zip(g.index, g["choice"].to_numpy(int) - 1)):
            pobs[idx] = pr[i, c]
    s2["p_obs"] = pd.Series(pobs)

    d = s2[s2["valid"] & (s2["trial"] - s2["prev_trial"] == 1) & (s2["prev_valid"] == True)].copy()
    d = d[d["prev_resp"].isin(["UP", "DOWN"]) & d["prev_dir"].isin(["UP", "DOWN"])]
    d["missed_down"] = (d["prev_dir"] == "DOWN") & (d["prev_resp"] == "UP")
    d["switch"] = (d["choice"] != d["prev_choice"]).astype(int)
    d["rt_bin"] = pd.cut(d["rt"], BINS, labels=BLAB)
    r1 = s1[s1["valid"] & (s1["trial"] - s1["prev_trial"] == 1) & (s1["prev_valid"] == True) & (s1["prev_iti"] == 1)]
    return d, r1


def draw(a, b, c, d, r1, nums):
    """The three panels into axes a, b, c (Fig. 2d, e, f); numbers into nums."""
    # a: response times after a 1 s display
    edges = np.arange(0, 1025, 25)
    grp = [("Study 2: missed downward", d[(d.prev_iti == 1) & d.missed_down].rt, "#C8433F"),
           ("Study 2: other displays", d[(d.prev_iti == 1) & ~d.missed_down].rt, "#3B6FB6"),
           ("Study 1", r1.rt, "#8C8C8C")]
    for lab, rt, col in grp:
        h, _ = np.histogram(rt[rt < 1000], bins=edges)
        a.stairs(h / len(rt), edges, color=col, lw=1.1, label=lab)
        nums[f"a_{lab}"] = dict(n=int(len(rt)), share_lt_250=float((rt < 250).mean()), share_lt_50=float((rt < 50).mean()))
    a.axvspan(0, 250, color="#EEEEEE", lw=0, zorder=0)
    a.text(135, 0.055, "< 250 ms:\nremoved", ha="center", va="center", fontsize=5.8, color="#555555")
    a.set_xlim(0, 1000); a.set_xlabel("response time of the next choice (ms)"); a.set_ylabel("share of choices per 25 ms")
    a.set_title("After a 1 s interval", fontsize=8)
    a.legend(frameon=False, fontsize=5.8, loc="upper right", handlelength=1.2, borderaxespad=0.0)
    a.set_ylim(0, 0.085)
    # b: model probability of the observed choice
    for lab, mask, col in (("after a missed downward display", d.missed_down, "#C8433F"), ("after any other display", ~d.missed_down, "#3B6FB6")):
        x = d[mask].groupby("rt_bin", observed=False)["p_obs"].agg(["mean", "sem", "size"])
        b.errorbar(np.arange(4) + (0.08 if col == "#3B6FB6" else -0.08), x["mean"], yerr=x["sem"], fmt="o-", color=col, ms=3.5, lw=1,
                   capsize=0, label=lab)
        nums[f"b_{lab}"] = {k: dict(mean=float(v["mean"]), n=int(v["size"])) for k, v in x.iterrows()}
    b.axhline(1 / 3, color="#555555", ls=":", lw=0.9); b.text(3.4, 1 / 3 + 0.01, "1/3", fontsize=6.2, color="#555555", ha="right", va="bottom")
    b.set_xticks(range(4)); b.set_xticklabels(BLAB, fontsize=6.0, rotation=30, ha="right", rotation_mode="anchor"); b.set_xlabel("response time of the next choice (ms)")
    b.set_ylabel("model probability of the choice made"); b.set_ylim(0.25, 0.75)
    b.set_title("Fit to learned values", fontsize=8)
    b.legend(frameon=False, fontsize=5.6, loc="upper left", handlelength=1.2)
    # c: keys and switching, fast vs slower choices
    KC = {1: "#3B6FB6", 2: "#8C8C8C", 3: "#C8433F"}
    cells = [("< 250", d.missed_down & (d.rt < 250)), ("≥ 400", d.missed_down & (d.rt >= 400)),
             ("< 250", ~d.missed_down & (d.rt < 250)), ("≥ 400", ~d.missed_down & (d.rt >= 400))]
    for i, (lab, m) in enumerate(cells):
        x = d[m]; bottom = 0.0
        for k in (1, 2, 3):
            sh = float((x.choice == k).mean()); c.bar(i, sh, 0.7, bottom=bottom, color=KC[k], edgecolor="white", lw=0.5); bottom += sh
        c.plot(i, x.switch.mean(), "D", color="black", ms=3.4, zorder=4)
        c.text(i, 1.02, f"{len(x)}", ha="center", va="bottom", fontsize=6)
        nums[f"c_{['missed', 'missed', 'other', 'other'][i]}_{lab}"] = dict(n=int(len(x)), share_Q=float((x.choice == 1).mean()), share_T=float((x.choice == 2).mean()),
                                                      share_P=float((x.choice == 3).mean()), p_switch=float(x.switch.mean()))
    c.set_xticks(range(4)); c.set_xticklabels([l for l, _ in cells], fontsize=6.3); c.set_ylim(0, 1.12)
    c.text(0.5, -0.2, "after a missed\ndownward display", ha="center", va="top", fontsize=6.0, transform=c.get_xaxis_transform())
    c.text(2.5, -0.2, "after any other\ndisplay", ha="center", va="top", fontsize=6.0, transform=c.get_xaxis_transform())
    c.set_xlabel("response time (ms)", fontsize=6.5, labelpad=1)
    c.set_ylabel("share of choices on each key"); c.set_title("Which key registered", fontsize=8)
    c.legend(handles=[Patch(color=KC[1], label="Q"), Patch(color=KC[2], label="T"), Patch(color=KC[3], label="P"),
                      Line2D([], [], marker="D", ls="none", color="black", ms=3.4, label="p(switch)")],
             frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=4, handlelength=1.0, columnspacing=0.8)

