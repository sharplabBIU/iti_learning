"""Figure 6: generalization, the ITI effect in an external dataset and in AI learners that do not experience time (ITI =
inter-trial interval).

  a  the Tomov et al. (2020) panel, drawn by fMRI_study_generalization/code/plot_fig6a_tomov.py and pasted pixel for pixel
  b  one trial as an agent had it, verbatim from its run record (ai_agents/runs/AIAGENT_told_sonnet5_timecost1_005.json,
     trial 15, a win followed by a 4 s ITI): the screenshot it was sent (only on its first practice and first task trial;
     ai_agents/runs/images/), the message it read, its reasoning summary as returned by the API, and its reply, which the
     harness turned into a key press on the live page; plus the count of reasoning summaries, over all 14 agents, that
     mention the wait
  c  switching after a win, 1 s against 4 s: Study 4 participants and the agents
  d  the same after a loss
  e  switching over the session in 10-trial bins
  f  choosing the best option over the session in 10-trial bins, Study 4 participants and the agents
The numbers are added to figures/figure_numbers.json (key "fig6").
Run: python3 figures/make_generalization_figure.py   -> figures/Fig6_generalization.png / .pdf
(after ai_agents/analyze_agents.py and plot_fig6a_tomov.py)
"""
import json
import re
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from PIL import Image
import make_manuscript_figures as MF
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch

plt, COL = MF.plt, MF.COL
from analysis_common.data import load_study, tidy_trials         # noqa: E402  (path set up by make_manuscript_figures)
from analysis_common.behavior import switch_table                # noqa: E402
from analysis_common import config as C                          # noqa: E402

AI = MF.REPO / "ai_agents"
PANEL_A = MF.REPO / "fMRI_study_generalization/results/fig6a_tomov.png"
RUN = AI / "runs/AIAGENT_told_sonnet5_timecost1_005.json"; TRIAL = 15
SHOT = AI / "runs/images/AIAGENT_told_sonnet5_timecost1_005_trial.png"
COND = [("time_blind", "agents,\ntime-blind"), ("time_told", "agents,\ntime-told")]
TICK = {"time_blind": "agents\n(blind)", "time_told": "agents\n(told)"}   # short bar labels for the narrow panels c, d
INSTR_MARK = {"none": ("o", "no extra instruction"), "Play like a human would.": ("^", "“Play like a human would.”"),
              "Play like a human that is sensitive to time costs.": ("s", "“… sensitive to time costs.”")}
DPI = int(plt.rcParams["savefig.dpi"]); nums = {}

# ---- the example trial and the reasoning summaries ----------------------------------------------------------------------------------
run = json.load(open(RUN))
ex = next(e for e in run["exchanges"] if f"\nTRIAL {TRIAL}\n" in e["prompt"])
assert "You won $1" in ex["prompt"] and "for 4 seconds" in ex["prompt"], ex["prompt"]
reply = json.loads(ex["raw_reply"])["key"]; thought = ex["thinking_summary"].strip()
n_sum = n_time = 0
for f in sorted((AI / "runs").glob("AIAGENT_*.json")):
    if "_mock_" in f.name:
        continue
    for e in json.load(open(f))["exchanges"]:
        t = e.get("thinking_summary") or ""
        if t:
            n_sum += 1; n_time += bool(re.search(r"\bwait|\bsecond|fixation|\bcross\b|\bdelay|\bpause|\binterval|\bITI\b", t, re.I))
nums.update(example=dict(agent=run["pid"], model=run["model"], condition=run["condition"], instruction=run["persona"], trial=TRIAL,
                         reply=reply, reasoning=thought), reasoning_summaries=n_sum, reasoning_mentioning_the_wait=n_time)

# ---- layout: panel a is pasted into a blank band at the top --------------------------------------------------------------------------
pa = Image.open(PANEL_A).convert("RGB"); W, HA = pa.size
assert W == round(MF.WIDTH * DPI), (W, MF.WIDTH * DPI)
top_in = HA / DPI; H = top_in + 5.35
fig = plt.figure(figsize=(MF.WIDTH, H))
y = lambda inch_from_top: 1 - inch_from_top / H

# ---- b: one trial as the agent had it -------------------------------------------------------------------------------------------------
bx = fig.add_axes([0.0, y(top_in + 2.25), 1.0, 2.0 / H]); bx.set_xlim(0, 7.2); bx.set_ylim(0, 2.0); bx.axis("off")
INK, MUTE = COL["ink"], "#5B6472"
bx.text(0.12, 1.93, f"One trial as an agent had it: Sonnet 5, time-told, “sensitive to time costs” instruction; trial {TRIAL}, a win then a 4 s ITI",
        fontsize=7.0, color=INK, va="top")
heads = [(0.12, "What it saw"), (1.62, "What it read, every trial"), (4.05, "What it reasoned"), (5.85, "What it did")]
for x0, h in heads:
    bx.text(x0, 1.66, h, fontsize=7.0, fontweight="bold", color=INK, va="top")
# 1. the screenshot
ia = fig.add_axes([0.12 / 7.2, y(top_in + 0.25 + 0.95), 1.32 / 7.2, 0.46 / H]); ia.imshow(Image.open(SHOT)); ia.axis("off")
bx.text(0.12, 0.62, "an image of the screen, sent on\n2 of its 114 turns: the first practice\nand the first task trial only",
        fontsize=5.8, color=MUTE, va="top", linespacing=1.25)
# 2. the message
bx.add_patch(FancyBboxPatch((1.62, 0.18), 2.12, 1.3, boxstyle="round,pad=0.0,rounding_size=0.06", fc="#1B2433", ec="none"))
lines = [("You pressed T (middle image). You won $1", "#F3EFE6"), ("", None),
         ("The images disappear and a fixation cross (+)", "#F3EFE6"), ("is shown for 4 seconds.", "#F0A35E"), ("", None),
         (f"TRIAL {TRIAL}", "#F3EFE6"), ("", None),
         ("Three images are shown side by side. The key", "#F3EFE6"), ("shown under each image selects it:", "#F3EFE6"),
         ("Q = left, T = middle, P = right.", "#F3EFE6"), ("", None), ("Reply with the key you press: Q, T or P.", "#F3EFE6")]
yy = 1.42
for txt, c in lines:
    if txt:
        bx.text(1.69, yy, txt, fontsize=5.3, family="monospace", color=c, va="top")
    yy -= 0.098 if txt else 0.05
bx.text(1.62, 0.12, "time-blind agents: the same, without “for 4 seconds”", fontsize=5.8, color=MUTE, va="top")
# 3. the reasoning
bx.add_patch(FancyBboxPatch((4.05, 0.62), 1.55, 0.86, boxstyle="round,pad=0.0,rounding_size=0.06", fc="#F3F1EC", ec="#D9D5CC", lw=0.6))
words, wrapped, cur = thought.split(), [], ""
for w_ in words:
    if len(cur) + len(w_) + 1 > 34:
        wrapped.append(cur); cur = w_
    else:
        cur = (cur + " " + w_).strip()
wrapped.append(cur)
bx.text(4.12, 1.42, "“" + "\n".join(wrapped) + "”", fontsize=6.0, style="italic", color=INK, va="top", linespacing=1.3)
bx.text(4.05, 0.52, f"reasoning summaries that mention\nthe wait: {n_time} of {n_sum} (all 14 agents)", fontsize=6.2, color=COL["long"],
        fontweight="bold", va="top", linespacing=1.25)
# 4. the action
bx.add_patch(FancyBboxPatch((5.85, 1.12), 1.15, 0.36, boxstyle="round,pad=0.0,rounding_size=0.06", fc="#1B2433", ec="none"))
bx.text(6.425, 1.30, '{"key": "' + reply + '"}', fontsize=7.0, family="monospace", color="#F3EFE6", ha="center", va="center")
bx.add_patch(FancyArrowPatch((6.425, 1.08), (6.425, 0.84), arrowstyle="-|>", mutation_scale=7, color=MUTE, lw=0.8))
bx.text(5.85, 0.80, "the harness pressed T on the\nlive task page, which saved the\nchoice as for any participant;\nthe page clock "
        "stood still while\nthe agent decided", fontsize=5.8, color=MUTE, va="top", linespacing=1.25)
for xa, xb in ((1.47, 1.58), (3.78, 4.0), (5.63, 5.8)):
    bx.add_patch(FancyArrowPatch((xa, 0.95), (xb, 0.95), arrowstyle="-|>", mutation_scale=8, color=MUTE, lw=0.9))

# ---- c, d: switching after wins and losses -------------------------------------------------------
tr, _, _ = load_study("study_4", C.EXCLUSION_MAIN)
ht = switch_table(tr); ht = ht[ht["consec1"]]
M = pd.read_csv(AI / "results/agent_metrics.csv"); M["instruction"] = M["instruction"].fillna("none")
gb = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1.25, 1.25], wspace=0.5, left=0.075, right=0.985, top=y(top_in + 2.62), bottom=y(H - 0.62))
w = 0.36; axes = {}
for key, title, pos in (("win", "After a win", 0), ("loss", "After a loss", 1)):
    ax = fig.add_subplot(gb[pos]); axes[key] = ax
    hs = ht[ht["prev_win"] == (1 if key == "win" else 0)]
    per = hs.groupby(["subject_id", "prev_iti"])["switch"].mean().unstack()
    groups = [("people", per[1], per[4], None)]
    for cond, clab in COND:
        g = M[M["condition"] == cond]; groups.append((TICK[cond], g[f"switch_{key}_1s"], g[f"switch_{key}_4s"], g))
    for gi, (glab, v1, v4, g) in enumerate(groups):
        for off, v, col in ((-w / 2, v1, COL["short"]), (w / 2, v4, COL["long"])):
            ax.bar(gi + off, v.mean(), w, color=col, alpha=1 if g is None else 0.55, edgecolor="none", yerr=v.sem(), capsize=2, error_kw=dict(lw=0.7))
        if g is not None:
            rng = np.random.default_rng(7 + gi)
            for _, r in g.iterrows():
                mk = INSTR_MARK[r["instruction"]][0]; jit = rng.uniform(-0.07, 0.07)
                for off, k in ((-w / 2, f"switch_{key}_1s"), (w / 2, f"switch_{key}_4s")):
                    ax.scatter(gi + off + jit, r[k], marker=mk, s=8, facecolor="white", edgecolor=INK, lw=0.6, zorder=4)
        d = v4 - v1
        nums[f"{key}_{glab.replace(chr(10), ' ')}"] = dict(p1=round(float(v1.mean()), 3), p4=round(float(v4.mean()), 3), diff=round(float(d.mean()), 3), n=int(len(d)))
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=5.8)
    ax.set_ylim(0, 1.0); ax.set_ylabel("p(switch)"); ax.set_title(title, pad=6, fontsize=8)
axes["loss"].axhline(2 / 3, color=COL["grey"], ls=":", lw=0.8)

# ---- e, f: switching and choosing the best option over the session ----------------------------------------------------------------
edges = np.arange(0, 101, 10); mids = edges[:-1] + 5
A = pd.read_csv(AI / "data/trials.csv")
at = tidy_trials(A[["subject_id", "trial", "choice", "rt", "win", "reward", "iti", "p1", "p2", "p3"]].copy())
ats = switch_table(at); ats = ats[ats["consec1"]]
cond_of = M.set_index("subject_id")["condition"]


def by_bin(tab, col):
    b = pd.cut(tab["trial"], edges, labels=mids)
    return tab.assign(bin=b).groupby(["subject_id", "bin"], observed=True)[col].mean().unstack().reindex(columns=mids)


hv = tr[tr["valid"]]; av = at[at["valid"]]
for pos, col, title, ylab, ylim in ((2, "switch", "Switching over the session", "p(switch)", (0, 0.8)),
                                    (3, "chose_best", "Choosing the best option", "p(chose best option)", (0, 1.0))):
    ax = fig.add_subplot(gb[pos]); axes[col] = ax
    hb = by_bin(ht if col == "switch" else hv, col); ab = by_bin(ats if col == "switch" else av, col)
    ax.fill_between(mids, hb.mean() - hb.sem(), hb.mean() + hb.sem(), color=COL["grey"], alpha=0.3, lw=0)
    ax.plot(mids, hb.mean(), color=COL["grey"], lw=1.4, label=f"people, Study 4 (n = {len(hb)})")
    nums[f"{col}_by_bin_people"] = [round(float(v), 3) for v in hb.mean()]
    for (cond, clab), ls in zip(COND, ("--", "-")):
        g = ab.loc[[s for s in ab.index if cond_of.get(s) == cond]]
        ax.fill_between(mids, g.mean() - g.sem(), g.mean() + g.sem(), color=INK, alpha=0.12, lw=0)
        ax.plot(mids, g.mean(), color=INK, lw=1.2, ls=ls, label=f"{clab.replace(chr(10), ' ')} (n = {len(g)})")
        nums[f"{col}_by_bin_{cond}"] = [round(float(v), 3) for v in g.mean()]
    if col == "chose_best":
        ax.axhline(1 / 3, color=COL["grey"], ls=":", lw=0.8)
        pr = hv.groupby("subject_id")["chose_best"].mean(); ar = av.groupby("subject_id")["chose_best"].mean()
        nums["chose_best_people"] = dict(mean=round(float(pr.mean()), 3), sem=round(float(pr.sem()), 3), n=int(len(pr)))
        for cond, _ in COND:
            v = ar.loc[[s for s in ar.index if cond_of.get(s) == cond]]
            nums[f"chose_best_{cond}"] = dict(mean=round(float(v.mean()), 3), sem=round(float(v.sem()), 3), n=int(len(v)))
        nums["chose_best_agents"] = dict(mean=round(float(ar.mean()), 3), n=int(len(ar)))
    ax.set_xlim(0, 100); ax.set_ylim(*ylim); ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("trial (10-trial bins)"); ax.set_ylabel(ylab); ax.set_title(title, pad=6, fontsize=8)
axes["switch"].legend(frameon=False, loc="upper right", fontsize=5.6, handlelength=1.8, borderaxespad=0.1)

handles = [Patch(color=COL["short"], label="1 s"), Patch(color=COL["long"], label="4 s")]
handles += [Line2D([], [], ls="none", marker=m, mfc="white", mec=INK, ms=3.5, label=l) for m, l in INSTR_MARK.values()]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=5, frameon=False, fontsize=6.0, handlelength=1.2,
           columnspacing=0.9, handletextpad=0.4)
fig.text(0.012, y(top_in + 0.08), "b", fontsize=11, fontweight="bold", va="top")
MF.letters(fig, [(axes["win"], "c"), (axes["loss"], "d"), (axes["switch"], "e"), (axes["chose_best"], "f")], dx=0.05)
out = MF.OUT / "Fig6_generalization.png"
fig.savefig(out); fig.savefig(MF.OUT / "Fig6_generalization.pdf"); plt.close(fig)
full = Image.open(out).convert("RGB"); assert full.size[0] == W, full.size
full.paste(pa, (0, 0)); full.save(out, dpi=(DPI, DPI))
nf = MF.OUT / "figure_numbers.json"; allnums = json.load(open(nf)) if nf.exists() else {}
allnums["fig6"] = nums; json.dump(allnums, open(nf, "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in nums.items() if k != "example"}, indent=1, default=float))
print("wrote", out, full.size)
