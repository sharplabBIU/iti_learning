"""The Study 4 measures for the language-model agents, with the Study 4 participants (main sample) for comparison (Results [61]).

Uses the definitions of the analysis package (tidy trials, best option, consecutive responded trial pairs, the ITI before a choice =
the iti stored on the previous trial), so every number is defined exactly as for the participants. Per agent: the chose-best rate,
switching after a 1 s and a 4 s ITI over all trials, after wins and after losses, the stay rates after a win and after a loss, the
questionnaire answers, reply quality from the run record (replies that were not just the answer, and the number of reasoning
summaries), and each measure's percentile among the Study 4 participants. With 14 agents the measures are descriptive.
Run from anywhere: python3 ai_agents/analyze_agents.py
Outputs (results/): agent_metrics.csv (one row per agent) and summary.md
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                                  # the repository root
sys.path.insert(0, str(REPO))
from analysis_common.data import tidy_trials                                   # noqa: E402
from analysis_common.behavior import switch_table                              # noqa: E402
from analysis_common.runner import md_table                                     # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)

# ---- data ---------------------------------------------------------------------------------------------------------------------
A = pd.read_csv(HERE / "data" / "trials.csv"); AP = pd.read_csv(HERE / "data" / "participants.csv")
H = pd.read_csv(REPO / "data" / "study_4" / "data" / "trials.csv"); HP = pd.read_csv(REPO / "data" / "study_4" / "data" / "participants.csv")
HP = HP[HP.n_trials_recorded >= 90]; H = H[H.subject_id.isin(HP.subject_id)]
cols = ["subject_id", "trial", "choice", "rt", "win", "reward", "iti", "p1", "p2", "p3"]
A_t = tidy_trials(A[cols].copy())
meta = AP.set_index("subject_id")


def metrics(t: pd.DataFrame) -> dict:
    v = t[t.valid]
    tab = switch_table(t); tab = tab[tab.consec1]
    m = dict(n_trials=len(t), n_valid=len(v), chose_best=v.chose_best.mean(), n_pairs=len(tab), n_switches=int(tab.switch.sum()),
             p_switch=tab.switch.mean())
    for lab, sub in (("all", tab), ("win", tab[tab.prev_win == 1]), ("loss", tab[tab.prev_win == 0])):
        for iti in (1, 4):
            s = sub[sub.prev_iti == iti]
            m[f"switch_{lab}_{iti}s"] = s.switch.mean() if len(s) else np.nan
            m[f"n_{lab}_{iti}s"] = len(s)
        m[f"iti_effect_{lab}"] = m[f"switch_{lab}_4s"] - m[f"switch_{lab}_1s"]
    m["stay_after_win"] = 1 - tab[tab.prev_win == 1].switch.mean()
    m["stay_after_loss"] = 1 - tab[tab.prev_win == 0].switch.mean()
    return m


def reply_quality(pid: str) -> dict:
    f = HERE / "runs" / f"{pid}.json"
    if not f.exists():
        return {}
    ex = json.load(open(f))["exchanges"]
    keys = [e for e in ex if e["kind"] == "key"]

    def clean(e):                     # structured-output runs keep the model's raw JSON in raw_reply; earlier runs only the reply
        if "raw_reply" in e:
            try:
                d = json.loads(e["raw_reply"])
                return set(d) == {"key"} and d["key"] in ("Q", "T", "P")
            except (json.JSONDecodeError, TypeError, AttributeError):
                return False
        return e["reply"].strip().upper() in ("Q", "T", "P")
    extra = [e for e in keys if not clean(e)]
    invented = [e for e in extra if "You pressed" in e.get("raw_reply", e["reply"]) or "TRIAL" in e.get("raw_reply", e["reply"])]
    return dict(key_replies=len(keys), key_replies_with_extra_text=len(extra), key_replies_inventing_next_screen=len(invented),
                reasoning_summaries=sum(1 for e in keys if e.get("thinking_summary")))


rows = []
for sid, t in A_t.groupby("subject_id"):
    r = dict(subject_id=sid, model=meta.loc[sid, "model"], condition=meta.loc[sid, "condition"], run_tag=meta.loc[sid, "run_tag"])
    r.update(metrics(t)); r.update(reply_quality(sid))
    for k in ("age", "birth_year", "bis_total", "pleasure_total", "bis_attention_passed"):
        r[k] = meta.loc[sid, k]
    rows.append(r)
M = pd.DataFrame(rows)
# the extra instruction (--persona) defines a separate group of agents; order agents by instruction, model, condition
M = M.merge(AP[["subject_id", "persona"]], on="subject_id", how="left")
M["instruction"] = M["persona"].where(M["persona"].notna() & (M["persona"].astype(str).str.len() > 0), "none")
M = M.sort_values(["instruction", "model", "condition", "subject_id"], key=lambda c: c.map(lambda v: (v != "none", v)) if c.name == "instruction" else c
                  ).reset_index(drop=True)
# humans: the same per-participant metrics, and each agent's percentile among them
Hm = pd.DataFrame([dict(subject_id=sid, **metrics(t)) for sid, t in H.groupby("subject_id")]).merge(
    HP[["subject_id", "bis_total", "pleasure_total"]], on="subject_id")
COMPARE = ["chose_best", "p_switch", "stay_after_win", "stay_after_loss", "iti_effect_all", "iti_effect_win", "iti_effect_loss",
           "bis_total", "pleasure_total"]
for c in COMPARE:
    hv = Hm[c].dropna().to_numpy()
    M[f"{c}_human_pct"] = [float((hv < x).mean() * 100 + (hv == x).mean() * 50) if pd.notna(x) else np.nan for x in M[c]]
M.to_csv(OUT / "agent_metrics.csv", index=False)

# ---- summary ------------------------------------------------------------------------------------------------------------------
show = ["subject_id", "model", "condition", "instruction", "chose_best", "p_switch", "n_switches", "switch_all_1s", "switch_all_4s",
        "switch_win_1s", "n_win_1s", "switch_win_4s", "n_win_4s", "switch_loss_1s", "n_loss_1s", "switch_loss_4s", "n_loss_4s",
        "stay_after_win", "stay_after_loss",
        "bis_total", "pleasure_total", "bis_attention_passed", "age", "birth_year",
        "key_replies_with_extra_text", "key_replies_inventing_next_screen"]
hum = {c: f"{Hm[c].mean():.3f} (sd {Hm[c].std():.3f})" for c in COMPARE}
lines = ["# Language-model agents: the Study 4 measures", "", f"{len(M)} agents; Study 4 humans (main sample): n = {len(Hm)}", "",
         md_table(M[[c for c in show if c in M.columns]].set_index("subject_id").T.reset_index().rename(columns={"index": "measure"}), 3), "",
         "Human mean (sd): " + "; ".join(f"{k} {v}" for k, v in hum.items()), "",
         "Percentile among the humans:", "", md_table(M[["subject_id"] + [f"{c}_human_pct" for c in COMPARE]].set_index("subject_id").T.reset_index().rename(columns={"index": "measure"}), 0)]
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("wrote", OUT)
