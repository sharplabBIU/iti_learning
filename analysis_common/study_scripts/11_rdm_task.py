"""11 (Study 2 only) - Accuracy on the random-dot-motion (RDM) task that filled the inter-trial interval (ITI).

In Study 2 a random-dot-motion display filled each ITI and participants judged its direction (Methods, Task). Accuracy is
computed per participant for the 1 s and the 4 s displays and compared by a paired t test (Results, Study 2). The display
stored on row t filled the ITI that followed trial t. The script exits at once for a study without the RDM task
(`has_rdm` in analysis_common/config.py).

Inputs: data/trials.csv (columns rdm_correct, iti), main sample.
Outputs (results/11_rdm_task/): rdm_stats.json (accuracy after 1 s and 4 s, t, df, p)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (script sits in data/study_N/analysis/)
from scipy import stats
from analysis_common.runner import Runner, fmt_p

R = Runner(__file__)
if not R.study.get("has_rdm"):
    sys.exit(f"{R.label} has no RDM task; nothing to do")
R.banner("11 RDM task during the ITI")
out = R.out("11_rdm_task")
trials, P = R.load()
# Per-participant accuracy for the 1 s and the 4 s displays (chance = 0.5 for two directions), paired t test.
v = trials[trials["valid"] & trials["rdm_correct"].notna()]
acc = v.pivot_table(index="subject_id", columns="iti", values="rdm_correct", aggfunc="mean").dropna()
t, p = stats.ttest_rel(acc[4], acc[1])
R.save_json(out, "rdm_stats", dict(accuracy_by_iti=dict(n=len(acc), df=len(acc) - 1, acc_1s=acc[1].mean(), acc_4s=acc[4].mean(),
                                                         t=float(t), p=float(p), chance=0.5)))
print(f"RDM accuracy: 1 s = {acc[1].mean():.3f}, 4 s = {acc[4].mean():.3f}, t({len(acc) - 1}) = {t:.2f}, {fmt_p(p)}")
print("wrote", out)
