"""20 - Is the exploration marker u consistent with the explore state of a hidden Markov model (HMM) of choices?
(Methods, Exploratory state.) Run from the Study 3 folder; it analyses Studies 3 and 4.

The marker u = 1 / (1 + n) of 19_exploration_marker_ppc.py (n the leaky count of the option's earlier choices, leak 0.85)
is compared with the filtered probability of an "explore" state in an HMM of choices alone (Ebitz, Albarran & Moore,
2018): one explore state that chooses at random and three exploit states that each repeat one option (lapse 0.02). The
stay probabilities of the explore and exploit states are fitted to each participant by maximum likelihood over a grid.
Both indices refer to the choice on the trial before the inter-trial interval (trial t-1). For each participant the
Spearman correlation between u and P(explore) is computed over post-win and over post-loss trials, and the correlations
are averaged across participants.

Inputs: each study's trial table (main sample).
Usage: python3 20_exploration_proxy_hmm.py [--studies 3 4]
Output: results/20_exploration_proxy_hmm/hmm_correlation.json in each study's folder
"""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                     # repository root, so analysis_common can be imported
import numpy as np
import pandas as pd
from scipy import stats
from analysis_common.data import load_study
from analysis_common.behavior import switch_table
from analysis_common import config as C

ROOT = HERE.parents[1]
LAM = 0.85
STUDIES = [int(x) for x in sys.argv[sys.argv.index("--studies") + 1:]] if "--studies" in sys.argv else [3, 4]
# HMM grids: stay probability of the explore state (14 values, 0.3 to 0.95) and of each exploit state (15 values, 0.5 to
# 0.99); lapse rate of the exploit states (0.02)
PE_GRID, PX_GRID, EPS = np.linspace(0.3, 0.95, 14), np.linspace(0.5, 0.99, 15), 0.02


def hmm_explore(ch):
    """Filtered P(explore) per trial for one participant's choices (0-indexed array ch).

    States: explore (uniform) + exploit option 0/1/2 (repeat it, lapse EPS); stay probabilities fitted by grid. The explore
    state stays with probability pe and otherwise moves to one of the exploit states; an exploit state stays with
    probability px and otherwise returns to explore. For each grid pair the forward pass gives the log-likelihood and the
    filtered P(explore | choices so far); the best pair's filtered values are returned."""
    n = len(ch); best = (-np.inf, None)
    # Emission probabilities of each trial's choice: explore 1/3; exploit k 1 - EPS if k was chosen, EPS / 2 otherwise
    em = np.empty((n, 4)); em[:, 0] = 1 / 3
    for k in range(3):
        em[:, k + 1] = np.where(ch == k, 1 - EPS, EPS / 2)
    for pe in PE_GRID:
        for px in PX_GRID:
            # transition matrix (rows: from explore, exploit 0, 1, 2)
            T = np.array([[pe, (1 - pe) / 3, (1 - pe) / 3, (1 - pe) / 3]] + [[1 - px] + [px if j == k else 0.0 for j in range(3)] for k in range(3)])
            # forward pass, starting from P(explore) = 0.5 and 1/6 for each exploit state
            a = np.array([0.5, 1 / 6, 1 / 6, 1 / 6]) * em[0]; ll = np.log(a.sum()); a /= a.sum(); post = [a[0]]
            for t in range(1, n):
                a = (a @ T) * em[t]; s = a.sum(); ll += np.log(s); a /= s; post.append(a[0])
            if ll > best[0]:
                best = (ll, np.array(post))
    return best[1]


def indices(t):
    """Per responded trial: u (read before the choice) and the filtered HMM P(explore), keyed as prev1_trial so that they
    can be merged onto the trial before the ITI in the switch table."""
    rows = []
    for sid, g in t[t["valid"]].sort_values(["subject_id", "trial"]).groupby("subject_id"):
        ch = g["choice"].to_numpy().astype(int) - 1; hm = hmm_explore(ch); n = np.zeros(3)
        for i, (tr, c) in enumerate(zip(g["trial"].to_numpy(), ch)):
            rows.append((sid, tr, 1.0 / (1.0 + n[c]), hm[i])); nc = n[c] + 1.0; n *= (1.0 - LAM); n[c] = nc
    return pd.DataFrame(rows, columns=["subject_id", "prev1_trial", "u", "explore"])


for s in STUDIES:
    out = ROOT / f"study_{s}/analysis/results/20_exploration_proxy_hmm"; out.mkdir(parents=True, exist_ok=True)
    t = load_study(f"study_{s}", C.EXCLUSION_MAIN)[0]
    tab = switch_table(t); tab = tab[tab["consec1"]].merge(indices(t), on=["subject_id", "prev1_trial"])
    res = {}
    for w, lab in ((1, "after_win"), (0, "after_loss")):
        x = tab[tab["prev_win"] == w]
        rho = x.groupby("subject_id").apply(lambda g: stats.spearmanr(g["u"], g["explore"])[0] if g["u"].nunique() > 1 else np.nan).dropna()
        res[lab] = dict(mean_within_participant_spearman=float(rho.mean()), n_participants=int(len(rho)))
        print(f"Study {s}, {lab}: mean within-participant Spearman rho(u, P(explore)) = {rho.mean():.3f} (n = {len(rho)})", flush=True)
    json.dump(dict(study=s, leak=LAM, **res), open(out / "hmm_correlation.json", "w"), indent=1)
    print("wrote", out)
