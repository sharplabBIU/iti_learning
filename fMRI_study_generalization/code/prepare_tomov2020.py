"""Build the tidy trial table for the Tomov, Truong, Hundia & Gershman (2020) fMRI two-armed bandit.

Source: github.com/tomov/Exploration-Data-Analysis, data/*.csv. The repository has no licence file, so the
data may be analysed here but must not be redistributed.

Design. 31 people x 2 sessions. Each session has 4 blocks (conditions RS/SR/RR/SS, as labelled in the raw
files) x 4 runs of 10 trials. Each 10-trial run is a new pair of arms with fixed, unknown mean rewards. The
arms' left/right positions vary, so option identity is the arm COLOUR, and the choice is the colour of the
chosen side. The sequence unit (the "subject" column) is session x block x run, i.e. 10 trials.

ITI definition. The inter-trial interval (ITI) preceding a choice is the SCHEDULED jitter itiDuration of the
previous trial (random, 4.8-7.3 s). The realised ITI (actualItiDuration) was adjusted to keep the run on
schedule and correlates -0.46 with the participant's own response time (RT). It is therefore NOT used as the
predictor: it produces a spurious negative effect through RT. It is kept in the column iti_realised.

Reward coding. The raw reward is a number; it is recoded to 1 if it is above the median reward of its 10-trial
run and 0 otherwise.

Input:  ../data/tomov2020/subjects/*.csv (one file per participant and session).
Output: ../data/tomov2020/tidy.csv, one row per trial (the iti on row t is the ITI before that row's choice; NaN on
        the first trial of each run). Columns:
        subject (session_bBLOCK_rRUN), participant, trial, choice (colour 1-8, "0" = missed response),
        reward, iti, iti_realised, prev_rt (RT on the previous trial), condition, session.

Run: python3 prepare_tomov2020.py (no arguments); then tomov_iti_glmm.py. reproduce.sh downloads the raw files into
data/tomov2020/subjects/ if they are missing.
"""
import glob, numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parent.parent / "data" / "tomov2020"
# One file per participant and session; rows without a trialId are not trials and are dropped.
# person = participant ID (e.g. uep001); sess = participant and session (e.g. uep001_1).
rows = []
for f in sorted(glob.glob(str(D / "subjects" / "*.csv"))):
    s = pd.read_csv(f); s = s[s.trialId.notna()].copy()
    person = s.participant.iloc[0].split("_")[0]; sess = s.participant.iloc[0]
    # One 10-trial run (one pair of arms) at a time, so that the shift(1) lags never cross runs.
    for (b, r), g in s.groupby(["blockId", "runId"]):
        g = g.sort_values("trialId").reset_index(drop=True)
        key = g["responseKey.keys"]                     # 1 = left arm chosen, 2 = right arm chosen
        # Arm colours are coded 0-7 in the raw files. The tidy table uses colour + 1 (1-8), so
        # that "0" marks only a missed response (tomov_iti_glmm.py drops choice "0").
        chosen = np.where(key == 1, g.leftColor + 1, np.where(key == 2, g.rightColor + 1, "none"))
        # iti, iti_realised and prev_rt come from the previous trial of the same run (shift(1)).
        for i in range(len(g)):
            rows.append(dict(subject=f"{sess}_b{int(b)}_r{int(r)}", participant=person, trial=int(g.trialId[i]),
                             choice=(str(int(float(chosen[i]))) if chosen[i] != "none" else "none"), reward=float(g.reward[i]), iti=(g.itiDuration.shift(1)[i]),
                             iti_realised=(g.actualItiDuration.shift(1)[i]), prev_rt=(g.responseTime.shift(1)[i]),
                             condition=g.condition[i], session=sess))
# Binarise reward within each 10-trial run (above the run's median = 1), code misses as "0", save and report.
d = pd.DataFrame(rows)
d["reward"] = (d.reward > d.groupby("subject").reward.transform("median")).astype(int)
d.loc[d.choice == "none", "choice"] = "0"
d.to_csv(D / "tidy.csv", index=False)
v = d[(d.choice != "0") & d.iti.notna()]
print(f"tomov2020: {d.participant.nunique()} people, {d.subject.nunique()} 10-trial runs, {len(v)} trials with a valid choice and a preceding interval")
print("scheduled interval quantiles (s):", v.iti.quantile([0, .25, .5, .75, 1]).round(2).to_dict())
