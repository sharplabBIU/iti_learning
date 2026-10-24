"""Raw exports -> tidy tables with pseudonymous participant IDs, the participant flow, and loaders.

This module turns each study's raw export into the tables every analysis uses, and loads them with
the inclusion rule applied. The raw exports hold Prolific IDs and demographics and are not part of the
repository (config.PRIVATE_ROOT); the tables in data/study_N/data/ are what prepare_study() writes
from them. Every analysis reads those tables through load_study().

Main steps (prepare_study, run by 00_prepare_data.py)
  1. Read the raw export: a CSV for Studies 1-2, a JSON file for Studies 3-4 (config.STUDIES).
  2. Tidy the trial table: validity, best option, probability of the chosen option.
  3. Build one row per participant with trial counts, learning measures, questionnaire totals, the
     main-sample flag and the participant's switching rate after 1 s and 4 s inter-trial intervals
     (ITIs).
  4. Replace the Prolific IDs by pseudonyms and drop the identifying columns (pseudonymize).
  5. Count how many participants pass each stage of the inclusion rule (participant_flow).

Outputs: data/study_N/data/trials.csv, participants.csv and exclusion_flow.csv.
load_study() reads these tables and applies the inclusion rule; subjects_for_fitting() converts the
trial table into the per-participant arrays the models use.

Conventions (identical for all studies)
  * one row per trial; `choice` in {1,2,3}, 0 = missed (no response within 2 s)
  * `iti` on row t is the ITI that FOLLOWED trial t (1 or 4 s); the ITI
    preceding the choice on trial t is therefore iti[t-1] (see behavior.switch_table)
  * `reward` is +1 (win) / -1 (loss); `win` is 1/0
  * `p1`, `p2`, `p3` are the reward probabilities of the three options on that trial
  * `best_option` = argmax(p1,p2,p3); `chose_best` = choice == best_option (missed = 0)
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

from . import config as C

TRIAL_COLS = ["subject_id", "trial", "choice", "rt", "win", "reward", "iti", "p1", "p2", "p3"]   # rt in ms
IDENTIFYING = ["prolific_id", "age", "birth_year"]                                             # never written to the shared tables


# ---------------------------------------------------------------------------
# raw readers
# ---------------------------------------------------------------------------
def _read_raw_csv(path, id_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a CSV export (Studies 1-2): one row per trial, questionnaire scores repeated on each row.

    Returns (trials, participants). Columns starting with rdm_ (the random-dot-motion task of
    Study 2) are kept in the trial table; the participant table takes the first value per
    participant of the questionnaire totals that exist (collected in Studies 1 and 2, not analysed).
    """
    df = pd.read_csv(path)
    if id_col != "subject_id":
        df = df.rename(columns={id_col: "subject_id"})
    extra = [c for c in df.columns if c.startswith("rdm_")]
    trials = df[TRIAL_COLS + extra].copy()
    const_cols = [c for c in ["pswq", "survey_physical_total"] if c in df.columns]
    subj = df.groupby("subject_id")[const_cols].first().reset_index()
    return trials, subj


def _read_raw_json(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a JSON export (Studies 3-4) with a "participants" list and an "allTrials" list.

    Returns (trials, participants). Besides the questionnaire totals and the BIS attention check, the
    participant table gets the mismatch between the age reported before practice and the birth year
    reported after the task (a data-quality flag); age and birth year themselves are dropped later
    (pseudonymize).
    """
    with open(path) as f:
        d = json.load(f)
    P = pd.DataFrame(d["participants"])
    T = pd.DataFrame(d["allTrials"])
    for c in ["trial", "choice", "rt", "win", "reward", "iti", "p1", "p2", "p3"]:
        T[c] = pd.to_numeric(T[c], errors="coerce")
    trials = T[TRIAL_COLS].copy()
    P = P.copy()
    for c in ["age", "birth_year", "bis_total", "pleasure_total"]:
        P[c] = pd.to_numeric(P[c], errors="coerce")

    def _tf(v):  # "true"/"false" text -> True/False; anything else -> missing
        s = str(v).strip().lower()
        return True if s == "true" else (False if s == "false" else np.nan)
    P["bis_attention_passed"] = P["bis_attention_passed"].map(_tf)
    P["age_birthyear_discrepancy"] = (2026 - P["birth_year"] - P["age"]).abs()   # age vs 2026 - birth year
    subj = P[["subject_id", "age", "birth_year", "age_birthyear_discrepancy", "bis_total", "pleasure_total", "bis_attention_passed"]]
    return trials, subj


# ---------------------------------------------------------------------------
# tidy trials / participants
# ---------------------------------------------------------------------------
def tidy_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Sort by participant and trial, and add the derived trial columns.

    valid = a response was made (choice 1-3); best_option = option with the highest reward
    probability (1-3); chose_best = 1 if a valid choice picked it; p_chosen = reward probability of
    the chosen option (missing for missed trials); p_max = the highest reward probability on that
    trial.
    """
    t = trials.sort_values(["subject_id", "trial"]).reset_index(drop=True).copy()
    for c in ["trial", "choice", "win", "reward", "iti"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["valid"] = t["choice"].isin([1, 2, 3])
    probs = t[["p1", "p2", "p3"]].to_numpy(dtype=float)
    t["best_option"] = probs.argmax(axis=1) + 1
    t["chose_best"] = ((t["choice"] == t["best_option"]) & t["valid"]).astype(int)
    # Column index of the chosen option (0-2). Missed trials get a placeholder index; their p_chosen
    # is then set to missing.
    idx = (t["choice"].fillna(1).astype(int) - 1).clip(0, 2).to_numpy()
    t["p_chosen"] = np.where(t["valid"], probs[np.arange(len(t)), idx], np.nan)
    t["p_max"] = probs.max(axis=1)
    return t


def participant_table(trials: pd.DataFrame, subj: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: trial counts, learning and win rates, questionnaire totals and flags.

    completed_task = all 100 trials recorded (the task ends early after five missed trials);
    reached_min_trials = the session reached trial 90, the main sample.
    """
    g = trials.groupby("subject_id")
    v = trials[trials["valid"]].groupby("subject_id")
    P = pd.DataFrame({
        "n_trials_recorded": g.size(),
        "n_valid": v.size(),
        "n_missed": g["valid"].apply(lambda s: int((~s).sum())),
        "chose_best_rate": v["chose_best"].mean(),            # over responded trials
        "mean_rt_ms": v["rt"].mean(),
        "win_rate": v["win"].mean(),
    }).reset_index()
    P["n_valid"] = P["n_valid"].fillna(0).astype(int)
    P = P.merge(subj, on="subject_id", how="left")
    P["completed_task"] = P["n_trials_recorded"] >= C.N_TASK_TRIALS
    P["reached_min_trials"] = P["n_trials_recorded"] >= C.MIN_TRIALS_REACHED
    return P


def pseudonymize(trials: pd.DataFrame, P: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace each Prolific ID by a pseudonym S<study>_<k> and drop the identifying columns.

    k numbers the participants in the sort order of their Prolific IDs (zero-padded), so that sorting
    by pseudonym gives the same participant order as sorting by Prolific ID, and every analysis gives
    the same result under either ID. The map is written to config.ID_MAP_CSV, outside the repository,
    the first time a study is prepared and reused afterwards, so the pseudonyms never change.
    """
    study = key.split("_")[1]
    ids = sorted(set(trials["subject_id"].astype(str)) | set(P["subject_id"].astype(str)))
    old = (pd.read_csv(C.ID_MAP_CSV, dtype=str) if C.ID_MAP_CSV.exists()
           else pd.DataFrame(columns=["study", "original_id", "pseudonym"]))
    mine = old[old["study"] == study]
    if len(mine):
        m = dict(zip(mine["original_id"], mine["pseudonym"]))
        missing = [i for i in ids if i not in m]
        if missing:
            raise ValueError(f"{len(missing)} participants of {key} have no pseudonym in {C.ID_MAP_CSV}")
    else:
        width = len(str(len(ids)))
        m = {o: f"S{study}_{k:0{width}d}" for k, o in enumerate(ids, start=1)}
        new = pd.DataFrame(dict(study=study, original_id=list(m), pseudonym=list(m.values())))
        C.ID_MAP_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([old, new], ignore_index=True).to_csv(C.ID_MAP_CSV, index=False)
    assert sorted(m[i] for i in ids) == [m[i] for i in ids], "the pseudonyms must keep the participant order"
    trials = trials.assign(subject_id=trials["subject_id"].astype(str).map(m))
    P = P.assign(subject_id=P["subject_id"].astype(str).map(m)).drop(columns=[c for c in IDENTIFYING if c in P.columns])
    return trials, P


def participant_flow(P: pd.DataFrame, study: dict) -> pd.DataFrame:
    """Table of (stage, n): how many participants the inclusion rule keeps, and flag counts (Methods).

    Indented stages are subsets of the stage above. The attention-check and age rows need the JSON
    export (Studies 3-4)."""
    main = P["reached_min_trials"]
    rows = [("participants with trial data", len(P)),
            (f"main sample: session reached trial {C.MIN_TRIALS_REACHED} of {C.N_TASK_TRIALS}", int(main.sum())),
            (f"  not in the main sample: session ended before trial {C.MIN_TRIALS_REACHED}", int((~main).sum())),
            (f"  main sample, completed all {C.N_TASK_TRIALS} trials", int((main & P["completed_task"]).sum())),
            ("  main sample, at least one missed trial", int((main & (P["n_missed"] > 0)).sum())),
            ("  main sample, largest number of missed trials of a participant", int(P.loc[main, "n_missed"].max()))]
    for q in study["questionnaires"].values():
        rows.append((f"  main sample with a {q['label']} score", int((main & P[q["col"]].notna()).sum())))
    if "bis_attention_passed" in P.columns:
        failed = P["bis_attention_passed"] == False                                       # noqa: E712 (missing = not answered)
        rows += [("failed the BIS attention check (flag only), all participants", int(failed.sum())),
                 ("  of which in the main sample", int((failed & main).sum()))]
    if "age_birthyear_discrepancy" in P.columns:
        off = P["age_birthyear_discrepancy"] > 2
        rows += [("|age - (2026 - birth year)| > 2 (flag only), all participants", int(off.sum())),
                 ("  of which in the main sample", int((off & main).sum()))]
    return pd.DataFrame(rows, columns=["stage", "n"])


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def prepare_study(key: str, verbose: bool = True) -> dict:
    """Build data/trials.csv, data/participants.csv, data/exclusion_flow.csv for one study.

    key is a study key from config.STUDIES (e.g. "study_3"). Needs the raw export in
    config.PRIVATE_ROOT/raw/<key>/. Returns dict(trials, participants, flow) and, with verbose, prints
    the participant flow.
    """
    s = C.STUDIES[key]
    p = C.study_paths(key)
    raw = p["raw_dir"] / s["raw_file"]
    if not raw.exists():
        raise FileNotFoundError(f"{raw}: the raw exports are not part of the repository; the tidy tables in "
                                f"{p['data_dir']} are the output of this step")
    if s["raw_kind"] == "csv":
        trials, subj = _read_raw_csv(raw, s["id_col"])
    else:
        trials, subj = _read_raw_json(raw)
    trials = tidy_trials(trials)
    P = participant_table(trials, subj)
    # Each participant's switching rate overall and after a 1 s or 4 s ITI, and the difference
    # (4 s minus 1 s). Only choices whose previous trial was also responded (consec1) are used, so
    # prev_iti is the actual ITI.
    from .behavior import switch_table
    st = switch_table(trials)
    st = st[st["consec1"]]
    P["p_switch_overall"] = P["subject_id"].map(st.groupby("subject_id")["switch"].mean())
    P["p_switch_after_1s"] = P["subject_id"].map(st[st.prev_iti == 1].groupby("subject_id")["switch"].mean())
    P["p_switch_after_4s"] = P["subject_id"].map(st[st.prev_iti == 4].groupby("subject_id")["switch"].mean())
    P["iti_effect_all"] = P["p_switch_after_4s"] - P["p_switch_after_1s"]
    trials, P = pseudonymize(trials, P, key)
    flow = participant_flow(P, s)
    p["data_dir"].mkdir(parents=True, exist_ok=True)
    trials.to_csv(p["trials_csv"], index=False)
    P.to_csv(p["participants_csv"], index=False)
    flow.to_csv(p["flow_csv"], index=False)
    if verbose:
        print(f"[{s['label']}] trials: {len(trials)} rows, {trials.subject_id.nunique()} participants")
        print(flow.to_string(index=False))
    return dict(trials=trials, participants=P, flow=flow)


def load_study(key: str, exclusion: str = C.EXCLUSION_MAIN, apply_rt_floor: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (trials, participants, study_config) with the inclusion rule applied.

    exclusion: 'min90'  the main sample: sessions that reached at least 90 of the 100 trials
               'none'   everyone with trial data
    apply_rt_floor: if the study defines rt_floor_ms (Study 2, 250 ms), a choice faster than the
               floor is marked invalid, i.e. treated exactly like a missed trial (not scored, and the
               switch on the next trial is not scored either, because its previous choice is
               missing). In Study 2 such choices are late Q+T+P answers (the Q, T and P keys pressed
               together) to the dot-motion display that land on the choice screen. The flag column
               rt_floor_invalid marks them. False gives the raw validity.
    """
    s = C.STUDIES[key]
    p = C.study_paths(key)
    trials = pd.read_csv(p["trials_csv"])
    P = pd.read_csv(p["participants_csv"])
    if exclusion == "min90":
        keep = P.loc[P["n_trials_recorded"] >= C.MIN_TRIALS_REACHED, "subject_id"]
    elif exclusion == "none":
        keep = P["subject_id"]
    else:
        raise ValueError(exclusion)
    trials = trials[trials["subject_id"].isin(keep)].copy()
    P = P[P["subject_id"].isin(keep)].copy()
    # Response-time floor (Study 2): too-fast choices become missed trials (chose_best 0, no
    # p_chosen)
    floor = s.get("rt_floor_ms")
    trials["rt_floor_invalid"] = False
    if apply_rt_floor and floor:
        fast = trials["valid"] & (pd.to_numeric(trials["rt"], errors="coerce") < floor)
        trials.loc[fast, "valid"] = False; trials.loc[fast, "rt_floor_invalid"] = True
        trials.loc[fast, "chose_best"] = 0; trials.loc[fast, "p_chosen"] = np.nan
    return trials, P, s


def subjects_for_fitting(trials: pd.DataFrame) -> list[dict]:
    """Per-subject arrays of responded trials, in the form the likelihoods expect.

    Returns a list with one dict per participant: subject_id; choices (0-indexed);
    rewards (+1 / -1); itis (seconds, the ITI that followed each trial); trial (actual trial
    numbers, so gaps mark missed trials); probs (reward probabilities, trials x 3).
    """
    out = []
    v = trials[trials["valid"]].sort_values(["subject_id", "trial"])
    for sid, g in v.groupby("subject_id", sort=False):
        out.append(dict(subject_id=sid,
                        choices=g["choice"].to_numpy().astype(int) - 1,     # 0-indexed
                        rewards=g["reward"].to_numpy().astype(float),        # +1 / -1
                        itis=g["iti"].to_numpy().astype(float),              # seconds (1 or 4)
                        trial=g["trial"].to_numpy().astype(int),
                        probs=g[["p1", "p2", "p3"]].to_numpy().astype(float)))
    return out
