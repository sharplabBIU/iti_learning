"""00 - Build the tidy data tables for this study from its raw export and document them in data/README.md.

This script runs for any study (1, 2, 3 or 4). The study is chosen by the folder the script is run from:
a copy in data/study_N/analysis/ processes study N (analysis_common/runner.py reads the folder name).

The raw exports hold Prolific IDs and demographics and are NOT part of the repository, so this step
cannot be rerun from the repository alone: the tables it wrote are in data/study_N/data/, and every
other script starts from them. It is kept to document how they were made. Steps (analysis_common/data.py,
prepare_study): read the raw export, build the trial and participant tables, replace each Prolific ID
by an order-preserving pseudonym (the map is kept with the raw exports), drop identifying columns, and
count the participant flow. Then write a README that describes every column.

Inputs: the raw export in config.PRIVATE_ROOT/raw/study_N/ (file name set by `raw_file` in
analysis_common/config.py).
Outputs: data/trials.csv (one row per trial), data/participants.csv (one row per participant),
data/exclusion_flow.csv (the participant flow of the Methods), and data/README.md.

Run from anywhere:  python3 data/study_N/analysis/00_prepare_data.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root (script sits in data/study_N/analysis/)
from analysis_common.runner import md_table, Runner
from analysis_common.data import prepare_study
from analysis_common import config as C

R = Runner(__file__)
R.banner("00 prepare data")
res = prepare_study(R.key)
s = R.study
T, P = res["trials"], res["participants"]
# One README table row per questionnaire total and per study-specific column.
cols = []
for q in s["questionnaires"].values():
    cols.append(f"| `{q['col']}` | {q['label']} total score |")
if "bis_attention_passed" in P.columns:
    cols += ["| `bis_attention_passed` | embedded attention-check item of the BIS-11 (True / False / blank); a flag, nobody was excluded |",
             "| `age_birthyear_discrepancy` | absolute difference between the age reported before practice and 2026 minus the birth year "
             "reported after the task (data-quality flag; age and birth year themselves are not shared) |"]
if "pswq" in P.columns:
    cols.append("| `pswq`, `survey_physical_total` | totals of the abbreviated Penn State Worry Questionnaire and of the Anxious Arousal "
                "subscale of the Mood and Anxiety Symptom Questionnaire (collected in Studies 1 and 2, not analysed in the paper) |")
extra = ""
if s.get("has_rdm"):
    extra = """
`trials.csv` also holds the random-dot-motion (RDM) task performed during the ITI: `rdm_correct` (1/0),
`rdm_rt_ms`, `rdm_target_direction` (UP / DOWN), `rdm_user_response` (UP = no simultaneous Q+T+P press,
DOWN = a press) and `rdm_direction_pair`. Choices faster than {floor} ms are treated as missed trials by
analysis_common/data.load_study (Methods).
""".format(floor=s["rt_floor_ms"])
cols_txt = "\n".join(cols)
readme = f"""# {s['label']} data ({s['subtitle']})

Tidy tables built by `analysis/00_prepare_data.py` (via `analysis_common/data.py`) from the raw export of
the task, which is not shared because it holds Prolific IDs and demographics. Participants are identified
by pseudonyms (`{P.subject_id.min()}`, ...) numbered in the sort order of their Prolific IDs.

## trials.csv (one row per trial, {len(T)} rows, {T.subject_id.nunique()} participants)
| column | meaning |
|---|---|
| `subject_id` | participant (pseudonym) |
| `trial` | 1..100 |
| `choice` | 1/2/3 = option chosen; **0 = missed** (no response within 2 s) |
| `rt` | response time (ms) |
| `win`, `reward` | 1/0 and +1/-1; reward 0 only on missed trials |
| `iti` | inter-trial interval in seconds (1 or 4) that **followed** this trial |
| `p1`,`p2`,`p3` | reward probability of each option on this trial (random walk) |
| `valid` | choice in 1..3 |
| `best_option`, `chose_best` | argmax(p1,p2,p3) and whether it was chosen (missed = 0) |
| `p_chosen`, `p_max` | reward probability of the chosen option / of the best option |
{extra}
## participants.csv (one row per participant)
| column | meaning |
|---|---|
| `subject_id` | participant (pseudonym) |
| `n_trials_recorded`, `n_valid`, `n_missed` | trial counts |
| `chose_best_rate` | proportion of responded trials on which the best option was chosen |
| `mean_rt_ms`, `win_rate` | mean response time and share of wins over responded trials |
| `completed_task` | all {C.N_TASK_TRIALS} trials recorded (the task ends early after five missed trials) |
| `reached_min_trials` | the session reached trial {C.MIN_TRIALS_REACHED}: **the main sample**, used by every analysis |
| `p_switch_overall`, `p_switch_after_1s`, `p_switch_after_4s`, `iti_effect_all` | switching summaries (consecutive responded trials) |
{cols_txt}

## exclusion_flow.csv
{md_table(res['flow'])}

Interval convention used everywhere: the interval that precedes the choice on trial t is the `iti`
stored on trial t-1 (`prev_iti` in the analysis tables).
"""
(R.paths["data_dir"] / "README.md").write_text(readme)
print("wrote", R.paths["data_dir"] / "README.md")
