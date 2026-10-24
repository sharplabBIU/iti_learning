# Study 4 data (the task of Study 3 with a new ITI sequence (unfilled ITI))

Tidy tables built by `analysis/00_prepare_data.py` (via `analysis_common/data.py`) from the raw export of
the task, which is not shared because it holds Prolific IDs and demographics. Participants are identified
by pseudonyms (`S4_001`, ...) numbered in the sort order of their Prolific IDs.

## trials.csv (one row per trial, 12498 rows, 135 participants)
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

## participants.csv (one row per participant)
| column | meaning |
|---|---|
| `subject_id` | participant (pseudonym) |
| `n_trials_recorded`, `n_valid`, `n_missed` | trial counts |
| `chose_best_rate` | proportion of responded trials on which the best option was chosen |
| `mean_rt_ms`, `win_rate` | mean response time and share of wins over responded trials |
| `completed_task` | all 100 trials recorded (the task ends early after five missed trials) |
| `reached_min_trials` | the session reached trial 90: **the main sample**, used by every analysis |
| `p_switch_overall`, `p_switch_after_1s`, `p_switch_after_4s`, `iti_effect_all` | switching summaries (consecutive responded trials) |
| `bis_total` | BIS-11 impulsivity (8 items) total score |
| `pleasure_total` | SHAPS anhedonia total score |
| `bis_attention_passed` | embedded attention-check item of the BIS-11 (True / False / blank); a flag, nobody was excluded |
| `age_birthyear_discrepancy` | absolute difference between the age reported before practice and 2026 minus the birth year reported after the task (data-quality flag; age and birth year themselves are not shared) |

## exclusion_flow.csv
| stage | n |
|---|---|
| participants with trial data | 135 |
| main sample: session reached trial 90 of 100 | 117 |
|   not in the main sample: session ended before trial 90 | 18 |
|   main sample, completed all 100 trials | 116 |
|   main sample, at least one missed trial | 46 |
|   main sample, largest number of missed trials of a participant | 5 |
|   main sample with a BIS-11 impulsivity (8 items) score | 117 |
|   main sample with a SHAPS anhedonia score | 117 |
| failed the BIS attention check (flag only), all participants | 1 |
|   of which in the main sample | 1 |
| \|age - (2026 - birth year)\| > 2 (flag only), all participants | 0 |
|   of which in the main sample | 0 |

Interval convention used everywhere: the interval that precedes the choice on trial t is the `iti`
stored on trial t-1 (`prev_iti` in the analysis tables).
