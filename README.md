# Inter-trial intervals bias reinforcement learning in humans: data and analysis code

Data and code to reproduce every analysis, figure and table of the main text and the Supplementary Information of
*Inter-trial intervals bias reinforcement learning in humans* (Yadid, Linkovski & Sharp). Four online studies of a
three-armed restless bandit in which the inter-trial interval (ITI) after each outcome was 1 s or 4 s (Study 1, unfilled
ITI, n = 96; Study 2, ITI filled with a random-dot-motion task, n = 96; Study 3, preregistered replication with
questionnaires, n = 439; Study 4, preregistered replication with an ITI sequence built against the hypothesis, n = 117),
an external fMRI bandit dataset (Tomov et al., 2020), and language-model agents that played the Study 4 task.

## Reproducing the paper

```bash
pip install -r requirements.txt
./reproduce.sh                  # everything, including the parameter and model recovery of Fig. 4d,e (several hours more)
./reproduce.sh --skip-recovery  # everything except the recovery (several hours on an 18-core laptop, mostly the model fits)
```

`reproduce.sh` runs every analysis below from the shared data, draws every figure and Supplementary Table 1, and finally
runs `check_reported_numbers.py`, which compares the regenerated results with each number reported in the paper and writes
`reproduction_report.md` (one row per number: the value in the paper, the regenerated value and the verdict). Independent
parts run side by side; `--serial` runs them one after another on smaller machines. Logs go to `logs/` (one per step, and
`logs/status.tsv` with the outcome and time of every step). With `--skip-recovery` the recovery results already in
`data/study_3/analysis/results/07_recovery/` are kept and used by Fig. 4.

Each analysis can also be run on its own, from any folder, for example `python3 data/study_3/analysis/02_iti_switching.py`.
A script analyses the study whose folder it sits in; the copies in `data/study_N/analysis/` are identical to the master
copies in `analysis_common/study_scripts/` (reproduce.sh checks this). Run the model fits (`06`) before the scripts that use
them (`10`, `19`, `09`, and Figs. 2 and 4). Every analysis uses the main sample: participants whose session reached trial 90
of the 100 (no exclusion criterion was preregistered).

Software: Python 3.13 with PyMC 6.3, nutpie 0.16, ArviZ 1.3, PyTensor 3.3, SciPy 1.18, statsmodels 0.15, pandas 3.0,
matplotlib 3.11 (`requirements.txt`; `logs/environment.txt` lists the versions of each run). Posterior summaries and
simulations can differ in the last digits between machines and library versions; `reproduction_report.md` marks such
values as "close".

## Last full run

TBD: the results of a complete run of `reproduce.sh` and its comparison with the paper (`reproduction_report.md`) will be
added. Until then the `results/` folders hold only the recovery results of Fig. 4d,e
(`data/study_3/analysis/results/07_recovery/`), which `--skip-recovery` uses; every other result is written by
`reproduce.sh`.

## What is shared, and what is not

* `data/study_N/data/` holds the tidy tables of each study: `trials.csv` (one row per trial), `participants.csv` (one row
  per participant) and `exclusion_flow.csv` (the participant flow of the Methods), with a data dictionary in `README.md`.
  Participants are identified by pseudonyms (`S3_001`, ...), numbered in the sort order of their Prolific IDs, so every
  analysis gives exactly the result it gives on the original IDs. Age, birth year, sex, ethnicity, nationality and the
  Prolific IDs are not shared.
* The raw exports of the task (with Prolific IDs and demographics) are not shared. `00_prepare_data.py` documents how the
  tidy tables were built from them; it needs the raw exports and is not part of `reproduce.sh`. The demographic
  description of the samples in the Methods comes from the Prolific exports and is therefore not reproducible from this
  repository.
* The Tomov et al. (2020) data have no licence and are not redistributed: `reproduce.sh` downloads them from the authors'
  repository (github.com/tomov/Exploration-Data-Analysis) into `fMRI_study_generalization/data/tomov2020/subjects/`.
* `ai_agents/` holds the complete run records of the language-model agents (every prompt, reply and reasoning summary),
  their task data and the harness that ran them. Rerunning the agents needs an Anthropic API key and the live task page;
  `reproduce.sh` reanalyses the saved runs.
* `tasks/` holds the browser tasks of the four studies (Firebase web apps). The Firebase project settings are replaced by
  placeholders; see `tasks/README.md`.

## Layout

```
reproduce.sh                  runs everything (see above); check_reported_numbers.py compares the results with the paper
analysis_common/              the shared analysis package (README.md there): data tables and loaders, behavioural tests,
                              Bayesian mixed models, the reinforcement-learning models (numpy and PyMC), model selection,
                              simulators; study_scripts/ holds the master copy of every analysis script
data/study_N/                 one folder per study
  data/                         trials.csv, participants.csv, exclusion_flow.csv, README.md (data dictionary)
  analysis/                     the analysis scripts of that study (copies of analysis_common/study_scripts/)
    results/                    one folder per script, written by it
figures/                      the scripts that draw Figs. 1-6, Supplementary Figs. 1-3 and Supplementary Table 1, and their
                              outputs (PNG at 300 dpi, PDF); figure_numbers.json holds numbers of the text computed there
fMRI_study_generalization/    the external dataset (Tomov et al., 2020): code/, results/, data/ (downloaded, not shared)
ai_agents/                    the language-model agents: harness, prompts, run records, data and analysis (README.md there)
tasks/                        the browser tasks (README.md there)
tests/test_likelihoods.py     the likelihood code against independent implementations, for every model
```

## Where each analysis of the paper comes from

`[n]` is the paragraph of the manuscript (Results, Methods). Scripts are in `data/study_N/analysis/` (S1-S4 = the studies
they are run for) and write to `data/study_N/analysis/results/<folder>/`.

### Main text

| Paper | Analysis | Script | Output |
|---|---|---|---|
| Methods [78], [83], [86] | participants with trial data, the main sample (96 of 101, 96, 439 of 488, 117 of 135), missed trials, attention-check and age flags | `00_prepare_data.py` (needs the raw exports) | `data/study_N/data/exclusion_flow.csv` |
| Results [21], [54]; Methods [89]; Fig. 1c | learning: chose-best rate against chance | `01_learning_check.py` (S1-S4) | `01_learning/learning_stats.json` |
| Results [21], [26], [29], [55]; Figs. 1d, 2b, 3a, 5a | p(switch) after a 1 s vs a 4 s ITI: all trials, after wins, after losses, post-win vs post-loss effect (paired t tests) | `02_iti_switching.py` (S1-S4) | `02_switching/main_effect.json`, `switching_by_outcome.json` |
| Results [31], [37]; Methods [91]-[93]; Fig. 5d | the preregistered mixed-effects logistic regression with the linear trial term; after wins and after losses; ITI x outcome interaction | `02_iti_switching.py` (S1-S4) | `02_switching/glmm.json`, `bglmm_*` |
| Results [21], [26], [32], [55]; Figs. 1e, 2c, 3b, 5b | the same contrasts in the first and second half of the session | `02b_iti_switching_phase_by_outcome.py` (S1-S4) | `02_switching/phase_by_outcome.json` |
| Results [25]; Methods [78]; Fig. 2d-f | choices faster than 250 ms in Study 2: response times, fit to learned values, keys | `figures/make_contamination_figure.py` (drawn with Fig. 2; uses the Study 2 fit of the model without an ITI term, `06_fit_models_nuts.py --models m08_trend_beta` in S2) | `figures/figure_numbers.json` (`study2_contamination`) |
| Results [26]; Methods [89] | dot-motion accuracy after 1 s and 4 s displays | `11_rdm_task.py` (S2) | `11_rdm_task/rdm_stats.json` |
| Results [30], [32], [54]; Methods [80], [84], [90]; Figs. 3c, 5c | the fixed ITI sequence (share of 4 s ITIs per block, correlation with trial number) and the decline in switching | `13_design_confound.py` (S3, S4) | `13_design_confound/design_confound.json` |
| Results [36], [55]; Methods [94] | the ITI after the choice two, three and four trials back (raw contrasts and mixed models with the trial term) | `16_position_controlled_contrasts.py` (S1-S4) | `16_position_controlled/` |
| Results [41]-[50], [56]; Methods [99]-[111]; Fig. 4a | the model sequence: hierarchical Bayesian fits by NUTS, PSIS-LOO by stage, final comparison with random-effects Bayesian model selection, group-level parameters | `06_fit_models_nuts.py` (S3; S4 with `--tune 4000 --draws 2000` for m08 and m12, as in reproduce.sh) | `06_model_fits/` |
| Results [42]-[45], [51]; Methods [112]; Fig. 4b,c | posterior predictive checks of every model (20 simulated data sets per model) | `10_posterior_predictive_checks.py` (S3, S4) | `10_ppc/`; the values of Fig. 4b in `figures/figure_numbers.json` (`study3_seq_checks`) |
| Results [50]; Methods [113]; Fig. 4d,e | parameter recovery of the winning model and model recovery among the three ITI models | `07_recovery_nuts.py` (S3; the commands are in reproduce.sh) | `07_recovery/` |
| Results [48], [51], [56]; Methods [95]; Fig. 4f | the exploratory-state marker u, its trend over quintiles, the leaks 0.5-0.9, the mixed models, and its posterior predictive check | `data/study_3/analysis/19_exploration_marker_ppc.py --studies 3 4` | `19_exploration_marker_ppc/` in S3 and S4 |
| Methods [95] | the marker against the explore state of a hidden Markov model of choices | `data/study_3/analysis/20_exploration_proxy_hmm.py --studies 3 4` | `20_exploration_proxy_hmm/hmm_correlation.json` in S3 and S4 |
| Results [52], [55]; Methods [96] | response times: ITI x stay x outcome (Bayesian linear mixed models) | `17_response_times.py` (S3, S4) | `17_response_times/response_times.json` |
| Results [53]; Methods [97] | impulsivity and anhedonia against the ITI effect (preregistered and exploratory) | `05_individual_differences.py` (S3) | `05_individual_differences/correlations.csv` |
| Results [53]; Methods [97] | the winning model's parameters against impulsivity and anhedonia (Studies 3 and 4, Spearman, Benjamini-Hochberg) | `data/study_3/analysis/09_parameter_trait_correlations.py` | `data/study_3/analysis/results/09_parameter_traits/` |
| Results [59]; Methods [114]-[119]; Fig. 6a | the external dataset: mixed-effects model of switching on the jittered ITI | `fMRI_study_generalization/code/prepare_tomov2020.py`, `tomov_iti_glmm.py`, `plot_fig6a_tomov.py` | `fMRI_study_generalization/results/` |
| Results [60], [61]; Methods [121]; Fig. 6b-f | language-model agents | `ai_agents/analyze_agents.py` (the runs: `ai_agents/run_agents.py`) | `ai_agents/results/`; `figures/figure_numbers.json` (`fig6`) |
| Figs. 1-4 | | `figures/make_manuscript_figures.py` | `figures/Fig1_study1`, `Fig2_study2`, `Fig3_study3`, `Fig4_modelling` |
| Fig. 5 | | `figures/make_study4_figure.py` | `figures/Fig5_study4` |
| Fig. 6 | | `figures/make_generalization_figure.py` | `figures/Fig6_generalization` |

### Supplementary Information

| Item | Analysis | Script | Output |
|---|---|---|---|
| Supplementary Fig. 1 | the ITI after the choice two, three or four trials back, when that option differs from the last one (Studies 1-4) | `16_position_controlled_contrasts.py` (S1-S4), then `figures/make_supp_fig1_earlier_iti.py` | `figures/SupplementaryFig1_earlier_ITI_other_option` |
| Supplementary Table 1 | the model-building sequence in Studies 3 and 4 | `06_fit_models_nuts.py` (S3, S4), then `figures/make_supp_table1.py` | `figures/SupplementaryTable1_model_comparison.md`, `.csv` |
| Supplementary Note 1, Supplementary Fig. 2 | only the last ITI predicts switching (Studies 3 and 4) | `16_position_controlled_contrasts.py` (S3, S4), then `figures/make_supp_fig2_lag_profile.py` | `figures/SupplementaryFig2_lag_profile` |
| Supplementary Note 2, Supplementary Fig. 3 | a slow wait expectation under the fixed ITI sequence (no data fitted) | `figures/make_supp_fig3_schedule_expectation.py` | `figures/SupplementaryFig3_schedule_expectation` |

## Conventions

* `iti` on row t of `trials.csv` is the ITI that followed trial t; the ITI before the choice on trial t (the prior ITI) is
  the `iti` of trial t - 1. A switch is scored on a trial whose previous trial was also responded.
* Outcomes are +1 (win) and -1 (loss). In the computational models the ITI is coded w = 0 after 1 s and 1 after 4 s.
* Model codes (m01-m12) and the names used in the paper are listed in `analysis_common/config.py` (`MODEL_SHORT`);
  m12, "ITI competes with value", is the winning model.
