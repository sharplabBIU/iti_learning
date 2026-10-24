# analysis_common

The shared analysis package. The analysis scripts in `data/study_N/analysis/` are copies of the master copies in
`study_scripts/`; a script analyses the study whose folder it sits in (`runner.py` reads the folder name). To change an
analysis, edit the master copy and copy it to the study folders that run it (`reproduce.sh` stops if a copy differs).

| module | contents |
|---|---|
| `config.py` | paths, study definitions (raw files, questionnaires and their preregistered directions), the inclusion rule (`EXCLUSION_MAIN = "min90"`: sessions that reached trial 90), model names (`MODEL_SHORT`, `MODEL_ABBR`) |
| `data.py` | raw export -> `data/trials.csv`, `data/participants.csv`, `data/exclusion_flow.csv` with pseudonymous IDs (`prepare_study`, run by `00_prepare_data.py`); `load_study(key)`; `subjects_for_fitting` |
| `behavior.py` | switch table with lagged ITIs, choices and outcomes; paired tests of p(switch) after 1 s vs 4 s; participant-level ITI effects; correlations |
| `bayes_glmm.py` | Bayesian mixed-effects logistic and linear regressions in PyMC (every mixed model of the paper), and the repeated fits of the posterior predictive checks |
| `likelihoods.py` | the reinforcement-learning models of the model sequence (m01-m12), their likelihood and simulator in numpy |
| `pymc_models.py` | the same likelihoods in PyTensor, hierarchical priors, sampling with nutpie (NUTS), PSIS-LOO |
| `bms.py` | random-effects Bayesian model selection (expected frequencies, exceedance and protected exceedance probabilities, Bayesian omnibus risk) |
| `simulate.py` | simulated data sets from fitted models (posterior predictive checks, recovery) |
| `runner.py` | the helper every analysis script uses: study, main sample, results folder |

| script (`study_scripts/`) | run for | what it does |
|---|---|---|
| `00_prepare_data.py` | S1-S4 | the tidy tables from the raw exports (not shared; not run by reproduce.sh) |
| `01_learning_check.py` | S1-S4 | chose-best rate against chance; the participant flow |
| `02_iti_switching.py` | S1-S4 | paired tests of the ITI effect; the preregistered mixed model with the linear trial term, by outcome and with the interaction |
| `02b_iti_switching_phase_by_outcome.py` | S1-S4 | the ITI effect after wins and losses in each half of the session |
| `05_individual_differences.py` | S3 | impulsivity and anhedonia against the ITI effect |
| `06_fit_models_nuts.py` | S3, S4 (S2: m08 only) | the model sequence by NUTS, compared by PSIS-LOO, with Bayesian model selection |
| `07_recovery_nuts.py` | S3 | parameter and model recovery |
| `09_parameter_trait_correlations.py` | S3 (pools S3 and S4) | the winning model's parameters against the questionnaires |
| `10_posterior_predictive_checks.py` | S3, S4 | posterior predictive checks of every model |
| `11_rdm_task.py` | S2 | accuracy on the dot-motion task |
| `13_design_confound.py` | S3, S4 | the fixed ITI sequence and the decline in switching |
| `16_position_controlled_contrasts.py` | S1-S4 | the ITI after the choice two, three and four trials back |
| `17_response_times.py` | S3, S4 | response times by ITI, stay and outcome |
| `19_exploration_marker_ppc.py` | S3 (analyses S3 and S4) | the exploratory-state marker and its posterior predictive check |
| `20_exploration_proxy_hmm.py` | S3 (analyses S3 and S4) | the marker against a hidden Markov model of choices |

PyTensor's C compiler is not needed: set `PYTENSOR_FLAGS=cxx=` (reproduce.sh does) and the samplers are compiled by numba
through nutpie.
