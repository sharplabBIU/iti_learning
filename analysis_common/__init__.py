"""Shared analysis code for the project on how the inter-trial interval (ITI) affects switching.

The analysis scripts in data/study_N/analysis/ (copies of analysis_common/study_scripts/) import this
package. The study a script analyses is given by the folder it sits in.

Modules
  config       paths, study definitions, the inclusion rule and model labels.
  data         raw exports -> tidy trial and participant tables with pseudonymous IDs, the
               participant flow, and the loaders every analysis uses.
  behavior     switching tables, paired tests of the ITI effect, participant-level ITI effects and
               their correlations with questionnaire scores.
  bayes_glmm   Bayesian generalized linear mixed models (GLMMs) in PyMC; every mixed model of the
               paper is fitted this way.
  likelihoods  the reinforcement-learning models of the paper's model sequence (value learning,
               choice bias, novelty, trend over the session, and the three ITI models), in numpy.
  pymc_models  the same likelihoods in PyTensor, with hierarchical priors, sampled by the No-U-Turn
               sampler (NUTS) in nutpie and compared by Pareto-smoothed importance-sampling
               leave-one-out cross-validation (PSIS-LOO).
  bms          random-effects Bayesian model selection: protected exceedance probability (PXP)
               and Bayesian omnibus risk (BOR).
  simulate     generative simulators for posterior predictive checks and recovery.
  runner       the helper every analysis script uses: study, sample and output folders.
"""
