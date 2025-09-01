# Experimentation Platform: Function Guide & Cheat Sheet

This guide provides a summary of all available functions in the experimentation platform.  
The table is ordered by the typical sequence of an analysis, from initial design to advanced estimation, to serve as a quick reference during project planning or technical discussions.

---

## Analysis Workflow & Function Map

| Stage | Function Name | What It Does | When to Use It |
|-------|---------------|--------------|----------------|
| **1. Experiment Design** | `calculate_sample_size_standard()` | Performs power analysis for binary or continuous metrics. | Before any RCT. The mandatory first step to determine required sample size and experiment duration. |
| | `calculate_sample_size_survival_mc()` | Performs power analysis for time-to-event metrics via Monte Carlo simulation. | Before a survival experiment (e.g., measuring time-to-churn) to determine sample size. |
| **2. RCT Analysis** | `analyze_rct_ate()` | Estimates the ATE for a standard A/B test using ANCOVA (CUPED) to reduce variance. | For any standard, user-level RCT where interference is not a concern. Your default RCT analyzer. |
| | `analyze_effects_with_xgboost()` | Estimates the ATE using a non-linear XGBoost model and SHAP values. | Alternative to linear models for RCTs, especially if you suspect complex interaction effects. |
| | `analyze_clustered_rct()` | Estimates the ATE for a clustered RCT using a Generalized Linear Mixed Model (GLMM). | When randomized by groups (cities, stores) to handle interference. |
| **3. Non-RCT Pre-processing** | `balance_observational_data()` | Pre-processes non-RCT data using a CEM (matching) and IPW (weighting) pipeline. | The mandatory first step for most non-RCT analyses. Its output (a balanced DataFrame) is the input for subsequent estimation functions. |
| **4. Non-RCT & Quasi-Experimental Estimation** | `estimate_observational_ate()` | Estimates the ATE on observational data using a full Balancing → AIPW pipeline. | For standard observational studies where you must assume "selection on observables." |
| | `analyze_with_meta_learner()` | Estimates the ATE using S-Learner or T-Learner (Random Forest-based). | Alternative for observational studies, good for capturing non-linearities. |
| | `estimate_observational_ate_dml_multi()` | Estimates ATEs for multiple, overlapping treatments using Double Machine Learning. | For complex observational scenarios with many simultaneous treatments and confounders. |
| | `run_did_twfe()` | Estimates the ATE from a staggered rollout using a Two-Way Fixed Effects model. | When you can leverage panel data from a staggered "natural experiment." Can be used after `balance_observational_data` for Matching-DiD. |
| | `run_iv_2sls()` | Estimates the ATE in the presence of unobserved confounding using an instrumental variable. | When you have a valid instrument that affects treatment but not the outcome directly. |
| | `run_synthetic_control()` | Estimates the effect for a single treated unit by creating a weighted counterfactual. | For case studies (e.g., measuring the impact of a policy change in a single city). |
| **5. Advanced Follow-up & Survival Analysis** | `analyze_survival_experiment()` | Measures the effect on a time-to-event outcome using a Cox Proportional Hazards model. | For any experiment (RCT or non-RCT) where the outcome is "time to X" (e.g., churn, conversion, failure). |
| | `find_heterogeneous_effects()` | Finds subgroups with different treatment effects (CATEs) using a Causal Tree. | After getting the main ATE, to answer "for whom did the treatment work best/worst?" |
