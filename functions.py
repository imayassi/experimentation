 # =============================================================================
#
# A Modular Python Platform for Causal Inference & Experimentation (v3)
#
# =============================================================================
#
# Author: Gemini
# Description:
# This script provides a comprehensive, modular toolkit for conducting causal
# analysis. It includes functions for designing and analyzing Randomized
# Controlled Trials (RCTs) of varying complexity, as well as a suite of
# quasi-experimental and observational methods for when randomization is not
# possible. This version includes standardized inputs, pipelining capabilities,
# a dedicated survival analysis module, and an XGBoost+SHAP estimator.
#
# =============================================================================

from __future__ import annotations
import warnings
from typing import List, Optional, Dict, Tuple, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

# Core stats and ML libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline

# Specialized causal inference libraries
from linearmodels.panel import PanelOLS
import rdd
from causalml.inference.tree import CausalTreeRegressor
from lifelines import CoxPHFitter

# Advanced modeling libraries
import xgboost as xgb
import shap

# --- UTILITY FUNCTIONS ---

def _split_numeric_categorical(df: pd.DataFrame, cols: List[str]) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2:
             num_cols.append(c)
        else:
             cat_cols.append(c)
    return num_cols, cat_cols

# =============================================================================
# MODULE 1: EXPERIMENT DESIGN
# =============================================================================

def calculate_sample_size_standard(
    metric_type: str, baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.8
) -> Dict[str, Any]:
    """Performs power analysis for continuous or binary metrics."""
    if metric_type == "binary":
        effect_size = proportion_effectsize(baseline_rate, baseline_rate + mde)
        analysis = NormalIndPower()
    elif metric_type == "continuous":
        effect_size = mde / baseline_rate
        analysis = TTestIndPower()
    else:
        raise ValueError("metric_type must be 'continuous' or 'binary'")
    sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0, alternative='two-sided')
    return {
        "sample_size_per_group": int(np.ceil(sample_size)),
        "total_sample_size": int(np.ceil(sample_size * 2)),
    }

def calculate_sample_size_survival_mc(
    baseline_hazard: float, expected_hazard_ratio: float, follow_up_time: float,
    n_simulations: int = 500, alpha: float = 0.05, power: float = 0.8
) -> Dict[str, Any]:
    """Calculates sample size for survival experiments using Monte Carlo simulation."""
    sample_sizes = np.arange(100, 10000, 200)
    for n in tqdm(sample_sizes, desc="Simulating sample sizes"):
        p_values = []
        for _ in range(n_simulations):
            t = np.random.choice([0, 1], size=n)
            control_time = np.random.exponential(1 / baseline_hazard, size=n)
            treatment_time = np.random.exponential(1 / (baseline_hazard * expected_hazard_ratio), size=n)
            time_to_event = np.where(t == 1, treatment_time, control_time)
            observed_time = np.minimum(time_to_event, follow_up_time)
            event_observed = (time_to_event <= follow_up_time).astype(int)
            sim_df = pd.DataFrame({'T': observed_time, 'E': event_observed, 'group': t})
            try:
                cph = CoxPHFitter().fit(sim_df, 'T', 'E', formula="group")
                p_values.append(cph.p_value_['group'])
            except: continue
        current_power = np.mean(np.array(p_values) < alpha)
        if current_power >= power:
            return {
                "method": "Monte Carlo Simulation for Survival",
                "estimated_total_sample_size": n,
                "achieved_power": current_power
            }
    return {"error": "Failed to reach desired power with max sample size."}

# =============================================================================
# MODULE 2: RCT ANALYSIS
# =============================================================================

def analyze_rct_ate(
    df: pd.DataFrame, treatment_cols: List[str], outcome_col: str, covariate_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Analyzes a standard A/B/n test (RCT) using ANCOVA (CUPED)."""
    formula = f"{outcome_col} ~ {' + '.join(treatment_cols)}"
    if covariate_cols:
        formula += " + " + " + ".join(covariate_cols)
    model = smf.ols(formula, data=df).fit(cov_type='HC1')
    results = {"method": "ANCOVA (CUPED)" if covariate_cols else "Simple OLS", "summary": str(model.summary())}
    for treat in treatment_cols:
        results[f"ate_{treat}"] = model.params[treat]
        results[f"ci95_{treat}"] = model.conf_int().loc[treat].tolist()
        results[f"pvalue_{treat}"] = model.pvalues[treat]
    return results

def analyze_effects_with_xgboost(
    df: pd.DataFrame,
    treatment_cols: List[str],
    outcome_col: str,
    covariate_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyzes an experiment using an XGBoost model and SHAP for effect estimation.

    This provides a non-linear alternative to linear regression for estimating marginal effects.
    In an RCT, the average SHAP value for a treatment is a valid estimate of the ATE.
    In an observational study, this should be run on data that has been balanced first.

    Args:
        df (pd.DataFrame): DataFrame for analysis. Can be raw RCT data or balanced observational data.
        treatment_cols (List[str]): Treatment columns.
        outcome_col (str): Outcome column.
        covariate_cols (Optional[List[str]]): Covariate columns.

    Returns:
        Dict[str, Any]: Dictionary with the estimated marginal effect (ATE) for each treatment.
    """
    y = df[outcome_col]
    feature_cols = treatment_cols + (covariate_cols or [])
    X = df[feature_cols].copy()

    # Handle categorical features using one-hot encoding
    num_cols, cat_cols = _split_numeric_categorical(df, feature_cols)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
        ],
        remainder='passthrough'
    )
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding for SHAP summary
    try:
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        processed_feature_names = num_cols + ohe_feature_names.tolist()
    except: # For older scikit-learn versions
        processed_feature_names = feature_cols # Won't be perfect but avoids crashing

    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names, index=X.index)

    # Train XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=250, early_stopping_rounds=10, random_state=42)
    # XGBoost needs a validation set for early stopping
    eval_set = [(X_processed_df, y)]
    model.fit(X_processed_df, y, eval_set=[(X_processed_df, y)], verbose=False)


    # Explain model with SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_processed_df)

    results = {"method": "XGBoost with SHAP"}
    
    # The mean of the SHAP values for a feature is its average marginal impact on the prediction
    # For a binary treatment in an RCT, this is the ATE.
    for i, col in enumerate(X_processed_df.columns):
        is_treatment_col = any(treat_col == col or col.startswith(treat_col + "_") for treat_col in treatment_cols)
        if is_treatment_col:
            mean_shap = np.mean(shap_values.values[:, i])
            results[f"ate_{col}"] = mean_shap
            
    return results

def analyze_clustered_rct(
    df: pd.DataFrame, treatment_cols: List[str], outcome_col: str, cluster_col: str, covariate_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Analyzes a clustered RCT using a Generalized Linear Mixed Model (GLMM)."""
    formula = f"{outcome_col} ~ {' + '.join(treatment_cols)}"
    if covariate_cols:
        formula += " + " + " + ".join(covariate_cols)
    model = smf.mixedlm(formula, data=df, groups=df[cluster_col]).fit()
    results = {"method": "GLMM for Clustered RCT", "summary": str(model.summary())}
    for treat in treatment_cols:
        results[f"ate_{treat}"] = model.params[treat]
        results[f"ci95_{treat}"] = model.conf_int().loc[treat].tolist()
        results[f"pvalue_{treat}"] = model.pvalues[treat]
    return results

# =============================================================================
# MODULE 3: OBSERVATIONAL DATA BALANCING (FOR PIPELINING)
# =============================================================================

def balance_observational_data(
    df: pd.DataFrame, treatment_col: str, covariate_cols: List[str]
) -> pd.DataFrame:
    """
    Pre-processes observational data using CEM and IPW to create a balanced sample.
    """
    # ... [Implementation from previous version, no changes needed] ...
    work = df.copy()
    cem_bins = 4
    for c in covariate_cols:
        if pd.api.types.is_numeric_dtype(work[c]):
             work[f"_cem_{c}"] = pd.qcut(work[c], q=min(cem_bins, work[c].nunique()), duplicates="drop", labels=False).astype(str)
        else:
             work[f"_cem_{c}"] = work[c].astype(str)
    bin_cols = [f"_cem_{c}" for c in covariate_cols]
    work["_cem_stratum"] = work[bin_cols].agg("|".join, axis=1)
    vc = work.groupby("_cem_stratum")[treatment_col].nunique()
    good_strata = vc[vc >= 2].index
    pruned_df = work[work["_cem_stratum"].isin(good_strata)].copy()
    X = pruned_df[covariate_cols]
    y = pruned_df[treatment_col]
    num, cat = _split_numeric_categorical(pruned_df, covariate_cols)
    transformers = []
    if num: transformers.append(("num", StandardScaler(), num))
    if cat: transformers.append(("cat", OneHotEncoder(handle_unknown='ignore', drop='first'), cat))
    if not transformers: raise ValueError("No valid covariates found for propensity model.")
    preproc = ColumnTransformer(transformers)
    model = Pipeline([("pre", preproc), ("clf", LogisticRegression(class_weight='balanced', max_iter=1000))])
    model.fit(X, y)
    e_hat_probs = model.predict_proba(X)
    e1 = e_hat_probs[:, 1]
    p_treat = np.mean(y)
    ipw = np.where(y == 1, p_treat / e1, (1 - p_treat) / (1 - e1))
    pruned_df['ipw'] = ipw
    print(f"Original N: {len(df)}, Pruned N: {len(pruned_df)}")
    return pruned_df.drop(columns=bin_cols + ["_cem_stratum"])


# =============================================================================
# MODULE 4: OBSERVATIONAL & QUASI-EXPERIMENTAL ANALYSIS
# =============================================================================

def estimate_observational_ate(
    df: pd.DataFrame, treatment_col: str, outcome_col: str, covariate_cols: List[str]
) -> Dict[str, Any]:
    """Estimates ATE from observational data using a Balancing -> AIPW pipeline."""
    balanced_df = balance_observational_data(df, treatment_col, covariate_cols)
    t = balanced_df[treatment_col].values
    y_out = balanced_df[outcome_col].values
    X = balanced_df[covariate_cols]
    num, cat = _split_numeric_categorical(balanced_df, covariate_cols)
    transformers = []
    if num: transformers.append(("num", StandardScaler(), num))
    if cat: transformers.append(("cat", OneHotEncoder(handle_unknown='ignore', drop='first'), cat))
    preproc = ColumnTransformer(transformers)
    Xp = preproc.fit_transform(X)
    m1 = LinearRegression().fit(Xp[t == 1], y_out[t == 1], sample_weight=balanced_df['ipw'][t==1])
    m0 = LinearRegression().fit(Xp[t == 0], y_out[t == 0], sample_weight=balanced_df['ipw'][t==0])
    mu1, mu0 = m1.predict(Xp), m0.predict(Xp)
    e_hat_model = LogisticRegression(class_weight='balanced').fit(Xp, t)
    e1 = e_hat_model.predict_proba(Xp)[:, 1]
    psi = (mu1 - mu0) + t * (y_out - mu1) / (e1 + 1e-12) - (1 - t) * (y_out - mu0) / (1 - e1 + 1e-12)
    ate = float(np.mean(psi))
    return {"method": "Observational ATE (CEM + IPW + AIPW)", "ate": ate}

def run_did_twfe(
    df: pd.DataFrame, treatment_cols: List[str], outcome_col: str, index_cols: List[str],
    time_varying_covariates: Optional[List[str]] = None, weights_col: Optional[str] = None
) -> Dict[str, Any]:
    """Analyzes a staggered rollout using a TWFE model, with optional weights for Matching-DiD."""
    entity_col, time_col = index_cols[0], index_cols[1]
    df = df.set_index([entity_col, time_col])
    formula = f"{outcome_col} ~ {' + '.join(treatment_cols)} + EntityEffects + TimeEffects"
    if time_varying_covariates:
        formula += " + " + " + ".join(time_varying_covariates)
    weights = df[weights_col] if weights_col else None
    model = PanelOLS.from_formula(formula, data=df, weights=weights)
    results = model.fit(cov_type='clustered', cluster_entity=True)
    output = {"method": "Matching-DiD" if weights_col else "DiD (TWFE)", "summary": str(results)}
    for treat in treatment_cols:
        output[f"ate_{treat}"] = results.params[treat]
        output[f"ci95_{treat}"] = results.conf_int().loc[treat].tolist()
        output[f"pvalue_{treat}"] = results.pvalues[treat]
    return output

# =============================================================================
# MODULE 5: ADVANCED FOLLOW-UP & SURVIVAL ANALYSIS
# =============================================================================

def analyze_survival_experiment(
    df: pd.DataFrame, treatment_cols: List[str], duration_col: str, event_col: str,
    covariate_cols: Optional[List[str]] = None, weights_col: Optional[str] = None
) -> Dict[str, Any]:
    """Analyzes a survival experiment using a Cox Proportional Hazards model."""
    formula = " + ".join(treatment_cols)
    if covariate_cols:
        formula += " + " + " + ".join(covariate_cols)
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col, formula=formula, weights_col=weights_col)
    summary = cph.summary.reset_index()
    results = {"method": "Cox Proportional Hazards", "summary": str(cph.summary)}
    for treat in treatment_cols:
        row = summary[summary['covariate'] == treat]
        if not row.empty:
            results[f"hazard_ratio_{treat}"] = row['exp(coef)'].iloc[0]
            results[f"ci95_{treat}"] = row[['exp(coef) lower 95', 'exp(coef) upper 95']].iloc[0].tolist()
            results[f"pvalue_{treat}"] = row['p'].iloc[0]
    return results

def find_heterogeneous_effects(
    df: pd.DataFrame, treatment_col: str, outcome_col: str, feature_cols: List[str]
) -> Dict[str, Any]:
    """Finds subgroups with different treatment effects using a Causal Tree."""
    y = df[outcome_col]
    treatment = df[treatment_col]
    X = df[feature_cols]
    treatment_str = treatment.map({0: 'control', 1: 'treatment'}).astype(str)
    causal_tree = CausalTreeRegressor(max_depth=4, min_samples_leaf=int(len(df)*0.05))
    causal_tree.fit(X=X, treatment=treatment_str, y=y)
    return {
        "method": "Causal Tree for Heterogeneous Effects (CATE)",
        "fitted_tree": causal_tree,
        "leaf_summary": causal_tree.summary()
    }

