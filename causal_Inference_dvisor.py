import openai
import os
from typing import List, Dict, Any, Optional

# =============================================================================
#
# Causal Inference Advisor powered by an LLM (v2 - Live API Call)
#
# Description:
# This script uses a large language model (LLM) to recommend a sequential
# analysis plan for a user's experiment. It takes a description of the
# experiment's design and constraints as input, consults a "knowledge base"
# of available functions from the experimentation platform, and generates a
# step-by-step guide.
#
# =============================================================================

# --- CONFIGURATION ---
# Load the API key from environment variables for security.
# In your terminal, run: export OPENAI_API_KEY='your_key_here'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- KNOWLEDGE BASE ---
AVAILABLE_FUNCTIONS_KNOWLEDGE_BASE = """
# Experimentation Platform: Available Functions

## 1. Experiment Design
- `calculate_sample_size_standard()`: Power analysis for binary or continuous metrics.
- `calculate_sample_size_survival_mc()`: Power analysis for time-to-event metrics.

## 2. RCT Analysis
- `analyze_rct_ate()`: Estimates ATE for standard A/B tests using ANCOVA (CUPED).
- `analyze_effects_with_xgboost()`: Estimates ATE using XGBoost and SHAP for non-linear effects.
- `analyze_clustered_rct()`: Estimates ATE for clustered RCTs using a GLMM.

## 3. Non-RCT Pre-processing
- `balance_observational_data()`: Pre-processes non-RCT data with CEM (matching) and IPW (weighting). Returns a balanced DataFrame.

## 4. Non-RCT & Quasi-Experimental Estimation
- `estimate_observational_ate()`: Full pipeline for observational data (Balancing -> AIPW).
- `analyze_with_meta_learner()`: Estimates ATE using S-Learner or T-Learner (Random Forest-based).
- `estimate_observational_ate_dml_multi()`: Estimates ATEs for multiple overlapping treatments using DML.
- `run_did_twfe()`: Estimates ATE from a staggered rollout using a Two-Way Fixed Effects model.
- `run_iv_2sls()`: Estimates ATE using an instrumental variable.
- `run_synthetic_control()`: Estimates effect for a single treated unit.

## 5. Advanced Follow-up & Survival Analysis
- `analyze_survival_experiment()`: Measures effect on time-to-event outcomes using a Cox Proportional Hazards model.
- `find_heterogeneous_effects()`: Finds subgroups with different treatment effects using a Causal Tree.
"""

def get_causal_analysis_plan(
    is_rct: bool,
    data_design: str,
    covariates: List[str],
    has_network_effects: bool,
    has_contamination: bool,
    user_question: str
) -> str:
    """
    Makes a call to an LLM to generate a sequential causal analysis plan.

    Args:
        is_rct (bool): Whether the experiment was a randomized controlled trial.
        data_design (str): The structure of the input data.
        covariates (List[str]): A list of the available pre-treatment covariates.
        has_network_effects (bool): Whether there is a risk of spillover/interference.
        has_contamination (bool): Whether the original randomization was compromised.
        user_question (str): The user's goal in their own words.

    Returns:
        str: A markdown-formatted, step-by-step analysis plan from the LLM.
    """
    if not OPENAI_API_KEY:
        return "ERROR: OPENAI_API_KEY is not set. Please set the environment variable."

    # CORRECTED: The system prompt must match the task.
    system_prompt = f"""
    You are an expert causal inference methodologist and data science advisor.
    Your task is to provide a user with a step-by-step guide to accurately estimate the impact of a treatment.

    You have access to a Python experimentation platform with the following functions:
    --- AVAILABLE FUNCTIONS KNOWLEDGE BASE ---
    {AVAILABLE_FUNCTIONS_KNOWLEDGE_BASE}
    --- END KNOWLEDGE BASE ---

    You must analyze the user's experimental design and provide a clear, sequential plan.

    **Instructions:**
    1.  **Acknowledge the Goal:** Start by restating the user's objective.
    2.  **Determine the Correct Pathway:** Based on the user's inputs (RCT, contamination, network effects), decide on the main analytical strategy.
    3.  **Provide a Step-by-Step Plan:** List the exact sequence of functions the user should call from the knowledge base. For each step, explain *why* it is necessary.
    4.  **Handle Contamination:** If `has_contamination` is True, you MUST state that the experiment can no longer be analyzed as a simple RCT and must be treated as an observational study.
    5.  **Handle Network Effects:** If `has_network_effects` is True in an RCT, you MUST recommend `analyze_clustered_rct()` or a similar cluster-based method.
    6.  **Recommend Unavailable Methods (If Necessary):** If the ideal statistical method is more advanced than what's available (e.g., modern DiD estimators like `csdid`), still recommend it. Clearly state that it is **"not available in the current platform"** but is the academic best practice.
    7.  **Format:** Present the final plan in clear, readable markdown.
    """

    user_context = f"""
    Here is my experiment:
    - **User's Goal:** "{user_question}"
    - **Was it an RCT?** {is_rct}
    - **Was there contamination?** {has_contamination}
    - **Was there a risk of network effects/interference?** {has_network_effects}
    - **What is the data design?** "{data_design}"
    - **What are my available covariates?** {covariates}

    Please provide the best sequential plan to get a highly accurate estimate of the treatment's impact.
    """

    # --- REAL LLM CALL ---
    # This now integrates your provided logic but uses the correct system prompt.
    try:
        openai.api_key = OPENAI_API_KEY
        # NOTE: "gpt-5" is a placeholder. Use a real model name like "gpt-4" or "gpt-4o".
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM call failed: {e}"


if __name__ == '__main__':
    # --- Example Usage ---
    # Scenario 1: A clean, standard A/B test.
    print("--- Generating Plan for a Clean A/B Test ---")
    plan_rct = get_causal_analysis_plan(
        is_rct=True,
        has_contamination=False,
        has_network_effects=False,
        data_design="cross-sectional",
        covariates=["user_tenure", "past_purchases"],
        user_question="measure the impact of a new checkout button on conversion"
    )
    print(plan_rct)
    print("\n" + "="*50 + "\n")

    # Scenario 2: A complex, contaminated, staggered rollout.
    print("--- Generating Plan for a Contaminated Staggered Rollout ---")
    plan_non_rct = get_causal_analysis_plan(
        is_rct=True,
        has_contamination=True,
        has_network_effects=False,
        data_design="panel",
        covariates=["user_tenure", "team_size", "pre_experiment_productivity"],
        user_question="understand the impact of our new AI tool on engineer productivity"
    )
    print(plan_non_rct)

