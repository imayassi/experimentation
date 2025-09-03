import openai
import os
from typing import List, Dict, Any, Optional

# =============================================================================
#
# Causal Inference Advisor powered by an LLM (v2)
#
# Description:
# This version is upgraded to accept a structured data schema (column_map) and
# prioritize advanced, non-parametric, multi-treatment estimators. The LLM
# is now instructed to recommend the most robust methods like Double Machine
# Learning (DML) with tree-based models as a first-class citizen.
#
# =============================================================================

# --- KNOWLEDGE BASE ---
# Represents the platform's current capabilities.
AVAILABLE_FUNCTIONS_KNOWLEDGE_BASE = """
# Experimentation Platform: Available Functions

## 1. Experiment Design
- `calculate_sample_size_standard()`
- `calculate_sample_size_survival_mc()`

## 2. RCT Analysis
- `analyze_rct_ate()`: Standard ANCOVA (CUPED).
- `analyze_effects_with_xgboost()`: Non-linear estimation with XGBoost and SHAP.
- `analyze_clustered_rct()`: For clustered RCTs using a GLMM.

## 3. Non-RCT Pre-processing
- `balance_observational_data()`: Pre-processes data with CEM (matching) and IPW (weighting).

## 4. Non-RCT & Quasi-Experimental Estimation
- `estimate_observational_ate()`: Full pipeline for observational data (Balancing -> AIPW).
- `analyze_with_meta_learner()`: S-Learner or T-Learner (Random Forest-based).
- `estimate_observational_ate_dml_multi()`: Estimates ATEs for multiple overlapping treatments using DML.
- `run_did_twfe()`: Estimates ATE from a staggered rollout using a Two-Way Fixed Effects model.
- `run_iv_2sls()`: Estimates ATE using an instrumental variable.
- `run_synthetic_control()`: Estimates effect for a single treated unit.

## 5. Advanced Follow-up & Survival Analysis
- `analyze_survival_experiment()`: Measures effect on time-to-event outcomes.
- `find_heterogeneous_effects()`: Finds subgroups with a Causal Tree.
"""

def get_causal_analysis_plan(
    is_rct: bool,
    data_design: str,
    column_map: Dict[str, List[str]],
    has_network_effects: bool,
    has_contamination: bool,
    user_question: str,
    openai_api_key: Optional[str] = None
) -> str:
    """
    Makes a call to an LLM to generate a sequential causal analysis plan with a
    preference for robust, tree-based, multi-treatment methods.

    Args:
        is_rct (bool): Whether the experiment was a randomized controlled trial.
        data_design (str): The structure of the input data.
        column_map (Dict[str, List[str]]): A dictionary mapping column types to lists of column names.
        has_network_effects (bool): Whether there is a risk of spillover/interference.
        has_contamination (bool): Whether the original randomization was compromised.
        user_question (str): The user's goal in their own words.
        openai_api_key (Optional[str]): OpenAI API key.

    Returns:
        str: A markdown-formatted, step-by-step analysis plan from the LLM.
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY is not set. Please provide it or set the environment variable."

    # The new system prompt prioritizes advanced methods.
    system_prompt = f"""
    You are an expert causal inference methodologist and data science advisor.
    Your task is to provide a user with a step-by-step guide to accurately estimate treatment effects, with a strong preference for modern, robust methods.

    **Your primary directive is to prioritize estimators that are non-parametric (tree-based) and can handle multiple treatments and high-dimensional covariates.** Methods like Double Machine Learning (DML) should be your default recommendation for complex observational studies.

    You have access to a Python experimentation platform with the following functions:
    --- AVAILABLE FUNCTIONS KNOWLEDGE BASE ---
    {AVAILABLE_FUNCTIONS_KNOWLEDGE_BASE}
    --- END KNOWLEDGE BASE ---

    You must analyze the user's experimental design and provide a clear, sequential plan.

    **Instructions:**
    1.  **Acknowledge the Goal:** Start by restating the user's objective.
    2.  **Interpret the `column_map`:**
        - `treatments`: These are the primary causal variables of interest. If there are multiple, you MUST recommend a multi-treatment estimator like `estimate_observational_ate_dml_multi`.
        - `treatment_related_features`: These signal advanced designs. If you see a feature like 'intensity' or 'dose', recommend a dose-response analysis. If you see a 'pre_post_flag', recommend a DiD-style analysis (`run_did_twfe`).
        - `covariates`: These are the confounders that must be controlled for.
        - `target`: This is the outcome variable.
    3.  **Recommend the Best Sequence:** Provide a step-by-step plan. For non-RCTs, this plan should almost always start with `balance_observational_data()`.
    4.  **Prioritize Robust Methods:** Given the user's preference, your default recommendation for a complex non-RCT with multiple treatments should be `estimate_observational_ate_dml_multi`. For simpler RCTs, recommend `analyze_effects_with_xgboost()` to capture non-linearities.
    5.  **Handle Contamination/Network Effects:** If `has_contamination` is True, treat the study as observational. If `has_network_effects` is True, recommend `analyze_clustered_rct()`.
    6.  **Recommend Unavailable Methods (If Necessary):** If the ideal method is missing (e.g., Generalized Random Forests for dose-response), recommend it as the academic best practice and state that it's **"not available in the current platform"**.
    7.  **Format:** Present the final plan in clear, readable markdown.
    """

    user_context = f"""
    Here is my experiment:
    - **User's Goal:** "{user_question}"
    - **Was it an RCT?** {is_rct}
    - **Was there contamination?** {has_contamination}
    - **Was there a risk of network effects/interference?** {has_network_effects}
    - **What is the data design?** "{data_design}"
    - **Data Schema (`column_map`):** {column_map}

    Please provide the best sequential plan to get a highly accurate estimate of the treatment's impact, prioritizing tree-based and multi-treatment methods.
    """

    try:
        openai.api_key = api_key
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
    # --- Example Usage with the New Input Format ---
    print("--- Generating Plan for a Complex, Multi-Treatment Observational Study ---")

    # Define the experiment using the new structured inputs
    is_rct_input = False
    data_design_input = "panel"
    column_map_input = {
        'treatments': ['chat_on', 'copilot_on', 'ia_on'],
        'treatment_related_features': ['copilot_intensity_n'],
        'covariates': ['age', 'tenure', 'region', 'y_pre', 'team_id'],
        'target': ['efficiency']
    }
    user_question_input = """
    I need to estimate the independent causal impact of 3 overlapping AI tools
    on engineer efficiency from a staggered, non-random rollout. I also want to
    understand if using Copilot more intensely has a bigger effect.
    """

    analysis_plan = get_causal_analysis_plan(
        is_rct=is_rct_input,
        data_design=data_design_input,
        column_map=column_map_input,
        has_network_effects=False,
        has_contamination=False,
        user_question=user_question_input
    )

    from IPython.display import display, Markdown
    display(Markdown(analysis_plan))

