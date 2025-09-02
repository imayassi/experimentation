import openai
import os
from typing import List, Dict, Any, Optional

# =============================================================================
#
# LLM-Powered Function Generator
#
# Description:
# This script defines a function that takes a recommended analysis plan from
# the 'causal_advisor' LLM and generates Python code for any recommended
# methodologies that are not currently available in the experimentation platform.
#
# =============================================================================

# --- KNOWLEDGE BASE ---
# This is the same knowledge base used by the advisor, representing the
# platform's current capabilities.
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
- `balance_observational_data()`: Pre-processes non-RCT data with CEM (matching) and IPW (weighting).

## 4. Non-RCT & Quasi-Experimental Estimation
- `estimate_observational_ate()`: Full pipeline for observational data (Balancing -> AIPW).
- `analyze_with_meta_learner()`: Estimates ATE using S-Learner or T-Learner.
- `estimate_observational_ate_dml_multi()`: Estimates ATEs for multiple overlapping treatments using DML.
- `run_did_twfe()`: Estimates ATE from a staggered rollout using a Two-Way Fixed Effects model.
- `run_iv_2sls()`: Estimates ATE using an instrumental variable.
- `run_synthetic_control()`: Estimates effect for a single treated unit.

## 5. Advanced Follow-up & Survival Analysis
- `analyze_survival_experiment()`: Measures effect on time-to-event outcomes.
- `find_heterogeneous_effects()`: Finds subgroups with a Causal Tree.
"""

def generate_missing_functions(
    llm_analysis_plan: str,
    openai_api_key: Optional[str] = None
) -> str:
    """
    Takes an analysis plan from the advisor LLM and generates Python code
    for any recommended methods missing from the platform.

    Args:
        llm_analysis_plan (str): The markdown string returned by the `get_causal_analysis_plan` function.
        openai_api_key (Optional[str]): Your OpenAI API key. If None, it will try to use the
                                       'OPENAI_API_KEY' environment variable.

    Returns:
        str: A string containing the generated Python code for the missing functions,
             ready to be added to the platform.
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY is not set. Please provide it or set the environment variable."

    # This prompt instructs a new LLM instance to act as a coder.
    system_prompt_for_coder = f"""
    You are an expert Python developer specializing in implementing causal inference libraries like `did`, `econml`, `statsmodels`, and `linearmodels`.
    Your task is to write complete, production-ready Python functions based on a methodologist's recommendation.

    You have been given a recommended analysis plan and a list of functions that are *already available* in our platform.
    Your job is to identify any methods mentioned in the plan that are **missing** from the available functions list and write the Python code for them.

    **Instructions:**
    1.  **Identify Missing Functions:** Carefully read the "Recommended Analysis Plan" and compare it against the "Available Functions Knowledge Base".
    2.  **Generate Code:** For each missing method, write a complete Python function.
    3.  **Adhere to Platform Standards:** The generated functions MUST:
        - Be fully self-contained, including all necessary imports (e.g., `from did.did import csdid`).
        - Accept a pandas DataFrame as the primary input, along with clear arguments for column names (e.g., `treatment_col`, `outcome_col`, `index_cols`).
        - Be well-documented with a docstring explaining what it does, its parameters, and what it returns.
        - Return a dictionary of results, including at least the ATE, p-value, and confidence interval.
    4.  **Handle No Missing Functions:** If all recommended methods are already in the knowledge base, simply return a Python comment block stating that, e.g., `# All recommended functions are already available.`
    5.  **Output Format:** Wrap all generated Python code in a single, clean markdown code block.
    """

    user_prompt_for_coder = f"""
    Here is the context:

    --- AVAILABLE FUNCTIONS KNOWLEDGE BASE ---
    {AVAILABLE_FUNCTIONS_KNOWLEDGE_BASE}
    --- END KNOWLEDGE BASE ---


    --- RECOMMENDED ANALYSIS PLAN ---
    {llm_analysis_plan}
    --- END RECOMMENDED ANALYSIS PLAN ---

    Please generate the Python code for any missing functions as per your instructions.
    """

    try:
        openai.api_key = api_key
        resp = openai.chat.completions.create(
            model="gpt-4o", # Using a powerful model for code generation
            messages=[
                {"role": "system", "content": system_prompt_for_coder},
                {"role": "user", "content": user_prompt_for_coder}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"# LLM call failed: {e}"


if __name__ == '__main__':
    # --- Example Usage ---
    # This is a sample plan that an "advisor" LLM might generate, specifically
    # recommending a modern DiD estimator that is NOT in our knowledge base.

    sample_advisor_plan = """
Hello! Your goal is to **"understand the impact of our new AI tool on engineer productivity"**.

Based on your inputs, the best approach is a Difference-in-Differences (DiD) analysis on your panel data. While the platform has a standard TWFE model (`run_did_twfe`), for the highest accuracy with a staggered rollout, it is academic best practice to use a modern estimator robust to treatment effect heterogeneity.

Here is your recommended sequential analysis plan:

### **Step 1: Core Estimation - Modern Difference-in-Differences**

**Why this is necessary:** This method correctly handles staggered rollouts where treatment effects can vary over time, which is a known issue for traditional TWFE models.

* **Recommended Method:** Use the Callaway & Sant'Anna estimator (`csdid`).
* **Platform Availability:** **Note: This method is not available in the current platform.** It is the recommended best practice.

### **Step 2 (Follow-up): Understand Heterogeneity**
...
"""

    print("--- Generating Code for Missing Functions Based on Advisor's Plan ---")
    print("\n[ADVISOR'S PLAN]:")
    print(sample_advisor_plan)
    
    generated_code = generate_missing_functions(
        llm_analysis_plan=sample_advisor_plan
    )

    print("\n" + "="*50 + "\n")
    print("[LLM CODER'S OUTPUT]:")
    print(generated_code)
