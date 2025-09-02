import openai
import os
from typing import List, Dict, Any, Optional

# =============================================================================
#
# LLM-Powered Function Generator & Script Assembler
#
# Description:
# This script contains two LLM-powered agents:
# 1. A "Generator" that writes Python code for missing causal methods.
# 2. An "Assembler" that takes the output of the Generator and a high-level
#    plan to create a complete, sequential, and runnable analysis script.
#
# =============================================================================

# --- KNOWLEDGE BASE ---
# Represents the platform's current capabilities.
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
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY is not set."

    system_prompt_for_coder = f"""
    You are an expert Python developer specializing in implementing causal inference libraries.
    Your task is to write complete, production-ready Python functions based on a methodologist's recommendation.
    You have been given a recommended analysis plan and a list of functions that are *already available* in our platform.
    Your job is to identify any methods mentioned in the plan that are **missing** from the available functions list and write the Python code for them.

    **Instructions:**
    1.  **Identify Missing Functions:** Carefully read the "Recommended Analysis Plan" and compare it against the "Available Functions Knowledge Base".
    2.  **Generate Code:** For each missing method, write a complete Python function.
    3.  **Adhere to Platform Standards:** The generated functions MUST be self-contained, accept a pandas DataFrame, be well-documented, and return a dictionary of results.
    4.  **Handle No Missing Functions:** If all recommended methods are already in the knowledge base, return a Python comment block stating that.
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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt_for_coder},
                {"role": "user", "content": user_prompt_for_coder}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"# LLM call failed: {e}"

def assemble_analysis_script(
    llm_analysis_plan: str,
    generated_functions_code: str,
    openai_api_key: Optional[str] = None
) -> str:
    """
    Takes the analysis plan and newly generated code to assemble a complete,
    runnable Python script for the user.
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY is not set."

    system_prompt_for_assembler = f"""
    You are an expert Python data scientist who writes clean, runnable analysis scripts.
    Your task is to assemble a complete script based on a high-level plan and blocks of pre-written code.

    **Instructions:**
    1.  **Understand the Sequence:** Read the "Recommended Analysis Plan" to understand the logical flow of the analysis.
    2.  **Structure the Script:** Create a Python script with the following sections:
        - Imports (pandas, numpy, and functions from `experimentation_platform.py`).
        - The "Newly Generated Functions" code block, if any.
        - A "Load Data" section with a placeholder for the user's DataFrame.
        - "Define Variables" section for columns like treatment, outcome, etc.
        - A step-by-step execution of the analysis plan.
    3.  **Narrate the Script:** For each step in the execution, copy the markdown text from the "Recommended Analysis Plan" and place it as a comment block to explain what's happening and why.
    4.  **Call the Functions:** After each comment block, write the corresponding Python function call (e.g., `balanced_df = balance_observational_data(...)`). Use placeholder variable names.
    5.  **Output Format:** Return the complete script as a single, clean Python code block.
    """

    user_prompt_for_assembler = f"""
    Here is the context:

    --- AVAILABLE FUNCTIONS KNOWLEDGE BASE (for import reference) ---
    {AVAILABLE_FUNCTIONS_KNOWLEDGE_BASE}
    --- END KNOWLEDGE BASE ---

    --- NEWLY GENERATED FUNCTIONS (to be included in the script) ---
    {generated_functions_code}
    --- END NEWLY GENERATED FUNCTIONS ---

    --- RECOMMENDED ANALYSIS PLAN (this is your guide) ---
    {llm_analysis_plan}
    --- END RECOMMENDED ANALYSIS PLAN ---

    Please generate the complete, runnable Python script that executes this plan.
    """
    try:
        openai.api_key = api_key
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt_for_assembler},
                {"role": "user", "content": user_prompt_for_assembler}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"# LLM call failed: {e}"


if __name__ == '__main__':
    # --- Example: Full End-to-End Workflow ---
    # Step 1: An "Advisor" LLM generates a high-level plan.
    sample_advisor_plan = """
Hello! Your goal is to **"understand the impact of our new AI tool on engineer productivity"**.
Based on your inputs, the best approach is a Difference-in-Differences (DiD) analysis. For the highest accuracy, it is academic best practice to use a modern estimator like Callaway & Sant'Anna (`csdid`).

Here is your recommended sequential analysis plan:

### **Step 1: Core Estimation - Modern Difference-in-Differences**
**Why this is necessary:** This method correctly handles staggered rollouts where treatment effects can vary over time.
* **Recommended Method:** Use the Callaway & Sant'Anna estimator (`csdid`).
* **Platform Availability:** **Note: This method is not available in the current platform.**

### **Step 2 (Follow-up): Understand Heterogeneity**
**Why this is necessary:** To find which user segments respond most differently to the treatment.
* **Function to call:** `find_heterogeneous_effects()`
"""
    print("--- 1. Advisor Plan Generated ---")
    print(sample_advisor_plan)

    # Step 2: A "Generator" LLM writes the code for the missing function.
    print("\n--- 2. Generating Code for Missing Functions ---")
    generated_code = generate_missing_functions(
        llm_analysis_plan=sample_advisor_plan
    )
    print(generated_code)

    # Step 3: An "Assembler" LLM takes the plan and new code to create a final script.
    print("\n--- 3. Assembling the Final, Runnable Script ---")
    final_script = assemble_analysis_script(
        llm_analysis_plan=sample_advisor_plan,
        generated_functions_code=generated_code
    )
    print(final_script)
