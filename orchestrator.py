import pandas as pd
import numpy as np
import os
import io
from contextlib import redirect_stdout
from typing import List, Dict, Any, Optional

# Import the LLM-powered agents from your other files
from causal_Inference_dvisor import *
from llm_powered_function_generator import *

# =============================================================================
#
# Causal Analysis Orchestrator
#
# Description:
# This script defines a high-level orchestrator function that manages the
# entire end-to-end causal analysis pipeline. It uses a chain of LLM calls
# to generate and execute a custom analysis plan, producing a final, narrated
# report with results from each step.
#
# =============================================================================

def orchestrate_causal_analysis(
    df: pd.DataFrame,
    is_rct: bool,
    data_design: str,
    platform_filepaths: List[str],
    has_network_effects: bool,
    has_contamination: bool,
    user_question: str,
    column_map: Dict[str, List[str]],
    openai_api_key: Optional[str] = None
) -> str:
    """
    Orchestrates the full end-to-end causal analysis pipeline.

    Args:
        df (pd.DataFrame): The input DataFrame for analysis.
        All other args are passed to the respective LLM agents.

    Returns:
        str: A string containing the complete, narrated output of the
             dynamically generated and executed analysis script.
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "ERROR: OPENAI_API_KEY is not set."

    # --- LLM Chain Step 1: The Advisor ---
    print("Step 1: Consulting the Causal Advisor for a high-level plan...")
    analysis_plan = get_causal_analysis_plan(
        is_rct=is_rct,
        data_design=data_design,
        platform_filepaths=platform_filepaths,
        has_network_effects=has_network_effects,
        has_contamination=has_contamination,
        user_question=user_question,
        column_map=column_map,
        df_sample=df.head(),
        openai_api_key=api_key
    )
    print("...Plan received.")

    # --- LLM Chain Step 2: The Generator ---
    print("\nStep 2: Checking for missing methods and generating new functions...")
    generated_code = generate_missing_functions(
        llm_analysis_plan=analysis_plan,
        openai_api_key=api_key
    )
    print("...Code generation complete.")

    # --- LLM Chain Step 3: The Assembler ---
    print("\nStep 3: Assembling the final, end-to-end, narrated analysis script...")
    final_executable_script = assemble_analysis_script(
        llm_analysis_plan=analysis_plan,
        generated_functions_code=generated_code,
        platform_filepaths=platform_filepaths,
        openai_api_key=api_key
    )
    print("...Script assembled.")

    # --- Execution Step 4: Run the Dynamic Pipeline ---
    print("\nStep 4: Executing the generated pipeline and capturing the narrated report...")
    
    # We need to create a namespace for the script to run in.
    # We will inject the user's DataFrame into this namespace.
    execution_namespace = {
        'df': df,
        'pd': pd,
        'np': np
        # Add other necessary imports if the generated script needs them
    }

    # Use StringIO to capture all print statements from the executed script
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            # The `exec` function runs the string as Python code
            exec(final_executable_script, execution_namespace)
        
        print("...Execution complete.")
        # Get the full narrated output from the StringIO buffer
        narrated_report = f.getvalue()

    except Exception as e:
        narrated_report = f"--- SCRIPT EXECUTION FAILED ---\n\nError:\n{e}\n\n--- Generated Script ---\n{final_executable_script}"

    return narrated_report


if __name__ == '__main__':
    # --- Example: Full End-to-End Orchestration ---
    # First, we need to generate some data for the analysis
    from synthetic_ai_panel import make_multitreat_overlap_panel_5treats
    
    print("--- Generating synthetic panel data for the example ---")
    full_df = make_multitreat_overlap_panel_5treats(n_engineers=500, months=6, seed=123)
    
    # Define the experiment context and data schema
    is_rct_input = False
    data_design_input = "panel"
    column_map_input = {
        'treatments': ['chat_on', 'copilot_on', 'ia_on'],
        'covariates': ['age', 'tenure', 'region', 'y_pre'],
        'target': ['efficiency']
    }
    user_question_input = "Estimate the independent impact of 3 AI tools on engineer efficiency from a staggered rollout."
    platform_files = ['experimentation_platform.py', 'causal_nonrct_pipeline.py']

    print("\n--- Starting the Causal Analysis Orchestrator ---")
    final_report = orchestrate_causal_analysis(
        df=full_df,
        is_rct=is_rct_input,
        data_design=data_design_input,
        platform_filepaths=platform_files,
        has_network_effects=False,
        has_contamination=False,
        user_question=user_question_input,
        column_map=column_map_input
    )

    print("\n" + "="*80)
    print("--- FINAL NARRATED REPORT ---")
    print("="*80)
    print(final_report)
