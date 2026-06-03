from __future__ import annotations

import asyncio

import policy_simulation_sotopia_intl_bjin_gemini25_flash as sim
from gemini_missing_resume_utils import run_missing_sotopia


if __name__ == "__main__":
    asyncio.run(
        run_missing_sotopia(
            sim=sim,
            result_prefix="sotopia_simulation_intl_results_run",
            interaction_log_prefix="sotopia_simulation_intl_interaction_log_run",
            default_prediction_key="international_co_research_ratio",
            average_filename=(
                "sotopia_simulation_intl_gemini25_flash_saved_run_average_predictions.json"
            ),
            failure_filename="sotopia_simulation_intl_gemini25_flash_resume_failures.json",
        )
    )
