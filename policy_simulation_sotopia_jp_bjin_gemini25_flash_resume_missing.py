from __future__ import annotations

import asyncio

import policy_simulation_sotopia_jp_bjin_gemini25_flash as sim
from gemini_missing_resume_utils import run_missing_sotopia


if __name__ == "__main__":
    asyncio.run(
        run_missing_sotopia(
            sim=sim,
            result_prefix="sotopia_simulation_jp_aip_results_run",
            interaction_log_prefix="sotopia_simulation_jp_aip_interaction_log_run",
            default_prediction_key=(
                "main_conference_accepted_articles_neurips_iclr_icml"
            ),
            average_filename=(
                "sotopia_simulation_jp_aip_gemini25_flash_saved_run_average_predictions.json"
            ),
            failure_filename=(
                "sotopia_simulation_jp_aip_gemini25_flash_resume_failures.json"
            ),
        )
    )
