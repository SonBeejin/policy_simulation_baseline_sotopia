from __future__ import annotations

import asyncio
import os

import policy_simulation_sotopia_fr_bjin as sim
from gemini_missing_resume_utils import run_missing_sotopia


if __name__ == "__main__":
    os.environ.setdefault("MISSING_START_RUN", "151")
    os.environ.setdefault("MISSING_END_RUN", "200")

    asyncio.run(
        run_missing_sotopia(
            sim=sim,
            result_prefix="sotopia_simulation_fr_results_run",
            interaction_log_prefix="sotopia_simulation_fr_interaction_log_run",
            default_prediction_key="number_of_publications_3ia_cote_dazur_2023",
            average_filename="sotopia_simulation_fr_run_average_predictions.json",
            failure_filename="sotopia_simulation_fr_resume_151_200_failures.json",
        )
    )
