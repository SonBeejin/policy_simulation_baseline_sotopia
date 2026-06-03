from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import policy_simulation_sotopia_ssf_bjin_gpt4o_mini as sim


START_RUN = int(os.getenv("START_RUN", "94"))
END_RUN = int(os.getenv("END_RUN", str(sim.NUM_RUNS)))
SKIP_EXISTING = os.getenv("SKIP_EXISTING", "1") != "0"


def _run_result_path(run_index: int) -> Path:
    return sim.RUN_OUTPUT_DIR / f"sotopia_simulation_ssf_results_run_{run_index:03d}.json"


def _interaction_log_path(run_index: int) -> Path:
    return (
        sim.RUN_OUTPUT_DIR
        / f"sotopia_simulation_ssf_interaction_log_run_{run_index:03d}.json"
    )


def _prediction_key_from_results(results: list[dict[str, Any]]) -> str:
    for result in results:
        values = result.get("prediction_values") or {}
        if values:
            return next(iter(values.keys()))
    return "annual_sci_e_paper_count_2021_2022"


def _load_run_result(run_index: int) -> dict[str, Any] | None:
    output_path = _run_result_path(run_index)
    if not output_path.exists():
        return None

    with open(output_path, encoding="utf-8") as file:
        results = json.load(file)

    prediction_key = _prediction_key_from_results(results)
    predictions: list[float] = []
    selected_personas: list[dict[str, Any]] = []

    for result in results:
        raw = (result.get("prediction_values") or {}).get(prediction_key)
        try:
            predictions.append(float(raw))
        except (TypeError, ValueError):
            pass

        selected_personas.append(
            {
                "agent_id": result.get("agent_id"),
                "occupation": result.get("occupation", ""),
                "age": result.get("age", ""),
                "district": result.get("district", ""),
            }
        )

    run_seed = None
    for result in results:
        if "run_seed" in result:
            run_seed = result.get("run_seed")
            break

    return {
        "run_index": run_index,
        "run_seed": run_seed,
        "output_path": str(output_path),
        "interaction_log_path": str(_interaction_log_path(run_index)),
        "num_agents": len(results),
        "selected_personas": selected_personas,
        "prediction_key": prediction_key,
        "predictions": predictions,
        "avg_prediction": sum(predictions) / len(predictions) if predictions else None,
        "results": results,
    }


def _run_index_from_file(path: Path) -> int | None:
    match = re.search(r"run_(\d{3})\.json$", path.name)
    return int(match.group(1)) if match else None


def _load_all_completed_runs() -> list[dict[str, Any]]:
    all_runs: list[dict[str, Any]] = []
    for path in sorted(sim.RUN_OUTPUT_DIR.glob("sotopia_simulation_ssf_results_run_*.json")):
        run_index = _run_index_from_file(path)
        if run_index is None:
            continue
        run_result = _load_run_result(run_index)
        if run_result is not None:
            all_runs.append(run_result)
    return sorted(all_runs, key=lambda item: item["run_index"])


async def main() -> None:
    policy_text = sim.load_policy_summary(sim.POLICY_PATH)
    policy_prompt = sim.build_policy_prompt(policy_text)
    semaphore = asyncio.Semaphore(sim.PARALLEL_RUNS)

    async def run_with_limit(run_index: int) -> dict[str, Any]:
        existing = _load_run_result(run_index) if SKIP_EXISTING else None
        if existing is not None:
            print(f"[simulation {run_index}/{sim.NUM_RUNS}] skip existing")
            return existing

        async with semaphore:
            return await sim.run_once(run_index, sim.NUM_RUNS, policy_prompt)

    sim.RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        asyncio.create_task(run_with_limit(run_index))
        for run_index in range(START_RUN, END_RUN + 1)
    ]
    await asyncio.gather(*tasks)

    all_runs = _load_all_completed_runs()
    interaction_log_refs = [
        {
            "run_index": run_result["run_index"],
            "run_seed": run_result["run_seed"],
            "interaction_log_path": run_result["interaction_log_path"],
        }
        for run_result in all_runs
    ]

    with open(sim.OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(all_runs, file, ensure_ascii=False, indent=2)
    with open(sim.INTERACTION_LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(interaction_log_refs, file, ensure_ascii=False, indent=2)

    valid_averages = [
        run_result["avg_prediction"]
        for run_result in all_runs
        if run_result["avg_prediction"] is not None
    ]
    overall_average = (
        sum(valid_averages) / len(valid_averages) if valid_averages else None
    )

    print("\n" + "=" * 70)
    print(f"Resumed runs {START_RUN}-{END_RUN}")
    print(f"Completed run files found: {len(all_runs)}/{sim.NUM_RUNS}")
    print(f"Results saved to: {sim.OUTPUT_PATH}")
    print(f"Interaction log refs saved to: {sim.INTERACTION_LOG_PATH}")
    print(f"Average of run averages: {overall_average}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
