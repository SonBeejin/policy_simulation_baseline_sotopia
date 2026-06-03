from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any


def _run_index_from_file(path: Path) -> int | None:
    match = re.search(r"run_(\d{3})\.json$", path.name)
    return int(match.group(1)) if match else None


def _prediction_key_from_results(
    results: list[dict[str, Any]], default_prediction_key: str
) -> str:
    for result in results:
        values = result.get("prediction_values") or {}
        if values:
            return next(iter(values.keys()))
    return default_prediction_key


def _load_run_result(
    sim: Any,
    run_index: int,
    result_prefix: str,
    interaction_log_prefix: str,
    default_prediction_key: str,
) -> dict[str, Any] | None:
    output_path = sim.RUN_OUTPUT_DIR / f"{result_prefix}_{run_index:03d}.json"
    if not output_path.exists():
        return None

    try:
        with open(output_path, encoding="utf-8") as file:
            results = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(results, list):
        return None

    prediction_key = _prediction_key_from_results(results, default_prediction_key)
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
        "interaction_log_path": str(
            sim.RUN_OUTPUT_DIR / f"{interaction_log_prefix}_{run_index:03d}.json"
        ),
        "num_agents": len(results),
        "selected_personas": selected_personas,
        "prediction_key": prediction_key,
        "predictions": predictions,
        "avg_prediction": sum(predictions) / len(predictions) if predictions else None,
        "results": results,
    }


def _load_all_completed_runs(
    sim: Any,
    result_prefix: str,
    interaction_log_prefix: str,
    default_prediction_key: str,
) -> list[dict[str, Any]]:
    all_runs: list[dict[str, Any]] = []
    for path in sorted(sim.RUN_OUTPUT_DIR.glob(f"{result_prefix}_*.json")):
        run_index = _run_index_from_file(path)
        if run_index is None:
            continue
        run_result = _load_run_result(
            sim,
            run_index,
            result_prefix,
            interaction_log_prefix,
            default_prediction_key,
        )
        if run_result is not None:
            all_runs.append(run_result)
    return sorted(all_runs, key=lambda item: item["run_index"])


def _build_average_summary(
    all_runs: list[dict[str, Any]], default_prediction_key: str
) -> dict[str, Any]:
    valid_runs = [
        run_result
        for run_result in all_runs
        if run_result.get("avg_prediction") is not None
    ]
    run_averages = [float(run_result["avg_prediction"]) for run_result in valid_runs]
    all_predictions = [
        float(prediction)
        for run_result in valid_runs
        for prediction in run_result.get("predictions", [])
    ]
    prediction_key = (
        valid_runs[0].get("prediction_key", default_prediction_key)
        if valid_runs
        else default_prediction_key
    )

    return {
        "prediction_key": prediction_key,
        "saved_run_file_count": len(all_runs),
        "valid_run_count": len(valid_runs),
        "total_agent_prediction_count": len(all_predictions),
        "average_of_run_averages": (
            sum(run_averages) / len(run_averages) if run_averages else None
        ),
        "average_of_all_agent_predictions": (
            sum(all_predictions) / len(all_predictions) if all_predictions else None
        ),
        "min_run_average": min(run_averages) if run_averages else None,
        "max_run_average": max(run_averages) if run_averages else None,
        "run_averages": [
            {
                "run_index": run_result["run_index"],
                "avg_prediction": run_result.get("avg_prediction"),
                "num_predictions": len(run_result.get("predictions", [])),
            }
            for run_result in valid_runs
        ],
    }


async def run_missing_sotopia(
    *,
    sim: Any,
    result_prefix: str,
    interaction_log_prefix: str,
    default_prediction_key: str,
    average_filename: str,
    failure_filename: str,
) -> None:
    start_run = int(os.getenv("MISSING_START_RUN", "1"))
    end_run = int(os.getenv("MISSING_END_RUN", str(sim.NUM_RUNS)))
    if end_run < start_run:
        raise ValueError(
            "MISSING_END_RUN must be >= MISSING_START_RUN: "
            f"{end_run} < {start_run}"
        )

    sim.NUM_AGENTS = int(os.getenv("NUM_AGENTS", str(sim.NUM_AGENTS)))
    sim.RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing_run_indices = [
        run_index
        for run_index in range(start_run, end_run + 1)
        if _load_run_result(
            sim,
            run_index,
            result_prefix,
            interaction_log_prefix,
            default_prediction_key,
        )
        is None
    ]

    print(
        f"[resume] missing runs in {start_run}-{end_run}: "
        f"{len(missing_run_indices)}"
    )
    if missing_run_indices:
        print(
            "[resume] run indices: "
            + ", ".join(str(run_index) for run_index in missing_run_indices)
        )

    policy_text = sim.load_policy_summary(sim.POLICY_PATH)
    policy_prompt = sim.build_policy_prompt(policy_text)
    semaphore = asyncio.Semaphore(sim.PARALLEL_RUNS)
    failures: list[dict[str, Any]] = []

    async def run_with_limit(run_index: int) -> None:
        async with semaphore:
            try:
                await sim.run_once(run_index, sim.NUM_RUNS, policy_prompt)
            except Exception as exc:
                failure = {
                    "run_index": run_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                print(
                    f"[simulation {run_index}/{sim.NUM_RUNS}] failed: "
                    f"{failure['error_type']}: {failure['error']}",
                    flush=True,
                )

    await asyncio.gather(
        *(asyncio.create_task(run_with_limit(run_index)) for run_index in missing_run_indices)
    )

    all_runs = _load_all_completed_runs(
        sim,
        result_prefix,
        interaction_log_prefix,
        default_prediction_key,
    )
    interaction_log_refs = [
        {
            "run_index": run_result["run_index"],
            "run_seed": run_result["run_seed"],
            "interaction_log_path": run_result["interaction_log_path"],
        }
        for run_result in all_runs
    ]
    average_summary = _build_average_summary(all_runs, default_prediction_key)

    average_path = sim.RUN_OUTPUT_DIR / average_filename
    failure_path = sim.RUN_OUTPUT_DIR / failure_filename
    with open(sim.OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(all_runs, file, ensure_ascii=False, indent=2)
    with open(sim.INTERACTION_LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(interaction_log_refs, file, ensure_ascii=False, indent=2)
    with open(average_path, "w", encoding="utf-8") as file:
        json.dump(average_summary, file, ensure_ascii=False, indent=2)
    with open(failure_path, "w", encoding="utf-8") as file:
        json.dump(failures, file, ensure_ascii=False, indent=2)

    remaining_missing = [
        run_index
        for run_index in range(start_run, end_run + 1)
        if _load_run_result(
            sim,
            run_index,
            result_prefix,
            interaction_log_prefix,
            default_prediction_key,
        )
        is None
    ]

    print("\n" + "=" * 70)
    print(f"Missing-fill range: {start_run}-{end_run}")
    print(f"Completed run files found: {len(all_runs)}/{sim.NUM_RUNS}")
    print(f"Remaining missing in range: {len(remaining_missing)}")
    if remaining_missing:
        print(
            "Remaining run indices: "
            + ", ".join(str(run_index) for run_index in remaining_missing)
        )
    print(f"Failure log saved to: {failure_path}")
    print(f"Results saved to: {sim.OUTPUT_PATH}")
    print(f"Interaction log refs saved to: {sim.INTERACTION_LOG_PATH}")
    print(f"Average summary saved to: {average_path}")
    print(f"Average of run averages: {average_summary['average_of_run_averages']}")
    print("=" * 70)
