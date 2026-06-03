r"""
Sotopia 스타일 다중 에이전트 상호작용을 활용한 BK21 정책 시뮬레이션.

프로젝트 루트에서 실행:
    python sotopia/policy_simulation_sotopia_fr_bjin_gpt4o_mini.py
=========================================================================
cd "C:\Users\BEEJIN\Desktop\비진이\HCCL\정책연구"
$env:PYTHONUTF8="1"
$env:PARALLEL_RUNS="8"
.\.venv\Scripts\python.exe sotopia/policy_simulation_sotopia_fr_bjin_gpt4o_mini.py
=========================================================================
출력 파일:
    sotopia_simulation_results.json
    sotopia_simulation_interaction_log.json
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

# 페르소나 출처:
# - "file": 기존 BK21 페르소나 JSON 사용 방식 유지
# - "sotopia_db": Sotopia AgentProfile 저장소에서 페르소나 읽기
PERSONA_SOURCE = "file"
os.environ["SOTOPIA_STORAGE_BACKEND"] = "local"

# sotopia를 가져오기 전에 redis_om이 인메모리 Redis 대체 구현을 쓰도록 패치한다.
# 기존 독립 실행 모드를 유지하기 위한 처리이며, 실제 Sotopia Redis/로컬 저장소
# 백엔드에서 AgentProfile 행을 읽을 때는 비활성화해야 한다.
# if PERSONA_SOURCE != "sotopia_db":
#     import fakeredis
#     import redis_om.checks as _redis_checks
#     import redis_om.model.model as _redis_model

#     _fake_server = fakeredis.FakeServer()
#     _fake_redis = fakeredis.FakeRedis(server=_fake_server)
#     _redis_model.get_redis_connection = lambda **kwargs: _fake_redis
#     _redis_checks.check_for_command = lambda conn, cmd: True

import requests

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
    io.StringIO()
):
    from sotopia.agents.base_agent import BaseAgent
    from sotopia.agents.llm_agent import Agents
    from sotopia.database import AgentProfile, EnvironmentProfile
    from sotopia.envs import ParallelSotopiaEnv
    from sotopia.messages import AgentAction, Observation

SCRIPT_DIR = Path(__file__).resolve().parent       # 프로젝트_루트/sotopia
BASE_DIR = SCRIPT_DIR.parent                       # 프로젝트_루트

PERSONAS_PATH = BASE_DIR / "simulation_personas_fr.json"
POLICY_PATH = BASE_DIR / "input" / "fr_3IA_2021.txt"
RESULT_DIR = BASE_DIR / "sotopia_result"
RUN_OUTPUT_DIR = RESULT_DIR / "policy_simulation_sotopia_fr_bjin_gpt4o_mini_runs"
OUTPUT_PATH = RUN_OUTPUT_DIR / "sotopia_simulation_fr_results_all_runs.json"
INTERACTION_LOG_PATH = (
    RUN_OUTPUT_DIR / "sotopia_simulation_fr_interaction_log_all_runs.json"
)

LUXIA_API_URL = os.getenv(
    "LUXIA_GPT4O_MINI_API_URL",
    "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o-mini/create",
)
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
MODEL_NAME = os.getenv("LUXIA_GPT4O_MINI_MODEL", "gpt-4o-mini-2024-07-18")
LUXIA_MAX_RETRIES = int(os.getenv("LUXIA_MAX_RETRIES", "6"))
LUXIA_RETRY_BASE_SECONDS = float(os.getenv("LUXIA_RETRY_BASE_SECONDS", "5"))
LUXIA_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

RANDOM_SEED_ENV = os.getenv("RANDOM_SEED")
RANDOM_SEED: int | None = (
    int(RANDOM_SEED_ENV) if RANDOM_SEED_ENV is not None else None
)
NUM_AGENTS = 5
NUM_RUNS = 200
PARALLEL_RUNS = int(os.getenv("PARALLEL_RUNS", "5"))
DISCUSSION_ROUNDS = 2


_total_prompt_tokens: int = 0
_total_completion_tokens: int = 0
_total_calls: int = 0
_token_lock = threading.Lock()


def deepseek_call(
    system_content: str, user_content: str
) -> tuple[str, dict[str, int]]:
    global _total_prompt_tokens, _total_completion_tokens, _total_calls

    if not LUXIA_API_KEY:
        raise RuntimeError("Set LUXIA_API_KEY in this file or environment before running.")

    headers = {"apikey": LUXIA_API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "stream": True,
        "temperature": 0.7,
    }
    chunks: list[str] = []
    usage: dict[str, int] = {}
    last_error: Exception | None = None
    for attempt in range(LUXIA_MAX_RETRIES + 1):
        chunks = []
        usage = {}
        response: requests.Response | None = None
        try:
            response = requests.post(
                LUXIA_API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                data_str = decoded[6:] if decoded.startswith("data: ") else decoded
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if not content:
                    content = chunk.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    chunks.append(content)
            break
        except requests.HTTPError as exc:
            last_error = exc
            status_code = response.status_code if response is not None else None
            if status_code not in LUXIA_RETRY_STATUS_CODES or attempt >= LUXIA_MAX_RETRIES:
                body = response.text[:1000] if response is not None else ""
                raise requests.HTTPError(f"{exc}; response body: {body}", response=response) from exc
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= LUXIA_MAX_RETRIES:
                raise
        sleep_seconds = min(LUXIA_RETRY_BASE_SECONDS * (2 ** attempt), 120)
        print(
            "[luxia retry] "
            f"attempt {attempt + 1}/{LUXIA_MAX_RETRIES} failed: "
            f"{type(last_error).__name__}: {last_error}. "
            f"retrying in {sleep_seconds:.1f}s",
            flush=True,
        )
        time.sleep(sleep_seconds)

    with _token_lock:
        _total_calls += 1
        _total_prompt_tokens += usage.get("prompt_tokens", 0)
        _total_completion_tokens += usage.get("completion_tokens", 0)
    return "".join(chunks), usage


def load_personas(path: Path, n: int | None, seed: int | None) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    all_personas: list[dict[str, Any]] = []
    for scenario in data.get("scenarios", []):
        all_personas.extend(scenario.get("participants", []))

    rng = random.Random(seed)
    return rng.sample(all_personas, min(n or len(all_personas), len(all_personas)))


def load_sotopia_db_personas(n: int, seed: int | None) -> list[dict[str, Any]]:
    profiles = list(AgentProfile.all())
    if not profiles:
        raise RuntimeError(
            "No AgentProfile rows found in Sotopia storage. Populate the Sotopia "
            "database first, or run with PERSONA_SOURCE=file."
        )

    rng = random.Random(seed)
    sampled_profiles = rng.sample(profiles, min(n, len(profiles)))

    personas: list[dict[str, Any]] = []
    for idx, profile in enumerate(sampled_profiles):
        first_name = str(getattr(profile, "first_name", "") or f"Agent{idx}")
        last_name = str(getattr(profile, "last_name", "") or "")
        full_name = " ".join(part for part in [first_name, last_name] if part)
        persona_id = str(getattr(profile, "pk", "") or full_name or f"agent_{idx}")
        public_info = str(getattr(profile, "public_info", "") or "")
        personality = str(getattr(profile, "personality_and_values", "") or "")
        decision_style = str(getattr(profile, "decision_making_style", "") or "")
        secret = str(getattr(profile, "secret", "") or "")
        persona_parts = [
            part
            for part in [public_info, personality, decision_style, secret]
            if part
        ]

        personas.append(
            {
                "persona_id": persona_id,
                "first_name": first_name,
                "last_name": last_name,
                "occupation": str(getattr(profile, "occupation", "") or ""),
                "age": getattr(profile, "age", 30) or 30,
                "district": str(getattr(profile, "tag", "") or ""),
                "gender": str(getattr(profile, "gender", "") or "Non-binary"),
                "gender_pronoun": str(
                    getattr(profile, "gender_pronoun", "") or "They/Them"
                ),
                "professional_persona": "\n".join(persona_parts),
            }
        )

    return personas


def load_simulation_personas(n: int, seed: int | None) -> list[dict[str, Any]]:
    return load_personas(PERSONAS_PATH, n, seed)


def load_policy_summary(path: Path) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read()


def build_policy_prompt(policy_text: str) -> str:
    return (
        "We are simulating a stakeholder discussion about the input 3IA "
        "Cote d'Azur policy document. Based on the policy document below "
        "and your professional persona, discuss the policy outcome indicator "
        "with the other agents.\n\n"
        f"{policy_text}\n\n"
        "Target indicator: number_of_publications_3ia_cote_dazur_2023. "
        "This is the Number of Publications attributed to 3IA Cote d'Azur "
        "in calendar year 2023.\n\n"
        "Guidance:\n"
        "- If the policy document gives an explicit publication value, target, "
        "or comparable baseline, use it as the main anchor.\n"
        "- If no explicit 2023 value is provided, estimate a reasonable annual "
        "publication count from the institute's research scale, teams, funding, "
        "partnerships, historical outputs, and your persona's expertise.\n"
        "- Keep the discussion focused on 3IA Cote d'Azur publications, not "
        "general AI outputs across France or unrelated institutions.\n\n"
        "Do not submit the final JSON prediction during the discussion rounds. "
        "Use the discussion to surface evidence, disagreements, assumptions, "
        "and revisions. A separate final prediction request will come after "
        "the interaction is complete."
    )
def parse_json_response(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def compact(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


class PolicyDiscussionAgent(BaseAgent[Observation, AgentAction]):
    def __init__(self, persona: dict[str, Any], policy_prompt: str) -> None:
        self.persona = persona
        self.policy_prompt = policy_prompt
        self.turns_spoken = 0
        self.latest_response = ""
        self.final_response = ""
        self.last_usage: dict[str, int] = {}
        self.total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        profile = AgentProfile(
            first_name=str(
                persona.get("first_name") or persona.get("persona_id", "agent")
            )[:20],
            last_name=str(persona.get("last_name", "")),
            age=persona.get("age") if isinstance(persona.get("age"), int) else 30,
            occupation=str(persona.get("occupation", "Policy stakeholder")),
            gender=str(persona.get("gender", "Non-binary")),
            gender_pronoun=str(persona.get("gender_pronoun", "They/Them")),
            public_info=str(persona.get("professional_persona", "")),
            personality_and_values="Evidence-oriented and professionally grounded",
            decision_making_style="Analytical",
        )
        super().__init__(agent_profile=profile)
        self._goal = ("Discuss 3IA Cote d'Azur policy effects with the other stakeholders and provide a prediction for the 2023 Number of Publications.")

    def act(self, obs: Observation) -> AgentAction:
        raise NotImplementedError("Use aact")

    def system_prompt(self) -> str:
        return (
            "You are participating in a Sotopia-style multi-agent policy "
            "discussion. Stay in character as the persona below. Be concise, "
            "specific, and evidence-based.\n\n"
            f"Persona ID: {self.persona.get('persona_id', self.agent_name)}\n"
            f"Occupation: {self.persona.get('occupation', '')}\n"
            f"Age: {self.persona.get('age', '')}\n"
            f"District: {self.persona.get('district', '')}\n"
            f"Professional persona: {self.persona.get('professional_persona', '')}"
        )

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if "speak" not in obs.available_actions:
            return AgentAction(action_type="none", argument="", to=[])

        if self.turns_spoken == 0:
            user_prompt = self.policy_prompt
        else:
            user_prompt = (
                "Continue the stakeholder discussion. Respond to the latest "
                "turns from the other agents, compare your view with theirs, "
                "and keep the conversation focused on the 2023 Number of Publications "
                "for 3IA Cote d'Azur.\n\n"
                f"Latest observation:\n{obs.to_natural_language()}"
            )

        response, usage = await asyncio.to_thread(
            deepseek_call, self.system_prompt(), user_prompt
        )
        self.turns_spoken += 1
        self.latest_response = response
        self.last_usage = usage
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.total_usage[k] += usage.get(k, 0)

        return AgentAction(action_type="speak", argument=response, to=[])

    async def final_prediction(self, interaction_log: list[dict[str, Any]]) -> str:
        discussion = "\n".join(
            f"Turn {item['turn']} | {item['agent']}: {item['argument']}"
            for item in interaction_log
        )
        user_prompt = (
            "The Sotopia discussion has ended. Now provide your final prediction "
            "after considering all interactions below.\n\n"
            f"{self.policy_prompt}\n\n"
            "Interaction log:\n"
            f"{discussion}\n\n"
            "Return valid JSON only in this format:\n"
            '{"prediction_values": {"number_of_publications_3ia_cote_dazur_2023": <number>}, '
            '"narrative": "<2-4 sentence rationale after the discussion>", '
            '"evidence": ["<evidence 1>", "<evidence 2>", "<evidence 3>"]}'
        )
        self.final_response, usage = await asyncio.to_thread(
            deepseek_call, self.system_prompt(), user_prompt
        )
        self.last_usage = usage
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.total_usage[k] += usage.get(k, 0)
        return self.final_response


def build_env_profile(num_agents: int) -> EnvironmentProfile:
    return EnvironmentProfile(
        codename="fr_3ia_publications_multi_agent_discussion",
        scenario=(
            "Multiple 3IA Cote d'Azur stakeholders discuss the input policy "
            "document and predict the 2023 Number of Publications. Each "
            "participant speaks from their professional background and reacts to others."
        ),
        agent_goals=[
            "Provide an evidence-based estimate of number_of_publications_3ia_cote_dazur_2023 and respond to other stakeholders."
            for _ in range(num_agents)
        ],
        relationship=0,
    )



async def run_sotopia_discussion(
    personas: list[dict[str, Any]], policy_prompt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Sotopia 환경 구성 절차:
    # 1. BK21 페르소나를 PolicyDiscussionAgent로 감싼다.
    # 2. round-robin 방식으로 발화하는 ParallelSotopiaEnv를 만든다.
    # 3. 정해진 토론 라운드 동안 발화를 수집한다. 이 단계에서는 최종 JSON 예측을
    #    제출하지 않고 상호작용 로그만 만든다.
    # 4. 토론이 끝난 뒤 각 에이전트에게 별도의 final_prediction 요청을 보낸다.
    # 5. final_prediction JSON을 파싱해 결과 목록에 반영한다.
    agents_map = {
        str(persona.get("persona_id", f"agent_{idx}")): PolicyDiscussionAgent(
            persona, policy_prompt
        )
        for idx, persona in enumerate(personas)
    }
    agents = agents_map
    # 평가자는 사용하지 않고, 에이전트 간 발화 흐름만 기록한다.
    env = ParallelSotopiaEnv(
        available_action_types={"speak", "none", "leave"},
        action_order="round-robin",
        evaluators=[],
        env_profile=build_env_profile(len(agents_map)),
    )
    # reset은 각 에이전트별 초기 관찰값을 반환한다.
    observations = env.reset(agents=agents, omniscient=True)

    interaction_log: list[dict[str, Any]] = []
    max_turns = len(agents_map) * DISCUSSION_ROUNDS
    # 토론 단계. round-robin 환경이 다음 발화자를 관리하며, 발화할 차례가 아닌
    # 에이전트는 "none" 행동을 반환한다.
    for _ in range(max_turns):
        actions: dict[str, AgentAction] = {}
        for name, agent in agents.items():
            action = await agent.aact(observations[name])
            actions[name] = action
            if action.action_type != "none":
                interaction_log.append(
                    {
                        "turn": env.turn_number + 1,
                        "agent": name,
                        "action_type": action.action_type,
                        "argument": action.argument,
                        "to": action.to,
                        "token_usage": agent.last_usage,
                    }
                )
        # env.step으로 행동을 환경에 반영하고 다음 관찰값을 받는다.
        observations, _rewards, terminated, truncated, _info = env.step(actions)
        if all(terminated.values()) or all(truncated.values()):
            break
    # 최종 예측 단계. 각 에이전트에게 전체 상호작용 로그를 제공하고,
    # interaction_log에는 action_type="final_prediction"으로 추가 기록한다.
    for name, agent in agents.items():
        final_response = await agent.final_prediction(interaction_log)
        interaction_log.append(
            {
                "turn": env.turn_number + 1,
                "agent": name,
                "action_type": "final_prediction",
                "argument": final_response,
                "to": [],
                "token_usage": agent.last_usage,
            }
        )
    # 원시 응답과 파싱된 최종 예측 JSON을 결과 객체로 정리한다.
    results: list[dict[str, Any]] = []
    for idx, (name, agent) in enumerate(agents.items()):
        parsed = parse_json_response(agent.final_response)
        persona = agent.persona
        result = {
            "agent_id": idx,
            "agent_name": name,
            "persona_id": persona.get("persona_id", name),
            "occupation": persona.get("occupation", ""),
            "age": persona.get("age", ""),
            "district": persona.get("district", ""),
            "raw_response": agent.final_response,
            "latest_response": agent.latest_response,
            "parsed_response": parsed,
            "total_token_usage": agent.total_usage,
        }
        if parsed:
            result.update(parsed)
        results.append(result)

    return results, interaction_log

async def run_once(
    run_index: int, total_runs: int, policy_prompt: str
) -> dict[str, Any]:
    run_seed = None if RANDOM_SEED is None else RANDOM_SEED + run_index
    output_path = RUN_OUTPUT_DIR / (
        f"sotopia_simulation_fr_results_run_{run_index:03d}.json"
    )
    interaction_log_path = RUN_OUTPUT_DIR / (
        f"sotopia_simulation_fr_interaction_log_run_{run_index:03d}.json"
    )

    print(f"[simulation {run_index}/{total_runs}] start")
    personas = load_simulation_personas(NUM_AGENTS, run_seed)
    results, interaction_log = await run_sotopia_discussion(personas, policy_prompt)

    predictions: list[float] = []
    for result in results:
        result["run_index"] = run_index
        result["run_seed"] = run_seed
        pred_val = (
            (result.get("prediction_values") or {})
            .get("number_of_publications_3ia_cote_dazur_2023")
        )
        if pred_val is not None:
            predictions.append(float(pred_val))

    avg_prediction = sum(predictions) / len(predictions) if predictions else None

    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(interaction_log_path, "w", encoding="utf-8") as file:
        json.dump(interaction_log, file, ensure_ascii=False, indent=2)
    print(f"[simulation {run_index}/{total_runs}] done")

    return {
        "run_index": run_index,
        "run_seed": run_seed,
        "output_path": str(output_path),
        "interaction_log_path": str(interaction_log_path),
        "num_agents": NUM_AGENTS,
        "selected_personas": [
            {
                "agent_id": idx,
                "occupation": persona.get("occupation", ""),
                "age": persona.get("age", ""),
                "district": persona.get("district", ""),
            }
            for idx, persona in enumerate(personas)
        ],
        "prediction_key": "number_of_publications_3ia_cote_dazur_2023",
        "predictions": predictions,
        "avg_prediction": avg_prediction,
        "results": results,
    }


async def main() -> None:
    policy_text = load_policy_summary(POLICY_PATH)
    policy_prompt = build_policy_prompt(policy_text)
    semaphore = asyncio.Semaphore(PARALLEL_RUNS)

    async def run_with_limit(run_index: int) -> dict[str, Any]:
        async with semaphore:
            return await run_once(run_index, NUM_RUNS, policy_prompt)

    tasks = [
        asyncio.create_task(run_with_limit(run_index))
        for run_index in range(1, NUM_RUNS + 1)
    ]
    all_runs = await asyncio.gather(*tasks)
    interaction_log_refs = [
        {
            "run_index": run_result["run_index"],
            "run_seed": run_result["run_seed"],
            "interaction_log_path": run_result["interaction_log_path"],
        }
        for run_result in all_runs
    ]

    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(all_runs, file, ensure_ascii=False, indent=2)
    with open(INTERACTION_LOG_PATH, "w", encoding="utf-8") as file:
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
    print(f"Completed {NUM_RUNS} Sotopia runs (parallel runs: {PARALLEL_RUNS})")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"Interaction log refs saved to: {INTERACTION_LOG_PATH}")
    print(f"Average of run averages: {overall_average}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())



