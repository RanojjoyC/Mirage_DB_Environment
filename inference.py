"""
inference.py — Baseline inference script for the Mirage_RL OpenEnv environment.
Place this file in the root directory of the project (Mirage_RL/).

MANDATORY ENVIRONMENT VARIABLES:
    HF_TOKEN         Your Hugging Face / API key.
    API_BASE_URL     The API endpoint for the LLM.
                     Default: https://router.huggingface.co/v1
    MODEL_NAME       The model identifier to use for inference.
                     Default: Qwen/Qwen2.5-72B-Instruct

STDOUT FORMAT (exact — one line per tag, no newlines within a line):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    HF_TOKEN=<key> API_BASE_URL=<url> MODEL_NAME=<model> python inference.py
"""

from __future__ import annotations

import os
import sys
import time
import textwrap
from typing import List, Optional

# ── Path setup: works when run locally or inside Docker ───────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── OpenAI client ─────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    sys.exit("ERROR: 'openai' package not found. Run: pip install openai>=1.0.0")

# ── Environment + task imports ────────────────────────────────────────────────
try:
    from Mirage_RL.server.Mirage_RL_environment import QueryEnv
    from Mirage_RL.server.tasks import TASKS, grade
    from Mirage_RL.models import QueryAction
except ImportError:
    try:
        from server.Mirage_RL_environment import QueryEnv          # type: ignore
        from server.tasks import TASKS, grade                      # type: ignore
        from models import QueryAction                             # type: ignore
    except ImportError as exc:
        sys.exit(
            f"ERROR: Cannot import Mirage_RL modules.\n"
            f"Run inference.py from the Mirage_RL/ directory or install the package.\n"
            f"Details: {exc}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK    = "mirage_rl"

TASK_ORDER              = ["easy", "medium", "hard"]
MAX_STEPS               = 10      # hard upper bound per episode (tasks finish naturally)
MAX_RETRIES             = 3
RETRY_DELAY             = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5     # score >= 0.5 counts as success


# ─────────────────────────────────────────────────────────────────────────────
# Mandatory log helpers  (exact format — do not modify field names or order)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Action string formatter  (used in [STEP] log)
# ─────────────────────────────────────────────────────────────────────────────

_JOIN_NAMES  = {0: "hash", 1: "nested_loop", 2: "merge_sort"}
_INDEX_NAMES = {0: "no_index", 1: "use_index"}

def format_action(action: QueryAction, tables: list) -> str:
    table_name = tables[action.next_table] if action.next_table < len(tables) else str(action.next_table)
    join_name  = _JOIN_NAMES.get(action.join_type, str(action.join_type))
    idx_name   = _INDEX_NAMES.get(action.use_index, str(action.use_index))
    return f"join({table_name},{join_name},{idx_name})"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a database query optimizer. Your job is to build an efficient join plan
    by deciding, one table at a time, which table to join next and how.

    You will see table statistics including estimated row counts, join selectivities,
    and index availability for each table. These estimates may not be exact — real
    database planners always operate on statistics that can differ from ground truth.

    TWO things determine your score:

    1. JOIN METHOD quality (60% of score): which algorithm and index you choose.
       Different join algorithms have very different cost profiles.
       Using an index when one is available significantly reduces scan cost.

    2. JOIN ORDER quality (40% of score): which table you pick at each step.
       Joining a table with a large output early causes the intermediate result
       to explode — every subsequent join must then probe against that larger set.
       To minimise total cost: prefer joining tables with the smallest
       (estimated_rows × selectivity) output first, saving large tables for later.

    The observation shows 'intermediate_size': the estimated size of the accumulated
    intermediate result from all joins so far. Keep this number small by joining
    highly selective, small-output tables early in the sequence.

    Respond with ONLY a valid JSON object — no markdown, no explanation:
    {"next_table": <int>, "join_type": <0|1|2>, "use_index": <0|1>}
""").strip()


def build_user_prompt(obs, task_config, step: int) -> str:
    lines = []
    for i in range(len(obs.tables)):
        status   = "JOINED" if i in obs.chosen_order else "available"
        idx_info = "indexed" if obs.has_index[i] else "no_index"
        est_out  = obs.table_rows[i] * obs.selectivities[i]   # estimated output rows
        lines.append(
            f"  [{i}] {obs.tables[i]:15s}  est_rows={obs.table_rows[i]:>12,}  "
            f"sel={obs.selectivities[i]:.3f}  est_output={est_out:>12,.0f}  "
            f"{idx_info:10s}  [{status}]"
        )

    remaining_display = [f"[{i}]{obs.tables[i]}" for i in obs.remaining_tables]
    joined_display    = [obs.tables[i] for i in obs.chosen_order]

    return textwrap.dedent(f"""
        Step {step} of episode  |  Task: {task_config.name}  [{task_config.difficulty.upper()}]

        Query:
        {obs.query_context}

        Table statistics (est_rows = planner estimates, may differ from true cardinality):
        {chr(10).join(lines)}

        Join progress:
          Already joined       : {joined_display if joined_display else "(none — first step)"}
          Remaining            : {remaining_display}
          Accumulated cost     : {obs.current_cost:.2f}
          Intermediate size    : {obs.intermediate_size:,.0f}  ← keep this small; explodes if you join large-output tables early

        Decide the next join. Choose next_table from {obs.remaining_tables}.
        Reason about join ORDER (intermediate blowup) AND join METHOD (algorithm + index).
        Output the JSON.
    """).strip()



# ─────────────────────────────────────────────────────────────────────────────
# LLM call + greedy fallback
# ─────────────────────────────────────────────────────────────────────────────

import random as _random

def random_fallback(obs) -> dict:
    """
    Random fallback when LLM fails — picks a uniformly random remaining table,
    random join type, random index usage. This ensures LLM failures are penalised
    rather than silently rescued by an optimal greedy choice.
    """
    table  = _random.choice(obs.remaining_tables)
    j_type = _random.randint(0, 2)
    use_ix = _random.randint(0, 1)
    return {"next_table": table, "join_type": j_type, "use_index": use_ix}


def get_action(client: OpenAI, obs, task_config, step: int) -> tuple[dict, Optional[str]]:
    """
    Ask LLM for next action. Returns (action_dict, error_or_None).
    Falls back to greedy after MAX_RETRIES failures.
    """
    user_prompt = build_user_prompt(obs, task_config, step)
    last_error: Optional[str] = None

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=128,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

            import json
            action = json.loads(text)

            # Validate
            if action.get("next_table") not in obs.remaining_tables:
                raise ValueError(f"next_table={action.get('next_table')} not in {obs.remaining_tables}")
            action["join_type"] = int(action.get("join_type", 2))
            action["use_index"] = int(action.get("use_index", 1))
            if action["join_type"] not in (0, 1, 2):
                action["join_type"] = 2
            if action["use_index"] not in (0, 1):
                action["use_index"] = 1

            return action, None

        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    # All retries exhausted → random fallback (penalises LLM failures)
    return random_fallback(obs), f"llm_failed:{last_error}"


# ─────────────────────────────────────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str, client: OpenAI) -> float:
    """Run one task episode. Returns final score in [0.0, 1.0]."""
    task_config = TASKS[task_id]
    env = QueryEnv()

    rewards:     List[float]  = []
    steps_taken: int          = 0
    score:       float        = 0.0
    success:     bool         = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=task_id, seed=42)  # seed=42 → reproducible scenario + noise

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_dict, error = get_action(client, obs, task_config, step)

            action = QueryAction(
                next_table=int(action_dict["next_table"]),
                join_type=int(action_dict["join_type"]),
                use_index=int(action_dict["use_index"]),
            )

            obs = env.step(action)

            reward       = obs.reward
            done         = obs.done
            steps_taken  = step

            rewards.append(reward)

            log_step(
                step=step,
                action=format_action(action, list(obs.tables)),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # Final grader score — the environment returns the episode score natively on done
        score   = rewards[-1] if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} exception: {exc}", flush=True)
        score   = 0.0
        success = False

    finally:
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print(
            "ERROR: No API key found.\n"
            "Set HF_TOKEN (or API_KEY / OPENAI_API_KEY) before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    results: list[dict] = []
    for task_id in TASK_ORDER:
        score = run_task(task_id, client)
        results.append({"task_id": task_id, "score": score})

    # Human-readable summary (goes to stdout after all [END] lines)
    avg = sum(r["score"] for r in results) / len(results)
    sep = "=" * 52
    print(f"\n{sep}", flush=True)
    print("  MIRAGE_RL BASELINE RESULTS", flush=True)
    print(sep, flush=True)
    for r in results:
        bar = "█" * int(r["score"] * 20)
        print(f"  {r['task_id']:8s} | {r['score']:.2f} | {bar}", flush=True)
    print(sep, flush=True)
    print(f"  {'AVERAGE':8s} | {avg:.2f}", flush=True)
    print(sep, flush=True)


if __name__ == "__main__":
    main()
