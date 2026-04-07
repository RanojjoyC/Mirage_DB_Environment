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
    You are an expert database query optimizer.
    Your goal is to minimise total join cost by choosing the join order and strategy.

    COST MODEL:
      step_cost  =  base_rows  x  selectivity  x  join_multiplier
      base_rows  =  rows x 0.5  if use_index=1 AND table has an index,  else rows
      join_type 0 (hash):         multiplier = 1.0
      join_type 1 (nested_loop):  multiplier = 2.0   -- MOST expensive, avoid
      join_type 2 (merge_sort):   multiplier = 0.8   -- CHEAPEST, prefer this

    STRATEGY: join the table with the lowest effective cost first.
    Always prefer merge_sort (join_type=2) and use the index (use_index=1) when available.

    Respond with ONLY a JSON object — no markdown, no explanation:
    {"next_table": <int>, "join_type": <0|1|2>, "use_index": <0|1>}
""").strip()


def build_user_prompt(obs, task_config, step: int) -> str:
    lines = []
    for i in range(len(obs.tables)):
        status   = "JOINED" if i in obs.chosen_order else "pending"
        idx_info = "indexed" if obs.has_index[i] else "no_index"
        eff_cost = obs.table_rows[i] * (0.5 if obs.has_index[i] else 1) * obs.selectivities[i] * 0.8
        lines.append(
            f"  [{i}] {obs.tables[i]:15s} rows={obs.table_rows[i]:>8,} "
            f"sel={obs.selectivities[i]:.3f} {idx_info:10s} best_cost~{eff_cost:>9.2f} [{status}]"
        )

    remaining_display = [f"[{i}]{obs.tables[i]}" for i in obs.remaining_tables]
    joined_display    = [obs.tables[i] for i in obs.chosen_order]

    return textwrap.dedent(f"""
        Step {step} — Task: {task_config.name} ({task_config.difficulty.upper()})

        Query being optimised:
        {obs.query_context}

        Tables:
        {chr(10).join(lines)}

        Already joined : {joined_display if joined_display else "(none)"}
        Remaining      : {remaining_display}
        Accumulated cost so far: {obs.current_cost:.4f}

        Note: table_rows are ESTIMATED (planner statistics with noise).
        The actual execution cost is computed on true cardinalities.
        Prefer small, highly-selective tables first.

        Choose next_table from: {obs.remaining_tables}
    """).strip()


# ─────────────────────────────────────────────────────────────────────────────
# LLM call + greedy fallback
# ─────────────────────────────────────────────────────────────────────────────

def greedy_fallback(obs) -> dict:
    """Deterministic fallback: pick lowest-effective-cost remaining table."""
    best = min(
        obs.remaining_tables,
        key=lambda i: obs.table_rows[i] * (0.5 if obs.has_index[i] else 1) * obs.selectivities[i] * 0.8,
    )
    return {"next_table": best, "join_type": 2, "use_index": 1}


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

    # All retries exhausted → greedy deterministic fallback
    return greedy_fallback(obs), f"llm_failed:{last_error}"


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

        # Final grader score — normalised in [0.0, 1.0]
        score   = grade(env._scenario.tables, env.state.final_cost)
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
