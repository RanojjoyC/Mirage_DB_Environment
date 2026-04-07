"""
Mirage_RL_environment.py — Production query join-order optimisation environment.

Core design:
  - Episodes are drawn from a pool of enterprise query scenarios (e-commerce,
    analytics, financial) sampled randomly or by seed for reproducibility.
  - Cardinality estimation noise (log-normal) is applied to table_rows in the
    observation — the agent sees estimated stats, not ground truth, matching
    real planner conditions.
  - Cost is computed on TRUE row counts (hidden from the agent's observation),
    reflecting actual query execution cost rather than planner estimates.
  - Rewards are normalised to [0.0, 1.0] at every step.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from Mirage_RL.models import QueryAction, QueryObservation, QueryState
    from Mirage_RL.server.tasks import (
        TaskConfig, TableSpec, Scenario, TASKS,
        apply_estimation_noise, compute_cost_bounds,
        compute_step_cost_bounds, compute_order_quality, grade,
    )
except ImportError:
    from models import QueryAction, QueryObservation, QueryState          # type: ignore
    from server.tasks import (                                             # type: ignore
        TaskConfig, TableSpec, Scenario, TASKS,
        apply_estimation_noise, compute_cost_bounds,
        compute_step_cost_bounds, compute_order_quality, grade,
    )

# Join-type cost multipliers (mirrors the agent's cost model exactly)
_JOIN_MULTIPLIER = {0: 1.0, 1: 2.0, 2: 0.8}


class QueryEnv(Environment):
    """
    Multi-task, multi-scenario database query join-order optimisation environment.

    Episode lifecycle:
      1. reset(task_id, seed) — sample a scenario, apply cardinality noise
      2. step(action)         — choose next table + join strategy; get [0,1] reward
      3. repeat until all tables joined (done=True)

    Reward design:
      Per-step : normalised quality of this join decision → [0.0, 1.0]
      Final    : overall grader score from grade() → [0.0, 1.0]
      Invalid  : action on already-joined table → 0.0 reward (no progress, no crash)
    """

    def __init__(self) -> None:
        self._task: TaskConfig      = TASKS["medium"]
        self._scenario: Scenario    = TASKS["medium"].scenarios[0]

        # True rows (used for cost), estimated rows (shown to agent)
        self._true_rows:  List[int] = []
        self._est_rows:   List[int] = []
        self._cost_bounds: tuple[float, float] = (1.0, 0.0)  # (worst, best)

        # Intermediate result size tracking:
        #   true  — used internally for cost accuracy
        #   est   — shown to agent (product of est_rows × sel for joined tables)
        self._true_running_size: float = 1.0
        self._est_running_size:  float = 1.0

        self._state  = QueryState()
        self._step_count: int  = 0
        self._done:        bool = False
        self._rng = random.Random()

    # ──────────────────────────────────────────────────────────────────────────
    # OpenEnv API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "medium", seed: Optional[int] = None, **kwargs) -> QueryObservation:
        """
        Reset to a new episode.

        Args:
            task_id: "easy" | "medium" | "hard"
            seed:    Optional integer seed for reproducible episode sampling.
                     The same seed + task_id always produces the same scenario
                     and cardinality noise, enabling reproducible baselines.
        """
        self._task = TASKS.get(task_id, TASKS["medium"])

        # Seed the RNG: deterministic when seed given, random otherwise
        effective_seed = seed if seed is not None else random.randint(0, 2**31)
        self._rng.seed(effective_seed)

        # Sample a scenario from the pool
        self._scenario = self._rng.choice(self._task.scenarios)

        # Apply cardinality estimation noise to get the agent-visible estimates
        self._true_rows = [t.true_rows for t in self._scenario.tables]
        self._est_rows  = [
            apply_estimation_noise(t.true_rows, t.noise_sigma, self._rng)
            for t in self._scenario.tables
        ]

        # Precompute cost bounds using TRUE rows (for normalisation)
        self._cost_bounds = compute_cost_bounds(self._scenario.tables)

        # Reset episode state
        n = len(self._scenario.tables)
        self._state.chosen_order     = []
        self._state.remaining_tables = list(range(n))
        self._state.current_cost     = 0.0
        self._state.final_cost       = 0.0
        self._state.scenario_name    = self._scenario.name
        self._step_count = 0
        self._done       = False
        self._true_running_size = 1.0
        self._est_running_size  = 1.0

        return self._make_obs(reward=0.0)

    def step(self, action: QueryAction) -> QueryObservation:
        """
        Execute one join-order decision.

        Returns:
            QueryObservation with reward in [0.0, 1.0].
        """
        # ── Guard: invalid table selection ────────────────────────────────────
        if action.next_table not in self._state.remaining_tables:
            # Zero reward; episode does NOT terminate (agent must recover)
            return self._make_obs(reward=0.0)

        # ── Apply action ──────────────────────────────────────────────────────
        idx = action.next_table
        self._state.chosen_order.append(idx)
        self._state.remaining_tables.remove(idx)
        self._step_count += 1

        # Cost computed on TRUE rows (not the noisy estimate the agent sees)
        added_cost  = self._true_step_cost(idx, action)
        self._state.current_cost += added_cost

        # ── Normalised per-step reward ────────────────────────────────────────
        # MUST evaluate bounds using the running_size BEFORE it gets multiplied
        table_spec                = self._scenario.tables[idx]
        worst_step, best_step     = compute_step_cost_bounds(table_spec, self._true_running_size)
        if worst_step == best_step:
            step_reward = 1.0
        else:
            step_reward = (worst_step - added_cost) / (worst_step - best_step)
            step_reward = float(max(0.0, min(1.0, step_reward)))

        # Update intermediate size trackers for the NEXT step
        sel = self._scenario.tables[idx].selectivity
        self._true_running_size *= self._true_rows[idx] * sel
        self._est_running_size  *= self._est_rows[idx]  * sel

        # ── Episode complete? ─────────────────────────────────────────────────
        if not self._state.remaining_tables:
            self._done               = True
            self._state.final_cost   = self._state.current_cost
            final_score = grade(
                self._scenario.tables,
                self._state.final_cost,
                chosen_order=list(self._state.chosen_order),
            )
            return self._make_obs(reward=final_score)

        return self._make_obs(reward=step_reward)

    @property
    def state(self) -> QueryState:
        return self._state

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _true_step_cost(self, table_idx: int, action: QueryAction) -> float:
        """
        Compute join cost using TRUE row count (not the noisy estimate), 
        adding an intermediate size penalty for cascading costs.
        This strongly encourages joining highly selective, small tables early.
        """
        true_rows = self._true_rows[table_idx]
        sel       = self._scenario.tables[table_idx].selectivity
        has_idx   = self._scenario.tables[table_idx].has_index

        base_rows = true_rows * 0.5 if (action.use_index and has_idx) else true_rows
        mult      = _JOIN_MULTIPLIER.get(action.join_type, 1.0)
        
        # Additive penalty: the size of the accumulated running result
        penalty   = max(0.0, self._true_running_size - 1.0)
        return (base_rows * sel * mult) + penalty

    def _make_obs(self, reward: float) -> QueryObservation:
        """Build observation — agent sees ESTIMATED rows and intermediate size."""
        return QueryObservation(
            done              = self._done,
            reward            = round(reward, 6),
            tables            = list(t.name for t in self._scenario.tables),
            table_rows        = list(self._est_rows),
            selectivities     = [t.selectivity   for t in self._scenario.tables],
            has_index         = [t.has_index      for t in self._scenario.tables],
            chosen_order      = list(self._state.chosen_order),
            remaining_tables  = list(self._state.remaining_tables),
            step_number       = self._step_count,
            current_cost      = round(self._state.current_cost, 6),
            query_context     = self._scenario.query_context,
            intermediate_size = round(self._est_running_size, 2),
        )