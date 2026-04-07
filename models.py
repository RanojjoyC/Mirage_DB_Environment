from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import List

# ──────────────── ACTION ─────────────────────────────────────────────────────
class QueryAction(Action):
    next_table: int = Field(..., ge=0,   description="Index of next table to join (from remaining_tables)")
    join_type:  int = Field(..., ge=0, le=2, description="Join algorithm: 0=hash(1.0×) 1=nested-loop(2.0×) 2=merge-sort(0.8×)")
    use_index:  int = Field(..., ge=0, le=1, description="Index scan: 0=full-table-scan 1=index-scan(halves base rows if index present)")

# ──────────────── OBSERVATION ────────────────────────────────────────────────
class QueryObservation(Observation):
    # Schema visible to the agent
    tables:           List[str]    # table names in this query scenario
    table_rows:       List[int]    # cardinality estimates (noisy — simulates planner statistics)
    selectivities:    List[float]  # join predicate selectivities
    has_index:        List[int]    # index availability per table (1=yes, 0=no)

    # Episode progress
    chosen_order:     List[int]    # table indices already joined, in insertion order
    remaining_tables: List[int]    # table indices not yet joined (valid choices for next_table)
    step_number:      int          # 0-based step counter
    current_cost:     float        # accumulated join cost so far (based on true cardinalities)
    intermediate_size: float = Field(
        default=1.0,
        description=(
            "Estimated size of the intermediate result accumulated so far. "
            "Computed as the product of (est_rows × selectivity) for each joined table. "
            "Joining large-output tables early causes this to explode, compounding all "
            "subsequent join costs. Keep this small by joining selective tables first."
        ),
    )

    # Enterprise context
    query_context:    str = Field(default="", description="SQL-like description of the query being optimized")

# ──────────────── STATE ──────────────────────────────────────────────────────
class QueryState(State):
    chosen_order:     List[int]  = Field(default_factory=list)
    remaining_tables: List[int]  = Field(default_factory=list)
    current_cost:     float      = 0.0
    final_cost:       float      = 0.0
    scenario_name:    str        = ""   # which scenario was sampled this episode