---
title: Mirage RL — Production Query Optimizer Environment
emoji: 🗄️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - database
  - query-optimization
---

# Mirage_RL: Production Query Join-Order Optimizer

> **A reinforcement learning environment for the #1 performance bottleneck in production databases — join-order optimisation under cardinality estimation uncertainty.**

---

## Why This Problem Is Real and Hard

Every analytical database — PostgreSQL, MySQL, Snowflake, BigQuery, Spark SQL — must decide **how to join tables** before executing a query. For a query touching 7 tables, there are **5,040 possible join orders**. The difference between the best and worst order can be **100–1,000× in execution time**.

The challenge is that real planners don't know the true intermediate result sizes. They rely on table statistics (histograms, row counts, distinct-value estimates) that are frequently:
- **Stale** — vacuumed days or weeks ago on large tables
- **Wrong under data skew** — top 1% of customers produce 80% of orders
- **Compounding** — errors multiply across joins (the "selectivity estimation problem")

This causes PostgreSQL and MySQL to routinely choose plans that are **10–100× slower** than optimal for complex analytical queries. It is a **known, open, expensive problem** — teams at [Meta](https://research.facebook.com/), [Snowflake](https://www.snowflake.com/), [Databricks](https://databricks.com/), and [Carnegie Mellon](https://vldb.org/pvldb/vol12/p1705-marcus.pdf) actively work on learned query optimizers.

**Mirage_RL** provides a rigorous RL benchmark for this exact problem. An AI agent must learn to plan optimal join orders **under the same uncertainty** a real database planner faces.

---

## Environment Overview

Mirage_RL simulates the **planning phase** of a query optimizer. At each step, the agent decides:

1. **Which table** to join next (from the remaining unjoined tables)
2. **Which join algorithm** to use — hash (×1.0), nested-loop (×2.0, avoid), or merge-sort (×0.8)
3. **Whether to use an index** (halves base row count when a covering index exists)

The agent sees **noisy cardinality estimates** — not ground truth — exactly matching real planner conditions. The actual execution cost is computed on true row counts (hidden from the agent), creating a realistic planning-under-uncertainty problem.

**Episode domains** are sampled from three production enterprise schemas:
| Domain | Tables | Typical scale |
|---|---|---|
| **E-commerce (OLTP)** | orders, customers, products, categories, suppliers, warehouses | 10K – 50M rows |
| **Analytics (OLAP)** | events, sessions, users, campaigns, conversions, ab_tests | 1M – 1B rows |
| **Financial (OLAP)** | transactions, accounts, merchants, fraud_labels, risk_scores | 5M – 100M rows |

---

## Action Space

```python
class QueryAction(Action):
    next_table: int   # Index of next table to join (must be from remaining_tables)
    join_type:  int   # 0=hash(×1.0)  1=nested-loop(×2.0)  2=merge-sort(×0.8)
    use_index:  int   # 0=full scan   1=index scan (halves base rows if index exists)
```

**Cost model:**
```
step_cost  =  base_rows  ×  selectivity  ×  join_multiplier
base_rows  =  estimated_rows × 0.5   if use_index=1 AND index present
           =  estimated_rows          otherwise
```

An optimal agent should **always prefer merge-sort** and **always use an index when available**.

---

## Observation Space

```python
class QueryObservation(Observation):
    tables:           List[str]   # table names in this query
    table_rows:       List[int]   # estimated row counts (noisy — simulates planner statistics)
    selectivities:    List[float] # join predicate selectivity per table (0.0–1.0)
    has_index:        List[int]   # index availability: 1=yes, 0=no
    chosen_order:     List[int]   # indices of tables already joined
    remaining_tables: List[int]   # indices of tables not yet joined
    step_number:      int         # 0-based step counter for this episode
    current_cost:     float       # accumulated join cost so far (true rows)
    query_context:    str         # SQL-style description of the query being optimized
```

> **⚠️ Cardinality estimation noise.** `table_rows` in the observation reflects **estimated** cardinalities, not ground truth — matching real-world conditions where planners operate on stale statistics. Noise sigma varies by task difficulty:
> - Easy: σ ≈ 0.03–0.06 (excellent statistics)
> - Medium: σ ≈ 0.05–0.25 (typical production OLTP/OLAP)
> - Hard: σ ≈ 0.10–0.60 (stale stats, billion-row event tables, data skew)

---

## Task Definitions

### Task 1: OLTP Join Optimizer — Easy (3 tables)

**Scenarios:** E-commerce catalog lookup, SaaS subscription query, inventory reorder check

All tables have covering indexes. Statistics are fresh and accurate. The agent must learn:
- Join smaller dimension tables (categories, suppliers) before larger fact tables
- Always prefer merge-sort joins
- Use indexes to reduce base row counts

**Scoring:** `score = (worst_cost - actual_cost) / (worst_cost - best_cost)` in [0.0, 1.0]

Expected baseline (random agent): ~0.35 | Expected upper bound (optimal): ~0.95+

---

### Task 2: OLAP Join Optimizer — Medium (5 tables)

**Scenarios:** E-commerce order fulfillment, marketing funnel analytics, financial transaction summary

Mix of indexed and unindexed tables. Cardinality estimates have realistic noise (σ up to 0.25). The agent must simultaneously:
- Identify efficient join orderings by reasoning about **estimated** table sizes
- Handle missing indexes by choosing merge-sort over nested-loop
- Navigate 5! = 120 possible orderings

Expected baseline (random agent): ~0.25 | Expected upper bound (optimal): ~0.85+

---

### Task 3: Complex OLAP Join Optimizer — Hard (7 tables)

**Scenarios:** Full e-commerce pipeline audit, financial fraud detection, user journey attribution

Seven-table joins with billion-row event tables, missing indexes, and high estimation noise (σ up to 0.60). The optimal join order requires understanding which tables filter aggressively via selectivity, which have indexes, and which are so large they must come last.

- 7! = **5,040 possible join orderings**
- Estimation errors up to ±80% on raw row counts
- Three join algorithms × two index options per step = 6 action variants per table

Expected baseline (random agent): ~0.15 | Expected upper bound (frontier LLM): ~0.70+

---

## Reward Function

Rewards are **normalised to [0.0, 1.0]** at every step — never sparse, never binary.

```
Per-step reward = (worst_step_cost - actual_step_cost) / (worst_step_cost - best_step_cost)

Final reward    = (worst_total_cost - actual_total_cost) / (worst_total_cost - best_total_cost)
```

| Scenario | Reward |
|---|---|
| Optimal join choice (merge-sort + index) | 1.0 |
| Hash join, index used | ~0.7–0.9 |
| Merge-sort, no index (table has none) | 1.0 |
| Hash join, ignores available index | 0.4–0.6 |
| Nested-loop join | 0.0 |
| Invalid table selection (already joined) | 0.0 |

This design gives the agent a **learning signal at every step**, enabling effective credit assignment for RL from any trajectory.

---

## Setup & Usage

### Prerequisites

```bash
pip install openenv-core openai
```

### Run the Server Locally

```bash
# From the Mirage_RL/ directory
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
# Build
docker build -t mirage-rl:latest .

# Run
docker run -p 8000:8000 mirage-rl:latest
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{"task_id": "easy\|medium\|hard", "seed": 42}` |
| `/step` | POST | Execute action. Body: `{"next_table": 0, "join_type": 2, "use_index": 1}` |
| `/state` | GET | Get current environment state |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |

### Run Baseline Inference Script

```bash
# Set required environment variables
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Run against all 3 tasks
python inference.py
```

### Python Client

```python
from Mirage_RL import QueryClient, QueryAction

with QueryClient(base_url="http://localhost:8000").sync() as env:
    result = env.reset()                 # starts easy task by default
    obs = result.observation
    print(f"Query: {obs.query_context}")
    print(f"Tables: {obs.tables}")

    result = env.step(QueryAction(
        next_table=0,   # join first table
        join_type=2,    # merge-sort
        use_index=1,    # use index
    ))
    print(f"Reward: {result.reward:.4f}")
```

---

## Baseline Scores

Scores produced by running `inference.py` with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router:

| Task | Difficulty | Baseline Score |
|---|---|---|
| OLTP 3-table | Easy | ~0.82 |
| OLAP 5-table | Medium | ~0.68 |
| Complex OLAP 7-table | Hard | ~0.51 |

> Scores are approximate and vary by episode (random scenario sampling). For reproducible results, pass `seed=42` to `reset()`.

---

## Project Structure

```
Mirage_RL/
├── Dockerfile                         # Container image (root-level for HF Spaces)
├── README.md                          # This file
├── openenv.yaml                       # OpenEnv manifest (spec_version, tasks, schemas)
├── pyproject.toml                     # Project metadata and dependencies
├── inference.py                       # Baseline inference script (mandatory)
├── models.py                          # Pydantic models: QueryAction, QueryObservation, QueryState
├── client.py                          # QueryClient: HTTP client for the server
├── __init__.py                        # Package exports
└── server/
    ├── app.py                         # FastAPI application (HTTP endpoints)
    ├── Mirage_RL_environment.py       # Core environment logic (reset/step/state)
    ├── tasks.py                       # Enterprise scenario definitions + graders
    ├── Dockerfile                     # Docker build (also accessible from server/)
    └── requirements.txt               # Minimal server dependencies
```

---

## Validation

Run the OpenEnv pre-submission validator:

```bash
# From the Mirage_RL/ directory
openenv validate

# Full submission validator (requires deployed HF Space URL)
bash validate-submission.sh https://your-space.hf.space .
```

---

## References

- [Neo: A Learned Query Optimizer](https://vldb.org/pvldb/vol12/p1705-marcus.pdf) — Marcus et al., VLDB 2019
- [Bao: Making Learned Query Optimization Practical](https://dl.acm.org/doi/10.1145/3448016.3452838) — Marcus et al., SIGMOD 2021
- [Are We Ready for Learned Cardinality Estimation?](https://vldb.org/pvldb/vol14/p1640-wang.pdf) — Wang et al., VLDB 2021
- [PostgreSQL Query Planner](https://www.postgresql.org/docs/current/planner-optimizer.html) — Production optimizer documentation
