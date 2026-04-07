"""
tasks.py — Enterprise-grade query optimization task definitions for Mirage_RL.

Each task tier (easy / medium / hard) contains a pool of production-realistic
query scenarios sampled randomly per episode so agents cannot memorize solutions.

Domains covered:
  - E-commerce (OLTP):   orders, customers, products, inventory, suppliers
  - Analytics (OLAP):    events, sessions, campaigns, conversions, attribution
  - Financial (OLAP):    transactions, accounts, merchants, fraud signals, risk

Key design decisions:
  - Table cardinalities match realistic production scales (10K – 1B rows)
  - Cardinality estimation noise (log-normal) simulates the core real-world
    challenge: a planner's row estimates are always wrong. Noise sigma maps to:
      σ=0.04  excellent statistics (small, frequently vacuumed)
      σ=0.15  typical OLTP tables
      σ=0.25  typical analytics/OLAP tables
      σ=0.40  poor statistics (large tables, column correlations)
      σ=0.60  very poor (event tables, multi-column predicates, data skew)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Schema building-blocks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TableSpec:
    """Configuration for one table in a query scenario."""
    name:         str
    true_rows:    int    # ground-truth row count (used for actual cost)
    selectivity:  float  # join predicate selectivity (fraction of rows kept)
    has_index:    int    # 1 = covering/clustered index present, 0 = heap scan
    noise_sigma:  float  # log-normal σ for cardinality estimation error


@dataclass
class Scenario:
    """A complete multi-table query scenario."""
    name:          str            # short identifier
    domain:        str            # "ecommerce" | "analytics" | "financial" | "saas"
    query_context: str            # SQL-style description shown to the agent
    tables:        List[TableSpec]


@dataclass
class TaskConfig:
    task_id:     str
    name:        str
    difficulty:  str              # "easy" | "medium" | "hard"
    description: str
    scenarios:   List[Scenario]   # pool sampled randomly per episode


# ─────────────────────────────────────────────────────────────────────────────
# Production Scenario Pools
# ─────────────────────────────────────────────────────────────────────────────

# ── EASY  (3 tables, clean statistics, all indexes present) ──────────────────
EASY_SCENARIOS: List[Scenario] = [
    Scenario(
        name="ecommerce_catalog_lookup",
        domain="ecommerce",
        query_context=(
            "SELECT p.name, c.label, s.region "
            "FROM products JOIN categories ON p.category_id=c.id "
            "JOIN suppliers ON p.supplier_id=s.id "
            "WHERE c.segment='Electronics' AND s.active=true"
        ),
        tables=[
            TableSpec("products",   true_rows=500_000,  selectivity=0.04, has_index=1, noise_sigma=0.05),
            TableSpec("categories", true_rows=10_000,   selectivity=0.45, has_index=1, noise_sigma=0.03),
            TableSpec("suppliers",  true_rows=50_000,   selectivity=0.12, has_index=1, noise_sigma=0.04),
        ],
    ),
    Scenario(
        name="saas_active_subscriptions",
        domain="saas",
        query_context=(
            "SELECT u.email, a.plan, s.renewal_date "
            "FROM users JOIN accounts ON u.account_id=a.id "
            "JOIN subscriptions ON a.id=s.account_id "
            "WHERE a.tier='enterprise' AND s.status='active'"
        ),
        tables=[
            TableSpec("users",         true_rows=2_000_000,  selectivity=0.08, has_index=1, noise_sigma=0.05),
            TableSpec("accounts",      true_rows=800_000,    selectivity=0.15, has_index=1, noise_sigma=0.04),
            TableSpec("subscriptions", true_rows=1_200_000,  selectivity=0.10, has_index=1, noise_sigma=0.04),
        ],
    ),
    Scenario(
        name="inventory_reorder_check",
        domain="ecommerce",
        query_context=(
            "SELECT p.sku, w.location, i.quantity "
            "FROM products JOIN warehouses ON i.warehouse_id=w.id "
            "JOIN inventory ON p.id=i.product_id "
            "WHERE i.quantity < p.reorder_point AND w.region='US-WEST'"
        ),
        tables=[
            TableSpec("products",   true_rows=500_000,   selectivity=0.05, has_index=1, noise_sigma=0.04),
            TableSpec("warehouses", true_rows=5_000,     selectivity=0.55, has_index=1, noise_sigma=0.03),
            TableSpec("inventory",  true_rows=2_500_000, selectivity=0.03, has_index=1, noise_sigma=0.06),
        ],
    ),
]

# ── MEDIUM  (5 tables, realistic noise, mixed index coverage) ─────────────────
MEDIUM_SCENARIOS: List[Scenario] = [
    Scenario(
        name="ecommerce_order_fulfillment",
        domain="ecommerce",
        query_context=(
            "SELECT o.id, c.name, p.sku, cat.label, w.region "
            "FROM orders JOIN customers ON o.customer_id=c.id "
            "JOIN products ON o.product_id=p.id "
            "JOIN categories ON p.category_id=cat.id "
            "JOIN warehouses ON o.warehouse_id=w.id "
            "WHERE o.status='pending' AND o.created_at > NOW() - INTERVAL '7 days'"
        ),
        tables=[
            TableSpec("orders",     true_rows=10_000_000, selectivity=0.08, has_index=1, noise_sigma=0.15),
            TableSpec("customers",  true_rows=2_000_000,  selectivity=0.12, has_index=1, noise_sigma=0.10),
            TableSpec("products",   true_rows=500_000,    selectivity=0.05, has_index=1, noise_sigma=0.08),
            TableSpec("categories", true_rows=10_000,     selectivity=0.40, has_index=1, noise_sigma=0.05),
            TableSpec("warehouses", true_rows=5_000,      selectivity=0.55, has_index=0, noise_sigma=0.05),
        ],
    ),
    Scenario(
        name="marketing_funnel_analytics",
        domain="analytics",
        query_context=(
            "SELECT s.id, u.segment, c.name, cv.revenue, ch.source "
            "FROM sessions JOIN users ON s.user_id=u.id "
            "JOIN campaigns ON s.campaign_id=c.id "
            "JOIN conversions ON s.id=cv.session_id "
            "JOIN channels ON c.channel_id=ch.id "
            "WHERE c.type='paid' AND cv.revenue > 0"
        ),
        tables=[
            TableSpec("sessions",    true_rows=50_000_000, selectivity=0.05, has_index=1, noise_sigma=0.25),
            TableSpec("users",       true_rows=5_000_000,  selectivity=0.15, has_index=1, noise_sigma=0.12),
            TableSpec("campaigns",   true_rows=100_000,    selectivity=0.25, has_index=1, noise_sigma=0.07),
            TableSpec("conversions", true_rows=2_000_000,  selectivity=0.08, has_index=1, noise_sigma=0.18),
            TableSpec("channels",    true_rows=500,        selectivity=0.70, has_index=1, noise_sigma=0.02),
        ],
    ),
    Scenario(
        name="financial_transaction_summary",
        domain="financial",
        query_context=(
            "SELECT t.amount, a.holder, m.name, rs.score, cur.symbol "
            "FROM transactions JOIN accounts ON t.account_id=a.id "
            "JOIN merchants ON t.merchant_id=m.id "
            "JOIN risk_scores ON a.id=rs.account_id "
            "JOIN currencies ON t.currency_code=cur.code "
            "WHERE t.created_at >= CURRENT_DATE - 30 AND a.status='active'"
        ),
        tables=[
            TableSpec("transactions", true_rows=100_000_000, selectivity=0.03, has_index=1, noise_sigma=0.20),
            TableSpec("accounts",     true_rows=10_000_000,  selectivity=0.10, has_index=1, noise_sigma=0.10),
            TableSpec("merchants",    true_rows=500_000,     selectivity=0.18, has_index=1, noise_sigma=0.08),
            TableSpec("risk_scores",  true_rows=10_000_000,  selectivity=0.08, has_index=0, noise_sigma=0.20),
            TableSpec("currencies",   true_rows=200,         selectivity=0.80, has_index=1, noise_sigma=0.02),
        ],
    ),
]

# ── HARD  (7 tables, high noise, missing indexes, skewed large tables) ────────
HARD_SCENARIOS: List[Scenario] = [
    Scenario(
        name="ecommerce_full_pipeline_audit",
        domain="ecommerce",
        query_context=(
            "SELECT o.id, oi.qty, p.sku, cat.label, c.name, s.contact, w.region "
            "FROM orders JOIN order_items ON o.id=oi.order_id "
            "JOIN products ON oi.product_id=p.id "
            "JOIN categories ON p.category_id=cat.id "
            "JOIN customers ON o.customer_id=c.id "
            "JOIN suppliers ON p.supplier_id=s.id "
            "JOIN warehouses ON oi.warehouse_id=w.id "
            "WHERE o.created_at >= '2024-01-01' AND c.country='US' "
            "AND cat.segment='Electronics'"
        ),
        tables=[
            TableSpec("categories",  true_rows=1_000_000,   selectivity=0.10, has_index=1, noise_sigma=0.05), # 100K
            TableSpec("warehouses",  true_rows=3_000_000,   selectivity=0.10, has_index=0, noise_sigma=0.08), # 300K
            TableSpec("products",    true_rows=10_000_000,  selectivity=0.05, has_index=1, noise_sigma=0.10), # 500K
            TableSpec("suppliers",   true_rows=8_000_000,   selectivity=0.10, has_index=0, noise_sigma=0.12), # 800K
            TableSpec("customers",   true_rows=10_000_000,  selectivity=0.12, has_index=1, noise_sigma=0.15), # 1.2M
            TableSpec("orders",      true_rows=20_000_000,  selectivity=0.08, has_index=1, noise_sigma=0.20), # 1.6M
            TableSpec("order_items", true_rows=40_000_000,  selectivity=0.05, has_index=0, noise_sigma=0.35), # 2.0M
        ],
    ),
    Scenario(
        name="fraud_detection_pipeline",
        domain="financial",
        query_context=(
            "SELECT t.id, a.holder, m.category, fl.label, rs.score, d.fingerprint, loc.country "
            "FROM transactions JOIN accounts ON t.account_id=a.id "
            "JOIN merchants ON t.merchant_id=m.id "
            "JOIN fraud_labels ON t.id=fl.transaction_id "
            "JOIN risk_scores ON a.id=rs.account_id "
            "JOIN devices ON t.device_id=d.id "
            "JOIN locations ON t.location_id=loc.id "
            "WHERE t.amount > 10000 AND fl.is_flagged=true"
        ),
        tables=[
            TableSpec("locations",    true_rows=1_000_000,   selectivity=0.10, has_index=1, noise_sigma=0.10), # 100K
            TableSpec("fraud_labels", true_rows=3_000_000,   selectivity=0.10, has_index=0, noise_sigma=0.40), # 300K
            TableSpec("merchants",    true_rows=5_000_000,   selectivity=0.10, has_index=1, noise_sigma=0.08), # 500K
            TableSpec("devices",      true_rows=16_000_000,  selectivity=0.05, has_index=1, noise_sigma=0.25), # 800K
            TableSpec("risk_scores",  true_rows=12_000_000,  selectivity=0.10, has_index=0, noise_sigma=0.30), # 1.2M
            TableSpec("accounts",     true_rows=20_000_000,  selectivity=0.08, has_index=1, noise_sigma=0.12), # 1.6M
            TableSpec("transactions", true_rows=50_000_000,  selectivity=0.04, has_index=1, noise_sigma=0.25), # 2.0M
        ],
    ),
    Scenario(
        name="user_journey_attribution",
        domain="analytics",
        query_context=(
            "SELECT e.event_type, s.duration, u.segment, c.name, cv.revenue, ab.variant, pv.url "
            "FROM events JOIN sessions ON e.session_id=s.id "
            "JOIN users ON s.user_id=u.id "
            "JOIN campaigns ON s.campaign_id=c.id "
            "JOIN conversions ON s.id=cv.session_id "
            "JOIN ab_tests ON u.id=ab.user_id "
            "JOIN page_views ON s.id=pv.session_id "
            "WHERE s.started_at >= CURRENT_DATE - 7 AND c.status='active'"
        ),
        tables=[
            # Rebalanced: each table contributes cleanly to the cost range (no single dominator >30%)
            TableSpec("campaigns",   true_rows=1_000_000,   selectivity=0.10, has_index=1, noise_sigma=0.08), # 100K
            TableSpec("ab_tests",    true_rows=3_000_000,   selectivity=0.10, has_index=0, noise_sigma=0.30), # 300K
            TableSpec("users",       true_rows=5_000_000,   selectivity=0.10, has_index=1, noise_sigma=0.12), # 500K
            TableSpec("conversions", true_rows=16_000_000,  selectivity=0.05, has_index=1, noise_sigma=0.20), # 800K
            TableSpec("sessions",    true_rows=15_000_000,  selectivity=0.08, has_index=1, noise_sigma=0.25), # 1.2M
            TableSpec("events",      true_rows=32_000_000,  selectivity=0.05, has_index=0, noise_sigma=0.50), # 1.6M
            TableSpec("page_views",  true_rows=40_000_000,  selectivity=0.05, has_index=0, noise_sigma=0.55), # 2.0M
        ],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Task Registry
# ─────────────────────────────────────────────────────────────────────────────

TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        name="OLTP Join Optimizer — 3-Table Queries",
        difficulty="easy",
        description=(
            "Production OLTP queries joining 3 tables from e-commerce and SaaS schemas. "
            "All indexes are available. Statistics are accurate. "
            "Goal: select the optimal join order and strategy. "
            "Scoring penalises nested-loop joins and missed indexes."
        ),
        scenarios=EASY_SCENARIOS,
    ),
    "medium": TaskConfig(
        task_id="medium",
        name="OLAP Join Optimizer — 5-Table Queries with Estimation Noise",
        difficulty="medium",
        description=(
            "Analytical queries joining 5 tables across e-commerce, marketing, "
            "and financial schemas. Some tables lack indexes. "
            "Cardinality estimates contain realistic noise (σ 0.05–0.25), "
            "requiring the agent to reason under uncertainty about true table sizes."
        ),
        scenarios=MEDIUM_SCENARIOS,
    ),
    "hard": TaskConfig(
        task_id="hard",
        name="Complex OLAP Join Optimizer — 7-Table Queries with High Estimation Noise",
        difficulty="hard",
        description=(
            "Enterprise analytical queries joining 7 tables with billion-row event "
            "tables, missing indexes, and high cardinality estimation noise (σ up to 0.60). "
            "Models real-world conditions: data skew, stale statistics, and column correlations "
            "that cause traditional planners to underestimate result sizes by 10–100×."
        ),
        scenarios=HARD_SCENARIOS,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Cardinality Estimation Noise
# ─────────────────────────────────────────────────────────────────────────────

def apply_estimation_noise(true_rows: int, noise_sigma: float, rng: random.Random) -> int:
    """
    Simulate cardinality estimation error using log-normal noise.

    real planners: estimated_rows = true_rows × exp(N(0, σ²))

    σ=0.05 → ~5% error  (excellent statistics)
    σ=0.15 → ~15% error (typical OLTP)
    σ=0.25 → ~28% error (typical OLAP)
    σ=0.40 → ~49% error (stale statistics / large tables)
    σ=0.60 → ~82% error (data skew, column correlations)
    """
    if noise_sigma <= 0:
        return true_rows
    log_factor = rng.gauss(0.0, noise_sigma)
    estimated = int(true_rows * math.exp(log_factor))
    return max(1, estimated)


# ─────────────────────────────────────────────────────────────────────────────
# Cost Bound Computation  (based on true_rows, not estimates)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost_bounds(tables: List[TableSpec]) -> tuple[float, float]:
    """
    Analytical worst/best cost bounds for a scenario, accounting for cascading sizes.
    Joining early inflates all subsequent step costs via an additive intermediate size penalty.

    Worst: worst ordering by size (largest first), nested-loop (2.0×), no index
    Best:  best ordering by size (smallest first), merge-sort (0.8×), use index
    """
    # Order by intermediate output size to find best and worst sequences
    outputs = [(t.true_rows * t.selectivity, t) for t in tables]
    best_order = sorted(outputs, key=lambda x: x[0])
    worst_order = sorted(outputs, key=lambda x: x[0], reverse=True)

    def simulate_cost(order_list, is_worst: bool) -> float:
        total_cost = 0.0
        running_size = 0.0  # Cost penalty is 0 on the first table
        
        for output_size, t in order_list:
            if is_worst:
                step_cost = (t.true_rows * t.selectivity * 2.0) + running_size
            else:
                best_base = t.true_rows * 0.5 if t.has_index else t.true_rows
                step_cost = (best_base * t.selectivity * 0.8) + running_size
                
            total_cost += step_cost
            
            # For the first table, running_size becomes output_size. Afterwards it multiplies.
            if running_size == 0.0:
                running_size = output_size
            else:
                running_size *= output_size
                
        return total_cost

    worst = simulate_cost(worst_order, is_worst=True)
    best  = simulate_cost(best_order, is_worst=False)
    return worst, best


def compute_step_cost_bounds(table: TableSpec, running_size: float = 1.0) -> tuple[float, float]:
    """Worst/best cost bounds for a single table step, given additive penalty."""
    # Note: running_size defaults to 1.0 in env (or actual size later), we subtract 1.0 to behave like 0.0 on step 1
    penalty   = max(0.0, running_size - 1.0)
    worst     = (table.true_rows * table.selectivity * 2.0) + penalty
    best_base = table.true_rows * 0.5 if table.has_index else table.true_rows
    best      = (best_base * table.selectivity * 0.8) + penalty
    return worst, best


# ─────────────────────────────────────────────────────────────────────────────
# Order Quality Scorer  —  measures how well the join sequence minimises
# intermediate result explosion (independent of join algorithm choices)
# ─────────────────────────────────────────────────────────────────────────────

def compute_order_quality(chosen_order: List[int], tables: List[TableSpec]) -> float:
    """
    Score the join ordering quality using log-normalised rank mismatch.

    Optimal order: join tables with smallest (true_rows × selectivity) first,
    because this minimises intermediate result sizes throughout the sequence.
    Joining a large-output table early compounds the cost of every subsequent
    join — this is the core of real query optimisation difficulty.

    Method:
      1. Compute each table's output size = true_rows × selectivity
      2. For each position k, compare the log-output of the chosen k-th table
         against the log-output of the k-th optimal table (sorted ascending)
      3. Normalise mismatch against the worst possible ordering (descending)

    Returns 1.0 for a perfectly sorted order, 0.0 for reverse optimal.
    """
    n = len(tables)
    if n <= 1:
        return 1.0

    outputs = [t.true_rows * t.selectivity for t in tables]
    chosen_outputs  = [outputs[i] for i in chosen_order]
    optimal_outputs = sorted(outputs)                        # ascending — best ordering
    worst_outputs   = list(reversed(optimal_outputs))        # descending — worst ordering

    def log_mismatch(order_outputs: List[float]) -> float:
        return sum(
            abs(math.log1p(order_outputs[k]) - math.log1p(optimal_outputs[k]))
            for k in range(n)
        )

    actual_mismatch = log_mismatch(chosen_outputs)
    max_mismatch    = log_mismatch(worst_outputs)

    if max_mismatch < 1e-9:
        return 1.0

    score = 1.0 - actual_mismatch / max_mismatch
    return float(max(0.0, min(1.0, score)))


# ─────────────────────────────────────────────────────────────────────────────
# Grader  —  returns float in [0.0, 1.0]
# ─────────────────────────────────────────────────────────────────────────────

def grade(
    tables: List[TableSpec],
    final_cost: float,
    chosen_order: Optional[List[int]] = None,
) -> float:
    """
    Grade a completed episode.  Combined score = 60% method + 40% order.

    method_score: how well the agent chose join algorithms and index usage.
      = (worst_cost - actual_cost) / (worst_cost - best_cost)
      worst = all nested-loop, no indexes
      best  = all merge-sort, use index when available

    order_score: how well the agent sequenced the joins.
      Optimal = smallest-output tables first (minimises intermediate blowup).
      Computed via log-normalised rank mismatch vs optimal ordering.
      Only applied when all tables have been joined (full episode).

    Combined = 0.60 × method_score + 0.40 × order_score

    This ensures:
      - A model that always picks merge_sort but joins biggest tables first
        scores ~0.60 instead of ~1.00 on hard tasks.
      - A model with correct ordering but mediocre method choices still
        receives partial credit from the order component.
      - Difficulty genuinely increases: easy (3 tables, 6 orderings) vs
        hard (7 tables, 5040 orderings) differentiates planning depth.
    """
    # ── Method quality (join algorithm + index usage) ─────────────────────────
    worst, best = compute_cost_bounds(tables)
    if worst == best:
        method_score = 1.0
    else:
        method_score = float(max(0.0, min(1.0, (worst - final_cost) / (worst - best))))

    # ── Order quality (join sequence planning) ────────────────────────────────
    if chosen_order is not None and len(chosen_order) == len(tables):
        order_score = compute_order_quality(chosen_order, tables)
    else:
        # Incomplete episode or no order info — fall back to method score only
        order_score = method_score

    # ── Combined score ────────────────────────────────────────────────────────
    combined = 0.60 * method_score + 0.40 * order_score
    return float(max(0.0, min(1.0, combined)))
