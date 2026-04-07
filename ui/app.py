import streamlit as st
import time
import sys
import os

# Add project root to path so Mirage_RL package is found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Mirage_RL.client import QueryClient
from Mirage_RL.models import QueryAction
from Mirage_RL.training.agent import Agent

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mirage RL — DBMS Query Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border: 1px solid #3a3a5e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card h3 { color: #a0aec0; font-size: 0.85rem; font-weight: 400; margin: 0 0 8px 0; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: #7c3aed; }
.metric-card .sub { font-size: 0.75rem; color: #718096; margin-top: 4px; }

.step-card {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    border-left: 4px solid #7c3aed;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
    animation: fadeIn 0.4s ease;
}
.step-card .step-title { color: #e2e8f0; font-weight: 600; font-size: 1rem; }
.step-card .step-detail { color: #94a3b8; font-size: 0.85rem; margin-top: 6px; }
.step-card .reward-pos { color: #10b981; font-weight: 600; }
.step-card .reward-neg { color: #f87171; font-weight: 600; }

.table-chip {
    display: inline-block;
    background: #7c3aed22;
    border: 1px solid #7c3aed66;
    color: #a78bfa;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 2px;
    font-weight: 600;
}
.table-chip.chosen {
    background: #10b98122;
    border-color: #10b98166;
    color: #34d399;
}

.badge-hash   { background: #3b82f622; border: 1px solid #3b82f666; color: #60a5fa; border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; }
.badge-nested { background: #f59e0b22; border: 1px solid #f59e0b66; color: #fbbf24; border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; }
.badge-merge  { background: #10b98122; border: 1px solid #10b98166; color: #34d399; border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; }

.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #3a3a5e;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    text-align: center;
}
.hero-banner h1 { color: #e2e8f0; font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero-banner p  { color: #94a3b8; font-size: 1rem; margin: 10px 0 0 0; }

.pill-green { background: #10b98122; border: 1px solid #10b981; color: #34d399; border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; font-weight: 600; display: inline-block; }
.pill-red   { background: #ef444422; border: 1px solid #ef4444; color: #f87171; border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; font-weight: 600; display: inline-block; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🧠 Mirage RL — DBMS Query Optimizer</h1>
  <p>Reinforcement Learning agent learns the optimal join order, join strategy, and index usage for SQL queries</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    server_url = st.text_input("Server URL", value="http://localhost:8000")
    mode = st.radio("🎮 Mode", ["AI Agent (Trained)", "Manual Control"])
    st.markdown("---")
    st.markdown("### 📖 Join Types")
    st.markdown("**0 — Hash Join** · Fast for large, unordered sets")
    st.markdown("**1 — Nested Loop** · Good for small tables or indexed lookups")
    st.markdown("**2 — Merge Join** · Efficient for sorted/pre-sorted data")
    st.markdown("---")
    st.markdown("### 🗄️ Tables in Query")
    st.markdown("| Table | Rows | Selectivity | Index |")
    st.markdown("|---|---|---|---|")
    st.markdown("| A | 1,000 | 10% | ✅ |")
    st.markdown("| B | 5,000 | 50% | ❌ |")
    st.markdown("| C | 200 | 5% | ✅ |")

# ─── Connection check ─────────────────────────────────────────────────────────
import requests
col_status, col_btn = st.columns([3, 1])
with col_status:
    try:
        r = requests.get(f"{server_url}/health", timeout=2)
        st.markdown('<span class="pill-green">🟢 Server Connected</span>', unsafe_allow_html=True)
        server_ok = True
    except Exception:
        try:
            r = requests.get(f"{server_url}/docs", timeout=2)
            st.markdown('<span class="pill-green">🟢 Server Connected</span>', unsafe_allow_html=True)
            server_ok = True
        except Exception:
            st.markdown('<span class="pill-red">🔴 Server Offline — open new terminal and run: uv run server</span>', unsafe_allow_html=True)
            server_ok = False

agent = Agent(num_tables=3)

# ─── Run button ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run_col, _ = st.columns([1, 3])
with run_col:
    run_clicked = st.button("▶ Run Simulation", type="primary", use_container_width=True, disabled=not server_ok)

if run_clicked:
    env = QueryClient(base_url=server_url).sync().__enter__()
    result = env.reset()
    obs = result.observation

    # ── Top metrics row ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Query Environment")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><h3>Tables</h3><div class="value">{len(obs.tables)}</div><div class="sub">A, B, C</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><h3>Total Rows</h3><div class="value">{sum(obs.table_rows):,}</div><div class="sub">across all tables</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><h3>Indexed Tables</h3><div class="value">{sum(obs.has_index)}/{len(obs.tables)}</div><div class="sub">A and C</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><h3>Mode</h3><div class="value">{"🤖" if "AI" in mode else "🕹️"}</div><div class="sub">{mode.split("(")[0].strip()}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Table details ──
    st.markdown("### 🗂️ Table Details")
    table_cols = st.columns(len(obs.tables))
    table_names = obs.tables
    for i, col in enumerate(table_cols):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <h3>Table {table_names[i]}</h3>
              <div class="value" style="font-size:1.4rem">{obs.table_rows[i]:,}</div>
              <div class="sub">rows</div>
              <div class="sub" style="margin-top:8px">Selectivity: <b style="color:#a78bfa">{obs.selectivities[i]*100:.0f}%</b></div>
              <div class="sub">Index: <b style="color:{'#34d399' if obs.has_index[i] else '#f87171'}">{'✅ Yes' if obs.has_index[i] else '❌ No'}</b></div>
            </div>
            """, unsafe_allow_html=True)

    # ── Simulation loop ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔄 Step-by-Step Execution")

    done = False
    step = 0
    total_reward = 0.0
    step_log = []

    cost_chart_placeholder = st.empty()
    steps_placeholder = st.empty()

    join_names = {0: "Hash Join", 1: "Nested Loop", 2: "Merge Join"}
    join_badges = {0: "badge-hash", 1: "badge-nested", 2: "badge-merge"}

    while not done:
        step += 1

        if "Manual" in mode:
            st.markdown(f"**Step {step} — Choose action for remaining tables: {[table_names[t] for t in obs.remaining_tables]}**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                table_choice = st.selectbox("Next Table", obs.remaining_tables,
                    format_func=lambda x: f"Table {table_names[x]} ({obs.table_rows[x]:,} rows)",
                    key=f"t{step}")
            with col_b:
                join_choice = st.selectbox("Join Type", [0, 1, 2],
                    format_func=lambda x: join_names[x], key=f"j{step}")
            with col_c:
                index_choice = st.selectbox("Use Index", [0, 1],
                    format_func=lambda x: "Yes ✅" if x else "No ❌", key=f"i{step}")
            table, join, index = table_choice, join_choice, index_choice
        else:
            (table, join, index), _ = agent.select_action(obs)

        action = QueryAction(next_table=table, join_type=join, use_index=index)
        result = env.step(action)
        obs = result.observation
        done = result.done
        reward = result.reward
        total_reward += reward

        step_log.append({
            "step": step,
            "table": table_names[table],
            "join": join_names[join],
            "index": bool(index),
            "cost": obs.current_cost,
            "reward": reward,
        })

        # Rebuild steps display
        steps_html = ""
        for s in step_log:
            r_class = "reward-pos" if s["reward"] >= 0 else "reward-neg"
            r_sign = "+" if s["reward"] >= 0 else ""
            badge_cls = join_badges[list(join_names.values()).index(s["join"])]
            idx_label = "Index ✅" if s["index"] else "No Index ❌"
            steps_html += f"""
            <div class="step-card">
              <div class="step-title">Step {s['step']} → Join Table <span class="table-chip chosen">{s['table']}</span></div>
              <div class="step-detail">
                <span class="{badge_cls}">{s['join']}</span>&nbsp;
                &nbsp;{idx_label}&nbsp;&nbsp;
                Cumulative Cost: <b style="color:#e2e8f0">{s['cost']:.1f}</b>
                &nbsp;&nbsp;Reward: <span class="{r_class}">{r_sign}{s['reward']:.1f}</span>
              </div>
            </div>"""

        steps_placeholder.markdown(steps_html, unsafe_allow_html=True)

        if "AI" in mode:
            time.sleep(0.6)

    # ── Final Results ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏆 Final Results")

    final_cost = obs.current_cost
    chosen_order = [table_names[i] for i in obs.chosen_order]

    r1, r2, r3 = st.columns(3)
    r1.markdown(f'<div class="metric-card"><h3>Final Query Cost</h3><div class="value" style="color:#f87171">{final_cost:.1f}</div><div class="sub">lower is better</div></div>', unsafe_allow_html=True)
    r2.markdown(f'<div class="metric-card"><h3>Total Reward</h3><div class="value" style="color:#{"10b981" if total_reward >= 0 else "f87171"}">{total_reward:.1f}</div><div class="sub">agent performance</div></div>', unsafe_allow_html=True)
    r3.markdown(f'<div class="metric-card"><h3>Join Order</h3><div class="value" style="font-size:1.4rem">{"→".join(chosen_order)}</div><div class="sub">chosen sequence</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Cost breakdown chart ──
    st.markdown("### 📈 Cost Accumulation Per Step")
    import pandas as pd
    df = pd.DataFrame(step_log)
    st.line_chart(df.set_index("step")["cost"], use_container_width=True)

    st.markdown("### 📋 Full Step Log")
    st.dataframe(df.rename(columns={
        "step": "Step", "table": "Table Joined", "join": "Join Type",
        "index": "Used Index", "cost": "Cumulative Cost", "reward": "Reward"
    }), use_container_width=True)

    if total_reward > -500:
        st.success(f"✅ Simulation complete! Final cost: **{final_cost:.1f}** | Join order: **{'→'.join(chosen_order)}**")
    else:
        st.warning(f"⚠️ High cost path taken. Final cost: **{final_cost:.1f}**. Try AI Agent mode for better results.")