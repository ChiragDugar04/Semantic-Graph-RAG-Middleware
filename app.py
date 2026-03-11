"""
app.py

🕸️ Semantic Graph RAG Assistant — Streamlit Frontend
Chat-style interface with persistent Glass Box audit trail.

Run with:
    streamlit run app.py

FIX NOTES:
  - Removed st.rerun() after answer — audit trail now persists until
    next question. History loop re-renders all previous traces.
  - Added ⓪ Semantic Graph Traversal panel showing path, method,
    tables, and timing from the new graph pipeline.
"""

from __future__ import annotations

import yaml
import requests
import mysql.connector
import streamlit as st
import pandas as pd

from pathlib import Path
from datetime import datetime

from middleware.pipeline import run_pipeline, _graph
from middleware.models import MiddlewareTrace


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Semantic Graph RAG Assistant",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; }
[data-testid="stExpander"] summary { font-size: 0.82rem; color: #555; }
[data-testid="stMetricLabel"] p { font-size: 0.72rem !important; }
.stCodeBlock code { font-size: 0.78rem !important; }
.stCaption { color: #888 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state() -> None:
    defaults = {
        "messages":             [],
        "total_questions":      0,
        "healing_count":        0,
        "db_latencies":         [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# STATUS HELPERS
# ============================================================

def check_mysql() -> tuple[bool, str]:
    try:
        config_path = Path("config/db_config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["database"]
        conn = mysql.connector.connect(
            host=cfg["host"], port=cfg["port"],
            user=cfg["user"], password=cfg["password"],
            database=cfg["database"], connection_timeout=5,
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM employees")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return True, f"{cfg['database']} · {count} employees"
    except Exception as e:
        return False, str(e)[:50]


def check_ollama() -> tuple[bool, bool]:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        fast_ok  = any("qwen2.5:1.5b" in m for m in models)
        synth_ok = any("llama3.2:3b"  in m for m in models)
        return fast_ok, synth_ok
    except Exception:
        return False, False


def check_graph() -> tuple[int, int]:
    try:
        return _graph.node_count, _graph.edge_count
    except Exception:
        return 0, 0


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.header("🕸️ Semantic Graph RAG")
        st.caption("MySQL · Semantic Graph · LLM Synthesis")
        st.caption("No Vector DB · No Fine-tuning · No Text-to-SQL")

        st.divider()

        st.markdown("**System Status**")
        mysql_ok, mysql_msg = check_mysql()
        fast_ok, synth_ok = check_ollama()
        nodes, edges = check_graph()

        st.caption(f"{'🟢' if mysql_ok  else '🔴'} MySQL — {mysql_msg}")
        st.caption(f"{'🟢' if fast_ok   else '🔴'} qwen2.5:1.5b (extraction)")
        st.caption(f"{'🟢' if synth_ok  else '🔴'} llama3.2:3b (synthesis)")
        st.caption(f"🟢 Semantic Graph — {nodes} nodes · {edges} edges")

        st.divider()

        st.markdown("**Session Stats**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.total_questions)
        with col2:
            st.metric("Healed", st.session_state.healing_count)

        if st.session_state.db_latencies:
            avg_db = sum(st.session_state.db_latencies) / len(st.session_state.db_latencies)
            st.metric("Avg DB Time", f"{avg_db:.2f}ms")

        st.divider()

        st.markdown("**Architecture**")
        st.caption(
            "Question\n"
            "↓ rules + qwen2.5:1.5b\n"
            "Entity Extraction\n"
            "↓ NetworkX graph\n"
            "Graph Traversal\n"
            "↓ GraphQueryBuilder\n"
            "SQL Query\n"
            "↓ MySQL direct\n"
            "DB Result\n"
            "↓ llama3.2:3b\n"
            "Final Answer"
        )

        st.divider()

        st.markdown("**Single Entity**")
        for q in [
            "What is Sarah Connor's salary?",
            "What is the most expensive product?",
            "How many Laptop Pro 15 units are in stock?",
        ]:
            if st.button(q, key=f"demo_{q}", use_container_width=True):
                st.session_state["pending_question"] = q

        st.markdown("**Two-Hop Join**")
        for q in [
            "List all employees in Engineering",
            "Who is the highest paid in Marketing?",
            "Show me all pending orders",
        ]:
            if st.button(q, key=f"demo_{q}", use_container_width=True):
                st.session_state["pending_question"] = q

        st.markdown("**Multi-Hop (Wow)**")
        for q in [
            "Which employees are assigned to the API Gateway Rebuild project?",
            "Who are the employees on projects managed by Don Draper?",
            "Which employees in Engineering are working on projects managed by Sarah?",
        ]:
            if st.button(q, key=f"demo_{q}", use_container_width=True):
                st.session_state["pending_question"] = q

        st.divider()

        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ============================================================
# GLASS BOX
# ============================================================

_METHOD_BADGE = {
    "single_node": ("⬤ single node", "#6c757d"),
    "two_hop":     ("⬤ 2-hop join",  "#0d6efd"),
    "multi_hop":   ("⬤ multi-hop",   "#198754"),
}

_EXTRACT_BADGE = {
    "rules": ("⬤ rules", "#fd7e14"),
    "llm":   ("⬤ LLM",   "#6f42c1"),
}


def _badge(label: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;'
        f'padding:2px 8px;border-radius:10px;font-size:0.75rem">'
        f'{label}</span>'
    )


def render_glass_box(trace: MiddlewareTrace) -> None:
    with st.expander("🔍 Audit Trail — How this answer was generated", expanded=False):

        # ── ⓪ Graph Traversal ────────────────────────────────
        if trace.graph_traversal:
            gt = trace.graph_traversal
            st.markdown("**⓪ Semantic Graph Traversal**")

            if gt.path_taken:
                path_html = " → ".join(
                    _badge(f"[{node}]", "#0d6efd") for node in gt.path_taken
                )
                method_label, method_color = _METHOD_BADGE.get(
                    gt.traversal_method, ("⬤ unknown", "#888")
                )
                st.markdown(
                    f"{path_html} &nbsp; {_badge(method_label, method_color)}",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("— no path (no entities detected) —")

            ct1, ct2, ct3 = st.columns(3)
            with ct1:
                st.caption("**Tables**")
                st.code(", ".join(gt.tables_involved) or "—", language=None)
            with ct2:
                st.caption("**Joins**")
                st.code(str(gt.join_count), language=None)
            with ct3:
                st.caption("**Graph time**")
                st.code(f"{gt.traversal_time_ms:.2f}ms", language=None)

            st.divider()

        # ── ① ② ③ ─────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**① Entity Extraction**")
            if trace.entity_extraction:
                ee = trace.entity_extraction
                for ent in ee.entities:
                    st.markdown(_badge(ent, "#0d6efd"), unsafe_allow_html=True)
                extract_label, extract_color = _EXTRACT_BADGE.get(
                    ee.extraction_method, ("⬤ unknown", "#888")
                )
                st.markdown(
                    f"Method: {_badge(extract_label, extract_color)}",
                    unsafe_allow_html=True,
                )
                st.caption(f"Type: `{ee.question_type}`")
                st.caption(f"Latency: `{ee.latency_ms:.0f}ms`")
            elif trace.intent:
                st.code(trace.intent.intent_name, language=None)
                st.caption(f"Model: `{trace.intent.model_used}`")
                st.caption(f"Latency: `{trace.intent.latency_ms:.0f}ms`")

        with col2:
            st.markdown("**② Filters Applied**")
            if trace.entity_extraction:
                ee = trace.entity_extraction
                if ee.filters:
                    st.json(ee.filters)
                else:
                    st.caption("— no filters —")
                if ee.projections:
                    st.caption(f"Projections: `{ee.projections}`")
            elif trace.parameters:
                st.json(trace.parameters.params)

        with col3:
            st.markdown("**③ Database Execution**")
            if trace.db_result:
                if trace.db_result.self_healing_triggered:
                    st.warning("Self-healing triggered")
                    st.caption(f"Reason: {trace.db_result.healing_reason}")
                else:
                    st.success(f"{trace.db_result.row_count} row(s) returned")
                st.caption(f"DB time: `{trace.db_result.execution_time_ms:.3f}ms`")

        st.divider()

        # ── ④ SQL ─────────────────────────────────────────────
        st.markdown("**④ SQL Executed Against MySQL**")
        if trace.db_result and trace.db_result.query_executed:
            st.code(trace.db_result.query_executed, language="sql")
        else:
            st.caption("— no query executed —")

        # ── ⑤ Raw Data ────────────────────────────────────────
        if (trace.db_result and trace.db_result.rows
                and not trace.db_result.self_healing_triggered):
            st.markdown("**⑤ Raw Database Response**")
            st.dataframe(
                pd.DataFrame(trace.db_result.rows),
                use_container_width=True,
                hide_index=True,
            )

        # ── ⑥ Timing ─────────────────────────────────────────
        st.divider()
        st.markdown("**⑥ Pipeline Timing Breakdown**")

        timings = []
        if trace.entity_extraction:
            timings.append(("Extraction", f"{trace.entity_extraction.latency_ms:.0f}ms",
                             trace.entity_extraction.extraction_method))
        elif trace.intent:
            timings.append(("Intent LLM", f"{trace.intent.latency_ms:.0f}ms", "qwen2.5:1.5b"))

        if trace.graph_traversal:
            timings.append(("Graph", f"{trace.graph_traversal.traversal_time_ms:.2f}ms", "NetworkX"))

        if trace.db_result:
            timings.append(("MySQL", f"{trace.db_result.execution_time_ms:.3f}ms", "direct"))

        if trace.total_latency_ms > 0:
            known = 0
            if trace.entity_extraction:
                known += trace.entity_extraction.latency_ms
            elif trace.intent:
                known += trace.intent.latency_ms
            if trace.graph_traversal:
                known += trace.graph_traversal.traversal_time_ms
            if trace.db_result:
                known += trace.db_result.execution_time_ms
            synth = trace.total_latency_ms - known
            if synth > 0:
                timings.append(("Synthesis LLM", f"{synth:.0f}ms", "llama3.2:3b"))

        timings.append(("TOTAL", f"{trace.total_latency_ms:.0f}ms", "wall clock"))

        timing_cols = st.columns(len(timings))
        for i, (label, value, note) in enumerate(timings):
            with timing_cols[i]:
                st.metric(label=f"{label}\n{note}", value=value)


# ============================================================
# HELPERS
# ============================================================

def _caption_for_trace(trace: MiddlewareTrace) -> str:
    ts = trace.timestamp.strftime("%H:%M:%S")
    gt = trace.graph_traversal
    method_str = gt.traversal_method if gt else "unknown"
    path_str   = "→".join(gt.path_taken) if gt else "—"
    heal_note  = " · ⚠ self-healing" if trace.self_healing_triggered else ""
    db_ms = trace.db_result.execution_time_ms if trace.db_result else 0
    return (
        f"🕐 {ts} · ⏱ {trace.total_latency_ms:.0f}ms total "
        f"· 🕸️ {path_str} · ⚡ {method_str} "
        f"· 🗄️ db: `{db_ms:.3f}ms`{heal_note}"
    )


def update_stats(trace: MiddlewareTrace) -> None:
    st.session_state.total_questions += 1
    if trace.self_healing_triggered:
        st.session_state.healing_count += 1
    if trace.db_result:
        st.session_state.db_latencies.append(trace.db_result.execution_time_ms)
    st.session_state.db_latencies = st.session_state.db_latencies[-10:]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("🕸️ Semantic Graph RAG Assistant")
    st.caption(
        "Ask questions about your company data — "
        "answers are pulled directly from MySQL via dynamic graph traversal."
    )
    st.divider()

    # ── Render chat history (all previous messages + their glass boxes) ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("trace"):
                st.caption(_caption_for_trace(msg["trace"]))
                render_glass_box(msg["trace"])

    if not st.session_state.messages:
        st.info(
            "👋 Welcome! Type a question below or pick a demo query from the sidebar.\n\n"
            "Try: *\"Which employees in Engineering are working on projects managed by Sarah?\"*"
        )

    # ── Handle demo sidebar button ────────────────────────────
    pending = st.session_state.pop("pending_question", None)

    # ── Chat input ────────────────────────────────────────────
    prompt = st.chat_input("Ask about employees, departments, products, orders, or projects…")
    question = prompt or pending

    if question and question.strip():
        q = question.strip()

        # Append + render user message
        st.session_state.messages.append({"role": "user", "content": q, "trace": None})
        with st.chat_message("user"):
            st.markdown(q)

        # Run pipeline + render answer
        with st.chat_message("assistant"):
            with st.spinner("Querying database via semantic graph…"):
                trace = run_pipeline(q)

            update_stats(trace)

            if trace.self_healing_triggered:
                st.warning(trace.final_answer)
            else:
                st.markdown(trace.final_answer)

            # Caption + glass box rendered here — no st.rerun() so they STAY visible
            st.caption(_caption_for_trace(trace))
            render_glass_box(trace)

        # Save to history
        # Next question will trigger a natural Streamlit re-render which
        # shows all history including this trace via the loop above.
        st.session_state.messages.append({
            "role": "assistant",
            "content": trace.final_answer,
            "trace": trace,
        })
        # NOTE: intentionally NO st.rerun() here — that was causing the
        # audit trail to flash and disappear.


if __name__ == "__main__":
    main()
