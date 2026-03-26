from __future__ import annotations

import logging
import yaml
import requests
import mysql.connector
import streamlit as st
import pandas as pd

from pathlib import Path
from datetime import datetime
from middleware.pipeline import run_pipeline, _graph
from middleware.models import MiddlewareTrace, ExtractionConfidence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Graph RAG Assistant",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "total_questions": 0,
        "healing_count": 0,
        "db_latencies": [],
        "layer_counts": {"pattern": 0, "spacy": 0, "llm": 0, "llm_failed": 0},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
        return False, str(e)[:60]


def check_ollama() -> tuple[bool, bool]:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        fast_ok = any("qwen2.5:1.5b" in m for m in models)
        synth_ok = any("qwen2.5:1.5b" in m for m in models)
        return fast_ok, synth_ok
    except Exception:
        return False, False


def check_graph() -> tuple[int, int]:
    try:
        return _graph.node_count, _graph.edge_count
    except Exception:
        return 0, 0


def render_sidebar() -> None:
    with st.sidebar:
        st.header("🕸️ Graph RAG Assistant")
        st.caption("MySQL · Semantic Graph · 3-Layer Extraction")
        st.caption("No Vector DB · No Fine-tuning · No Text-to-SQL")

        st.divider()

        st.markdown("**System Status**")
        mysql_ok, mysql_msg = check_mysql()
        fast_ok, synth_ok = check_ollama()
        nodes, edges = check_graph()

        st.caption(f"{'🟢' if mysql_ok else '🔴'} MySQL — {mysql_msg}")
        st.caption(f"{'🟢' if fast_ok else '🔴'} qwen2.5:1.5b (LLM fallback)")
        st.caption(f"{'🟢' if synth_ok else '🔴'} qwen2.5:1.5b (synthesis)")
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

        lc = st.session_state.layer_counts
        total_q = max(st.session_state.total_questions, 1)
        st.markdown("**Extraction Layer Usage**")
        st.caption(f"🟢 Pattern (0ms): {lc['pattern']} ({lc['pattern'] * 100 // total_q}%)")
        st.caption(f"🔵 spaCy (~5ms): {lc['spacy']} ({lc['spacy'] * 100 // total_q}%)")
        st.caption(f"🟣 LLM (~15s): {lc['llm']} ({lc['llm'] * 100 // total_q}%)")
        if lc["llm_failed"] > 0:
            st.caption(f"🔴 LLM Failed: {lc['llm_failed']}")

        st.divider()

        st.markdown("**Architecture**")
        st.caption(
            "Question\n"
            "↓ Layer 1: Pattern Match (0ms)\n"
            "↓ Layer 2: spaCy Parse (~5ms)\n"
            "↓ Layer 3: LLM Fallback (~15s)\n"
            "Confidence Scoring\n"
            "↓ NetworkX Graph Traversal\n"
            "↓ GraphQueryBuilder\n"
            "↓ MySQL (pooled)\n"
            "↓ Template Synthesis\n"
            "Final Answer"
        )
        st.divider()


_METHOD_BADGE = {
    "single_node": ("⬤ single node", "#6c757d"),
    "two_hop": ("⬤ 2-hop join", "#0d6efd"),
    "multi_hop": ("⬤ multi-hop", "#198754"),
}

_EXTRACT_BADGE = {
    "pattern": ("⬤ pattern", "#fd7e14"),
    "spacy": ("⬤ spaCy", "#20c997"),
    "llm": ("⬤ LLM", "#6f42c1"),
    "llm_failed": ("⬤ LLM failed", "#dc3545"),
}

_CONFIDENCE_COLOR = {
    "high": "#198754",
    "medium": "#fd7e14",
    "low": "#dc3545",
}


def _badge(label: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;'
        f'padding:2px 8px;border-radius:10px;font-size:0.75rem">'
        f'{label}</span>'
    )


def render_glass_box(trace: MiddlewareTrace) -> None:
    with st.expander("🔍 Audit Trail — How this answer was generated", expanded=False):

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

                conf_score = ee.confidence_score
                conf_label = (
                    "high" if conf_score >= 0.85
                    else "medium" if conf_score >= ExtractionConfidence.TRUST_THRESHOLD
                    else "low"
                )
                conf_color = _CONFIDENCE_COLOR.get(conf_label, "#888")
                st.markdown(
                    f"Confidence: {_badge(f'{conf_score:.0%}', conf_color)}",
                    unsafe_allow_html=True,
                )
                st.caption(f"Latency: `{ee.latency_ms:.0f}ms`")

                if ee.escalation_reason:
                    st.warning(f"Escalation: {ee.escalation_reason}")

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
                if ee.confidence_breakdown:
                    with st.expander("Confidence breakdown"):
                        for k, v in ee.confidence_breakdown.items():
                            st.caption(f"{k}: {v:.2f}")

        with col3:
            st.markdown("**③ Database Execution**")
            if trace.db_result:
                if trace.db_result.self_healing_triggered:
                    st.warning("No results / fallback triggered")
                    st.caption(f"Reason: {trace.db_result.healing_reason}")
                else:
                    st.success(f"{trace.db_result.row_count} row(s) returned")
                st.caption(f"DB time: `{trace.db_result.execution_time_ms:.3f}ms`")

        st.divider()

        st.markdown("**④ SQL Executed Against MySQL**")
        if trace.db_result and trace.db_result.query_executed:
            st.code(trace.db_result.query_executed, language="sql")
        else:
            st.caption("— no query executed —")

        if (trace.db_result and trace.db_result.rows
                and not trace.db_result.self_healing_triggered):
            st.markdown("**⑤ Raw Database Response**")
            st.dataframe(
                pd.DataFrame(trace.db_result.rows),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        st.markdown("**⑥ Pipeline Timing Breakdown**")

        timings = []
        if trace.entity_extraction:
            timings.append((
                "Extraction",
                f"{trace.entity_extraction.latency_ms:.0f}ms",
                trace.entity_extraction.extraction_method,
            ))

        if trace.graph_traversal:
            timings.append((
                "Graph",
                f"{trace.graph_traversal.traversal_time_ms:.2f}ms",
                "NetworkX",
            ))

        if trace.db_result:
            timings.append((
                "MySQL",
                f"{trace.db_result.execution_time_ms:.3f}ms",
                "pooled",
            ))

        if trace.total_latency_ms > 0:
            known = (
                (trace.entity_extraction.latency_ms if trace.entity_extraction else 0)
                + (trace.graph_traversal.traversal_time_ms if trace.graph_traversal else 0)
                + (trace.db_result.execution_time_ms if trace.db_result else 0)
            )
            synth = trace.total_latency_ms - known
            if synth > 50:
                timings.append(("Synthesis", f"{synth:.0f}ms", "template/LLM"))

        timings.append(("TOTAL", f"{trace.total_latency_ms:.0f}ms", "wall clock"))

        timing_cols = st.columns(len(timings))
        for i, (label, value, note) in enumerate(timings):
            with timing_cols[i]:
                st.metric(label=f"{label}\n{note}", value=value)


def _caption_for_trace(trace: MiddlewareTrace) -> str:
    ts = trace.timestamp.strftime("%H:%M:%S")
    gt = trace.graph_traversal
    method_str = gt.traversal_method if gt else "unknown"
    path_str = "→".join(gt.path_taken) if gt else "—"
    heal_note = " · ⚠ fallback" if trace.self_healing_triggered else ""
    db_ms = trace.db_result.execution_time_ms if trace.db_result else 0
    ee = trace.entity_extraction
    method_note = f"[{ee.extraction_method}]" if ee else ""
    return (
        f"🕐 {ts} · ⏱ {trace.total_latency_ms:.0f}ms total "
        f"· 🕸️ {path_str} · ⚡ {method_str} "
        f"· 🗄️ db: `{db_ms:.3f}ms` · {method_note}{heal_note}"
    )


def update_stats(trace: MiddlewareTrace) -> None:
    st.session_state.total_questions += 1
    if trace.self_healing_triggered:
        st.session_state.healing_count += 1
    if trace.db_result:
        st.session_state.db_latencies.append(trace.db_result.execution_time_ms)
    st.session_state.db_latencies = st.session_state.db_latencies[-20:]

    if trace.entity_extraction:
        method = trace.entity_extraction.extraction_method
        lc = st.session_state.layer_counts
        if method in lc:
            lc[method] += 1
        elif "llm" in method:
            lc["llm_failed"] += 1


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("🕸️ Graph-based RAG Assistant")
    st.caption(
        "Ask questions about company data — "
        "answers pulled directly from MySQL via semantic graph traversal."
    )
    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("trace"):
                st.caption(_caption_for_trace(msg["trace"]))
                render_glass_box(msg["trace"])

    if not st.session_state.messages:
        st.info(
            "👋 Welcome! Ask a question about the company database.\n\n"
        )

    prompt = st.chat_input("Ask about employees, departments, products, orders, or projects…")

    if prompt and prompt.strip():
        q = prompt.strip()

        st.session_state.messages.append({"role": "user", "content": q, "trace": None})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("Querying database via semantic graph…"):
                trace = run_pipeline(q)

            update_stats(trace)

            if trace.self_healing_triggered:
                st.warning(trace.final_answer)
            else:
                st.markdown(trace.final_answer)

            st.caption(_caption_for_trace(trace))
            render_glass_box(trace)

        st.session_state.messages.append({
            "role": "assistant",
            "content": trace.final_answer,
            "trace": trace,
        })


if __name__ == "__main__":
    main()
