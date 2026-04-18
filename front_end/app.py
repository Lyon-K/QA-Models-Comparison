from pathlib import Path
import sys
from html import escape

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend import get_fact_check_result as fetch_backend_fact_check_result


PAGE_TITLE = "Public Health Misinformation Fact-Checking"
PAGE_BROWSER_TITLE = "Public Health Fact-Checking"
PAGE_DESCRIPTION = "Run the same query against the final QA model lineup."
TOP_BANNER_LINES = [
    "Compare T5, noRAG, RAG, graphRAG, and hybridRAG side by side.",
    "If a model cannot load, its card will show a clean demo fallback message.",
]
EXAMPLE_CLAIMS = [
    "Vaccines cause infertility",
    "Masks reduce oxygen intake",
    "Antibiotics treat viral infections",
]
MODEL_DISPLAY_ORDER = ["T5", "noRAG", "RAG", "graphRAG", "hybridRAG"]
MODEL_DESCRIPTIONS = {
    "T5": "Generative baseline model",
    "noRAG": "Direct LLM baseline without retrieval",
    "RAG": "Retrieval-Augmented Generation",
    "graphRAG": "Graph-based retrieval model",
    "hybridRAG": "Hybrid retrieval-augmented model",
}
MODEL_LABELS = {
    "T5": "Baseline",
    "noRAG": "LLM only",
    "RAG": "Retrieval-based",
    "graphRAG": "Graph-based retrieval",
    "hybridRAG": "Hybrid retrieval",
}
UNAVAILABLE_PREFIX = "This model is not enabled in the current demo environment"
UNAVAILABLE_MESSAGE = "This model is not enabled in the current demo environment"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1360px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .demo-hero {
            background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
            border: 1px solid #d7e3f1;
            border-radius: 22px;
            padding: 1.35rem 1.4rem 1.15rem 1.4rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.1rem;
        }
        .demo-subtitle {
            color: #475569;
            font-size: 0.98rem;
            line-height: 1.6;
            margin-top: 0.25rem;
        }
        .section-kicker {
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.4rem;
            font-weight: 700;
        }
        .input-shell {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 1rem 1rem 0.6rem 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
            margin-bottom: 0.75rem;
        }
        .stTextInput > div > div > input {
            border-radius: 14px !important;
            border: 1px solid #cbd5e1 !important;
            padding-top: 0.75rem !important;
            padding-bottom: 0.75rem !important;
        }
        .stButton > button {
            border-radius: 14px !important;
            border: 1px solid #cbd5e1 !important;
            padding: 0.72rem 1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0f766e 0%, #0f5f78 100%) !important;
            border: none !important;
            color: white !important;
            box-shadow: 0 10px 24px rgba(15, 118, 110, 0.22) !important;
        }
        .model-card {
            background: #ffffff;
            border: 1px solid #dbe4ee;
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            min-height: 420px;
        }
        .model-card.graph-card {
            background: linear-gradient(180deg, #f6fbff 0%, #ffffff 100%);
            border: 1px solid #bfd9ee;
            box-shadow: 0 10px 28px rgba(59, 130, 246, 0.08);
        }
        .model-card-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.7rem;
        }
        .model-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
        }
        .model-badge {
            display: inline-block;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            padding: 0.18rem 0.58rem;
            font-size: 0.73rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .model-card.graph-card .model-badge {
            background: #dff2ff;
            color: #0f4c81;
            border-color: #b9dcf8;
        }
        .model-description {
            color: #64748b;
            font-size: 0.88rem;
            margin-bottom: 0.9rem;
            min-height: 2.4em;
        }
        .card-section-title {
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #64748b;
            margin: 0.85rem 0 0.42rem 0;
            font-weight: 700;
        }
        .card-summary {
            color: #1f2937;
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 0;
        }
        .card-list {
            margin: 0;
            padding-left: 1.1rem;
            color: #1f2937;
        }
        .card-list li {
            margin-bottom: 0.35rem;
            line-height: 1.5;
        }
        .graph-evidence {
            background: #f3f8fd;
            border: 1px solid #d7e8f8;
            border-radius: 14px;
            padding: 0.72rem 0.8rem;
            margin-top: 0.7rem;
        }
        .graph-evidence-line {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.83rem;
            line-height: 1.55;
            color: #0f3e63;
            margin-bottom: 0.32rem;
        }
        .graph-evidence-line:last-child {
            margin-bottom: 0;
        }
        .takeaways-panel {
            background: linear-gradient(180deg, #fbfcfe 0%, #ffffff 100%);
            border: 1px solid #dde6ef;
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
        }
        .takeaways-list {
            margin: 0.35rem 0 0 0;
            padding-left: 1.1rem;
        }
        .takeaways-list li {
            margin-bottom: 0.45rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_page() -> None:
    st.set_page_config(
        page_title=PAGE_BROWSER_TITLE,
        page_icon=":material/medical_services:",
        layout="wide",
    )


def initialize_session_state() -> None:
    st.session_state.setdefault("claim_input", "")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("checked_claim", "")


def reset_app_state() -> None:
    st.session_state.claim_input = ""
    st.session_state.result = None
    st.session_state.checked_claim = ""


def get_fact_check_result(claim: str) -> dict[str, str]:
    return fetch_backend_fact_check_result(claim.strip())


def render_header() -> None:
    st.markdown(
        f"""
        <div class="demo-hero">
            <div class="section-kicker">QA Comparison Demo</div>
            <h1 style="margin:0;color:#0f172a;">{escape(PAGE_TITLE)}</h1>
            <div class="demo-subtitle">
                This demo compares different QA approaches for public health questions.
                Run one query across baseline, retrieval, hybrid retrieval, and graph-based reasoning models.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(PAGE_DESCRIPTION)
    with st.container():
        st.info("\n".join(TOP_BANNER_LINES))


def render_input_panel() -> None:
    st.markdown('<div class="input-shell">', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Try An Example</div>', unsafe_allow_html=True)
    chip_columns = st.columns(3, gap="small")

    for column, example_claim in zip(chip_columns, EXAMPLE_CLAIMS):
        with column:
            if st.button(example_claim, use_container_width=True):
                st.session_state.claim_input = example_claim

    st.text_input(
        "Enter a claim or question",
        placeholder="e.g. Vaccines cause infertility",
        key="claim_input",
    )

    action_col, clear_col = st.columns([3, 1], gap="small")

    with action_col:
        run_clicked = st.button("Run All Models", type="primary", use_container_width=True)

    with clear_col:
        clear_clicked = st.button("Clear", use_container_width=True)

    if run_clicked:
        if not st.session_state.claim_input.strip():
            st.warning("Please enter a claim or question.")
        else:
            with st.spinner("Running model inference..."):
                st.session_state.result = get_fact_check_result(st.session_state.claim_input)
            st.session_state.checked_claim = st.session_state.claim_input.strip()

    if clear_clicked:
        reset_app_state()

    st.markdown("</div>", unsafe_allow_html=True)


def _parse_structured_output(output: str) -> tuple[str, list[str], list[str]]:
    text = str(output or "").strip()
    if not text:
        return "", [], []

    lines = [line.rstrip() for line in text.splitlines()]
    summary_lines: list[str] = []
    bullet_lines: list[str] = []
    graph_evidence_lines: list[str] = []
    mode = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        lower_line = line.lower()
        if lower_line == "summary:":
            mode = "summary"
            continue
        if lower_line == "key points:":
            mode = "bullets"
            continue
        if lower_line == "graph evidence:":
            mode = "graph_evidence"
            continue
        if lower_line == "additional info:":
            mode = "additional"
            continue

        if mode == "bullets" and line.startswith("-"):
            bullet_lines.append(line[1:].strip())
        elif mode == "graph_evidence" and line.startswith("-"):
            graph_evidence_lines.append(line[1:].strip())
        elif mode == "summary":
            summary_lines.append(line)
        elif mode == "additional":
            continue

    summary = " ".join(summary_lines).strip()
    return summary, bullet_lines[:4], graph_evidence_lines[:4]


def _render_html_list(items: list[str]) -> str:
    return "".join(f"<li>{escape(item)}</li>" for item in items if item)


def _render_graph_evidence(items: list[str]) -> str:
    if not items:
        return ""
    lines = "".join(
        f'<div class="graph-evidence-line">{escape(item)}</div>' for item in items if item
    )
    return f"""
    <div class="card-section-title">Graph Evidence</div>
    <div class="graph-evidence">{lines}</div>
    """


def render_model_output(title: str, output: str) -> None:
    description = MODEL_DESCRIPTIONS.get(title, "")
    label = MODEL_LABELS.get(title, "")
    normalized_output = str(output).strip()
    is_unavailable = normalized_output.startswith(UNAVAILABLE_PREFIX)
    summary, bullet_points, graph_evidence = _parse_structured_output(normalized_output)
    card_class = "model-card graph-card" if title == "graphRAG" else "model-card"

    if is_unavailable:
        summary = UNAVAILABLE_MESSAGE
        bullet_points = []
        graph_evidence = []

    fallback_text = normalized_output if normalized_output else "No output returned."
    if not summary:
        summary = fallback_text

    bullet_html = _render_html_list(bullet_points)
    graph_html = _render_graph_evidence(graph_evidence) if title == "graphRAG" else ""

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="model-card-header">
                <div>
                    <div class="model-title">{escape(title)}</div>
                </div>
                <div class="model-badge">{escape(label)}</div>
            </div>
            <div class="model-description">{escape(description)}</div>
            <div class="card-section-title">Summary</div>
            <p class="card-summary">{escape(summary)}</p>
            <div class="card-section-title">Key Points</div>
            {"<ul class='card-list'>" + bullet_html + "</ul>" if bullet_points else "<ul class='card-list'><li>No key points available.</li></ul>"}
            {graph_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results() -> None:
    result = st.session_state.result

    st.markdown("## Model Outputs")
    if not result:
        st.info("Model outputs will appear here after you run a query.")
        return

    st.caption(f"Query: {st.session_state.checked_claim}")
    top_row = st.columns(3, gap="large")
    bottom_row = st.columns(2, gap="large")
    columns = [*top_row, *bottom_row]

    for column, model_name in zip(columns, MODEL_DISPLAY_ORDER):
        with column:
            render_model_output(model_name, result.get(model_name, "No output returned."))


def _model_status_text(result: dict, model_name: str, available_text: str, unavailable_text: str) -> str:
    output = str(result.get(model_name, "")).strip()
    if not output or output.startswith(UNAVAILABLE_PREFIX):
        return unavailable_text
    return available_text


def _classify_model_output(output: str) -> str:
    text = str(output).strip()
    if not text or text.startswith(UNAVAILABLE_PREFIX):
        return "unavailable"
    lower_text = text.lower()
    if "summary:" in lower_text and "key points:" in lower_text:
        if "retrieved context" in lower_text or "supporting context" in lower_text:
            return "grounded_structured"
        if len(text) > 650:
            return "long_structured"
        return "structured"
    if len(text) < 220:
        return "short"
    if "retrieved" in lower_text or "context" in lower_text:
        return "grounded"
    return "long"


def _compare_model_line(model_name: str, output: str) -> str:
    status = _classify_model_output(output)
    text = str(output or "").strip()
    lower_text = text.lower()

    if status == "unavailable":
        return f"{model_name}: not available in this environment."

    if model_name == "T5":
        if status in {"short", "structured"}:
            return "T5: the most concise baseline response, with a short explanation and fewer details."
        return "T5: a compact baseline answer that stays simpler than the retrieval-based models."

    if model_name == "noRAG":
        if status in {"long_structured", "long"}:
            return "noRAG: a longer LLM-only answer, usually more detailed but not explicitly retrieval-grounded."
        return "noRAG: an LLM-only answer that is readable but less grounded in retrieved evidence."

    if model_name == "RAG":
        if status in {"grounded_structured", "grounded", "long_structured"}:
            return "RAG: a retrieval-grounded answer that appears tied to supporting context."
        return "RAG: a retrieval-based answer that is generally more evidence-oriented than the baselines."

    if model_name == "hybridRAG":
        if status in {"grounded_structured", "long_structured"}:
            return "hybridRAG: one of the more detailed outputs, combining broader retrieval coverage with a polished answer."
        return "hybridRAG: broader retrieval coverage, though its answer can be less tightly focused than standard RAG."

    if model_name == "graphRAG":
        if "graph evidence:" in lower_text:
            return "graphRAG: graph-based retrieval with explicit relation-style evidence shown in the card."
        if "score:" in lower_text or "relationship" in lower_text or "context(" in lower_text:
            return "graphRAG: more relationship-driven, reflecting graph-linked context rather than only free-text retrieval."
        if status in {"structured", "grounded_structured", "grounded"}:
            return "graphRAG: graph-based retrieval that appears more connection-oriented than the other retrieval models."
        return "graphRAG: graph-based retrieval with a more general final answer style."

    return f"{model_name}: available."


def render_model_comparison_summary() -> None:
    st.markdown("## Model Comparison Summary")

    result = st.session_state.result
    if not result:
        st.info("A short comparison summary will appear here after you run a query.")
        return

    summary_lines = [
        _compare_model_line("T5", result.get("T5", "")),
        _compare_model_line("noRAG", result.get("noRAG", "")),
        _compare_model_line("RAG", result.get("RAG", "")),
        _compare_model_line("hybridRAG", result.get("hybridRAG", "")),
        _compare_model_line("graphRAG", result.get("graphRAG", "")),
    ]

    summary_html = _render_html_list(summary_lines)
    st.markdown(
        f"""
        <div class="takeaways-panel">
            <div class="section-kicker">Key Takeaways</div>
            <div style="color:#475569;font-size:0.96rem;">Quick presentation summary of the currently returned model outputs.</div>
            <ul class="takeaways-list">{summary_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.caption(
        "This demo is for educational purposes only and does not provide medical advice."
    )


def main() -> None:
    configure_page()
    inject_styles()
    initialize_session_state()
    render_header()
    render_input_panel()
    st.divider()
    render_results()
    st.divider()
    render_model_comparison_summary()
    st.divider()
    render_footer()


if __name__ == "__main__":
    main()
