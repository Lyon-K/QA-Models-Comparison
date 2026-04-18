from pathlib import Path
import sys

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
UNAVAILABLE_PREFIX = "This model is not enabled in the current demo environment"
UNAVAILABLE_MESSAGE = "This model is not enabled in the current demo environment"


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
    st.title(PAGE_TITLE)
    st.caption("This demo compares different QA approaches for public health questions.")
    st.write(PAGE_DESCRIPTION)
    with st.container():
        st.info("\n".join(TOP_BANNER_LINES))


def render_input_panel() -> None:
    st.markdown("Try an example:")
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


def render_model_output(title: str, output: str) -> None:
    description = MODEL_DESCRIPTIONS.get(title, "")
    normalized_output = str(output).strip()
    is_unavailable = normalized_output.startswith(UNAVAILABLE_PREFIX)

    with st.container(border=True):
        st.markdown(f"### {title}")
        if description:
            st.caption(description)

        if is_unavailable:
            st.write(UNAVAILABLE_MESSAGE)
            return

        if title == "T5":
            st.markdown("**Explanation**")
            st.write(normalized_output if normalized_output else "No output returned.")
            return

        st.write(normalized_output if normalized_output else "No output returned.")


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


def render_footer() -> None:
    st.caption(
        "This demo is for educational purposes only and does not provide medical advice."
    )


def main() -> None:
    configure_page()
    initialize_session_state()
    render_header()
    render_input_panel()
    st.divider()
    render_results()
    st.divider()
    render_footer()


if __name__ == "__main__":
    main()
