import streamlit as st


# -------------------------------------------------------------------
# 1. Page config
# -------------------------------------------------------------------
PAGE_TITLE = "Public Health Misinformation Fact-Checking"
PAGE_BROWSER_TITLE = "Public Health Fact-Checking"
PAGE_DESCRIPTION = (
    "Check whether a public health claim is supported by trusted evidence "
    "from public health sources."
)
TOP_BANNER_LINES = [
    "AI-powered fact-checking for public health claims.",
    "Get a clear verdict, explanation, and evidence from trusted sources.",
]


def configure_page() -> None:
    """Configure Streamlit page metadata."""
    st.set_page_config(
        page_title=PAGE_BROWSER_TITLE,
        page_icon=":material/medical_services:",
        layout="wide",
    )


# -------------------------------------------------------------------
# 2. Constants / mock data
# -------------------------------------------------------------------
EXAMPLE_CLAIMS = [
    "Vaccines cause infertility",
    "Masks reduce oxygen intake",
    "Antibiotics treat viral infections",
]
SOURCE_FILTERS = ["All sources", "WHO only", "CDC only"]


# Mock data used for frontend demo; replace with backend API response later.
FAKE_RESULTS = {
    "Vaccines cause infertility": {
        "claim": "Vaccines cause infertility",
        "verdict": "False",
        "summary": "There is no credible scientific evidence supporting this claim.",
        "what_this_means": "You can safely ignore this claim - it is not supported by credible evidence.",
        "trusted_sources": ["WHO", "CDC"],
        "key_insights": [
            "No infertility link found",
            "Large studies show vaccine safety",
            "Claim driven by misinformation",
        ],
        "explanation": (
            "There is no credible evidence that approved vaccines cause infertility. "
            "Public health agencies and large clinical studies have found no link between "
            "vaccination and reduced fertility in women or men. This claim often stems "
            "from misinformation that misrepresents immune responses or reproductive biology."
        ),
        "evidence_summary": [
            "Consensus: All sources refute this claim",
            "Sources: WHO (2), CDC (1)",
        ],
        "evidence": [
            {
                "source": "WHO",
                "stance": "Refutes claim",
                "title": "Vaccine Safety Basics",
                "text": (
                    "The WHO explains that vaccines used in immunization programs are carefully "
                    "evaluated for safety and there is no evidence that they cause infertility."
                ),
                "url": "https://www.who.int/news-room/questions-and-answers/item/vaccines-and-immunization-vaccine-safety",
            },
            {
                "source": "CDC",
                "stance": "Refutes claim",
                "title": "COVID-19 Vaccines and Fertility",
                "text": (
                    "The CDC states that there is currently no evidence showing that any vaccines, "
                    "including COVID-19 vaccines, cause fertility problems."
                ),
                "url": "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/planning-for-pregnancy.html",
            },
            {
                "source": "WHO",
                "stance": "Refutes claim",
                "title": "Myths and Misconceptions about Immunization",
                "text": (
                    "WHO myth-busting materials describe infertility claims as unsupported and "
                    "inconsistent with available clinical and population-level evidence."
                ),
                "url": "https://www.who.int/",
            },
        ],
        "final_verdict": "FALSE",
        "agreement_summary": "3 of 4 models refute the claim",
        "models": [
            {
                "model_name": "Seq2Seq",
                "model_type": "closed_book",
                "verdict": "FALSE",
                "summary": "This model classifies the infertility claim as false based on learned patterns.",
                "has_evidence": False,
                "evidence_sources": [],
            },
            {
                # updated model names to match final project setup
                "model_name": "RAG",
                "model_type": "rag",
                "verdict": "FALSE",
                "summary": "WHO and CDC evidence do not support a link between vaccines and infertility.",
                "has_evidence": True,
                "evidence_sources": ["WHO", "CDC"],
            },
            {
                "model_name": "Graph RAG",
                "model_type": "graph_rag",
                "verdict": "FALSE",
                "summary": "Retrieved public health guidance refutes the infertility claim.",
                "has_evidence": True,
                "evidence_sources": ["WHO"],
            },
            {
                "model_name": "Hybrid RAG",
                "model_type": "hybrid_rag",
                "verdict": "TRUE",
                "summary": "This model interpreted one retrieved passage incorrectly as supportive.",
                "has_evidence": True,
                "evidence_sources": ["WHO"],
            },
        ],
        "seq2seq_answer": "This claim is false. Vaccines are generally considered safe.",
        "rag_answer": (
            "This claim is false. WHO and CDC materials report that approved vaccines have "
            "not been shown to cause infertility."
        ),
    },
    "Masks reduce oxygen intake": {
        "claim": "Masks reduce oxygen intake",
        "verdict": "False",
        "summary": "There is no credible scientific evidence supporting this claim.",
        "what_this_means": "You can safely ignore this claim - it is not supported by credible evidence.",
        "trusted_sources": ["WHO", "CDC"],
        "key_insights": [
            "No oxygen reduction in normal use",
            "Breathable for healthy individuals",
            "No clinical evidence of harm",
        ],
        "explanation": (
            "Standard face masks are designed to be breathable while blocking respiratory droplets. "
            "Studies and public health guidance show that masks do not meaningfully reduce oxygen "
            "levels for the general public during normal use."
        ),
        "evidence_summary": [
            "Consensus: All sources refute this claim",
            "Sources: WHO (2), CDC (1)",
        ],
        "evidence": [
            {
                "source": "WHO",
                "stance": "Refutes claim",
                "title": "Mask Use in the Context of COVID-19",
                "text": (
                    "WHO guidance indicates that prolonged use of medical masks by healthy people "
                    "does not cause carbon dioxide intoxication or oxygen deficiency."
                ),
                "url": "https://www.who.int/",
            },
            {
                "source": "CDC",
                "stance": "Refutes claim",
                "title": "About Masks",
                "text": (
                    "The CDC explains that masks allow normal airflow while helping reduce the "
                    "spread of infectious particles."
                ),
                "url": "https://www.cdc.gov/",
            },
            {
                "source": "WHO",
                "stance": "Refutes claim",
                "title": "Coronavirus Disease Advice for the Public",
                "text": (
                    "WHO materials note that masks can be worn safely and should not cause oxygen "
                    "deprivation in routine community settings."
                ),
                "url": "https://www.who.int/",
            },
        ],
        "final_verdict": "FALSE",
        "agreement_summary": "4 of 4 models refute the claim",
        "models": [
            {
                "model_name": "Seq2Seq",
                "model_type": "closed_book",
                "verdict": "FALSE",
                "summary": "This model classifies the oxygen intake claim as false.",
                "has_evidence": False,
                "evidence_sources": [],
            },
            {
                "model_name": "RAG",
                "model_type": "rag",
                "verdict": "FALSE",
                "summary": "WHO guidance states that masks do not cause oxygen deficiency.",
                "has_evidence": True,
                "evidence_sources": ["WHO"],
            },
            {
                "model_name": "Graph RAG",
                "model_type": "graph_rag",
                "verdict": "FALSE",
                "summary": "CDC evidence indicates masks allow normal airflow during routine use.",
                "has_evidence": True,
                "evidence_sources": ["CDC"],
            },
            {
                "model_name": "Hybrid RAG",
                "model_type": "hybrid_rag",
                "verdict": "FALSE",
                "summary": "WHO and CDC passages consistently refute oxygen deprivation concerns.",
                "has_evidence": True,
                "evidence_sources": ["WHO", "CDC"],
            },
        ],
        "seq2seq_answer": "This claim is false. Masks do not significantly reduce oxygen intake.",
        "rag_answer": (
            "This claim is false. WHO and CDC guidance indicates masks remain breathable "
            "and do not cause meaningful oxygen deprivation for most people."
        ),
    },
    "Antibiotics treat viral infections": {
        "claim": "Antibiotics treat viral infections",
        "verdict": "False",
        "summary": "There is no credible scientific evidence supporting this claim.",
        "what_this_means": "You can safely ignore this claim - it is not supported by credible evidence.",
        "trusted_sources": ["CDC", "WHO"],
        "key_insights": [
            "Antibiotics do not treat viruses",
            "No recovery benefit for viral illness",
            "Misuse raises resistance risk",
        ],
        "explanation": (
            "Antibiotics are used to treat bacterial infections, not viral infections. "
            "Using antibiotics when they are not needed does not help patients recover from "
            "viruses and can contribute to antimicrobial resistance."
        ),
        "evidence_summary": [
            "Consensus: All sources refute this claim",
            "Sources: CDC (2), WHO (1)",
        ],
        "evidence": [
            {
                "source": "CDC",
                "stance": "Refutes claim",
                "title": "Antibiotic Use and Antimicrobial Resistance Facts",
                "text": "The CDC states that antibiotics do not work on viruses such as colds and flu.",
                "url": "https://www.cdc.gov/antibiotic-use/",
            },
            {
                "source": "WHO",
                "stance": "Refutes claim",
                "title": "Antimicrobial Resistance",
                "text": (
                    "WHO explains that misuse and overuse of antibiotics accelerate antimicrobial "
                    "resistance, including taking antibiotics for viral infections."
                ),
                "url": "https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance",
            },
            {
                "source": "CDC",
                "stance": "Refutes claim",
                "title": "Be Antibiotics Aware",
                "text": (
                    "CDC campaign materials emphasize that antibiotics are not effective against "
                    "viruses and should only be used when medically appropriate."
                ),
                "url": "https://www.cdc.gov/antibiotic-use/week/",
            },
        ],
        "final_verdict": "FALSE",
        "agreement_summary": "4 of 4 models refute the claim",
        "models": [
            {
                "model_name": "Seq2Seq",
                "model_type": "closed_book",
                "verdict": "FALSE",
                "summary": "This model classifies the antibiotic claim as false.",
                "has_evidence": False,
                "evidence_sources": [],
            },
            {
                "model_name": "RAG",
                "model_type": "rag",
                "verdict": "FALSE",
                "summary": "CDC evidence states antibiotics do not work on viruses.",
                "has_evidence": True,
                "evidence_sources": ["CDC"],
            },
            {
                "model_name": "Graph RAG",
                "model_type": "graph_rag",
                "verdict": "FALSE",
                "summary": "WHO guidance identifies antibiotic use for viral infections as misuse.",
                "has_evidence": True,
                "evidence_sources": ["WHO"],
            },
            {
                "model_name": "Hybrid RAG",
                "model_type": "hybrid_rag",
                "verdict": "FALSE",
                "summary": "Retrieved CDC and WHO sources consistently refute the claim.",
                "has_evidence": True,
                "evidence_sources": ["CDC", "WHO"],
            },
        ],
        "seq2seq_answer": "This claim is false. Antibiotics are not effective against viral infections.",
        "rag_answer": (
            "This claim is false. CDC and WHO sources explain that antibiotics target bacteria, "
            "not viruses, and misuse can worsen antimicrobial resistance."
        ),
    },
}


# -------------------------------------------------------------------
# 3. Session state init
# -------------------------------------------------------------------
def initialize_session_state() -> None:
    """Initialize Streamlit session state keys used by the app."""
    st.session_state.setdefault("claim_input", "")
    st.session_state.setdefault("source_filter", SOURCE_FILTERS[0])
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("checked_claim", "")


# -------------------------------------------------------------------
# 4. Helper functions
# -------------------------------------------------------------------
def get_verdict_summary(verdict: str) -> tuple[str, str]:
    """Return a display title and default summary for a verdict."""
    verdict_map = {
        "False": (
            "This claim is FALSE",
            "There is no credible scientific evidence supporting this claim.",
        ),
        "True": (
            "This claim is TRUE",
            "This claim is supported by credible public health evidence.",
        ),
        "Uncertain": (
            "This claim is UNCERTAIN",
            "There is not enough credible evidence in the current knowledge base.",
        ),
    }
    return verdict_map.get(verdict, verdict_map["Uncertain"])


def get_verdict_icon(verdict: str) -> str:
    """Return a small visual marker for the verdict."""
    return {"False": "X", "True": "OK", "Uncertain": "!"}.get(verdict, "!")



def get_user_decision_message(verdict: str) -> str:
    """Return user-facing guidance for a verdict."""
    messages = {
        "False": "You can safely ignore this claim - it is not supported by credible evidence.",
        "True": (
            "This claim appears reliable based on trusted evidence, but you should still "
            "follow official public health guidance."
        ),
        "Uncertain": (
            "Treat this claim cautiously - there is not enough credible evidence here to rely on it."
        ),
    }
    return messages.get(verdict, messages["Uncertain"])


def get_trusted_sources(evidence: list[dict]) -> list[str]:
    """Return unique evidence sources in display order."""
    sources = []
    for item in evidence:
        source = item.get("source")
        if source and source not in sources:
            sources.append(source)
    return sources


def get_trusted_sources_label(evidence: list[dict]) -> str:
    """Build a compact trusted-source label."""
    sources = get_trusted_sources(evidence)
    if not sources:
        return "Trusted sources used: Demo knowledge base"
    return f"Trusted sources used: {' / '.join(sources)}"


def filter_evidence_by_source(evidence: list[dict], source_filter: str) -> list[dict]:
    """Filter evidence items based on the selected source filter."""
    if source_filter == "All sources":
        return evidence

    source_name = source_filter.replace(" only", "")
    return [item for item in evidence if item.get("source") == source_name]


def build_evidence_summary(evidence: list[dict]) -> list[str]:
    """Build a fallback evidence summary from evidence items."""
    if not evidence:
        return ["Consensus: No supporting sources found", "Sources: None in current demo"]

    source_counts: dict[str, int] = {}
    stance_counts: dict[str, int] = {}
    for item in evidence:
        source = item.get("source", "Unknown")
        stance = item.get("stance", "Unspecified")
        source_counts[source] = source_counts.get(source, 0) + 1
        stance_counts[stance] = stance_counts.get(stance, 0) + 1

    source_line = ", ".join(
        f"{source} ({count})" for source, count in source_counts.items()
    )

    if len(stance_counts) == 1:
        stance = next(iter(stance_counts))
        if stance == "Refutes claim":
            consensus = "Consensus: All sources refute this claim"
        elif stance == "Supports claim":
            consensus = "Consensus: All sources support this claim"
        else:
            consensus = f"Consensus: {stance}"
    else:
        consensus = "Consensus: Sources show mixed positions"

    return [consensus, f"Sources: {source_line}"]


def build_default_models(verdict: str) -> list[dict]:
    """Return fallback model rows when model-level output is unavailable."""
    final_verdict = verdict.upper()
    return [
        {
            "model_name": "Seq2Seq",
            "model_type": "closed_book",
            "verdict": final_verdict,
            "summary": "No model-specific summary is available for this claim.",
            "has_evidence": False,
            "evidence_sources": [],
        },
        {
            "model_name": "RAG Model 1",
            "model_type": "rag",
            "verdict": final_verdict,
            "summary": "No retrieved evidence is available for this claim.",
            "has_evidence": False,
            "evidence_sources": [],
        },
        {
            "model_name": "RAG Model 2",
            "model_type": "rag",
            "verdict": final_verdict,
            "summary": "No retrieved evidence is available for this claim.",
            "has_evidence": False,
            "evidence_sources": [],
        },
        {
            "model_name": "RAG Model 3",
            "model_type": "rag",
            "verdict": final_verdict,
            "summary": "No retrieved evidence is available for this claim.",
            "has_evidence": False,
            "evidence_sources": [],
        },
    ]


def build_default_result(claim: str) -> dict:
    """Return a fallback result when the claim is not found in mock data."""
    verdict = "Uncertain"
    _, summary = get_verdict_summary(verdict)
    return {
        "claim": claim,
        "verdict": verdict,
        "summary": summary,
        "what_this_means": get_user_decision_message(verdict),
        "trusted_sources": [],
        "key_insights": [
            "Evidence is currently insufficient",
            "More trusted sources are needed",
        ],
        "explanation": "No matching result in demo database.",
        "evidence_summary": ["Consensus: No supporting sources found", "Sources: None in current demo"],
        "evidence": [],
        "final_verdict": "UNCERTAIN",
        "agreement_summary": "Model agreement summary is not available.",
        "models": build_default_models(verdict),
        "seq2seq_answer": "No matching result in demo database.",
        "rag_answer": "No matching result in demo database.",
    }


def reset_app_state() -> None:
    """Reset the app to its initial interactive state."""
    st.session_state.claim_input = ""
    st.session_state.source_filter = SOURCE_FILTERS[0]
    st.session_state.result = None
    st.session_state.checked_claim = ""


def format_model_type(model_type: str) -> str:
    """Convert model type values into readable labels."""
    return {
        "closed_book": "closed-book",
        "rag": "retrieval-augmented",
        "graph_rag": "graph-based retrieval",
        "hybrid_rag": "hybrid retrieval",
    }.get(model_type, model_type)


# -------------------------------------------------------------------
# 5. Data adapter function
# -------------------------------------------------------------------
def get_fact_check_result(claim: str, source_filter: str) -> dict:
    """Return a fact-check result for the given claim.

    TODO: replace mock lookup with backend API call. The backend should return
    the same dictionary shape used by the render functions below.
    """
    normalized_claim = claim.strip()
    result = FAKE_RESULTS.get(normalized_claim)

    if result is None:
        return build_default_result(normalized_claim)

    filtered_evidence = filter_evidence_by_source(result["evidence"], source_filter)
    evidence_summary = (
        result["evidence_summary"]
        if source_filter == "All sources"
        else build_evidence_summary(filtered_evidence)
    )

    return {
        **result,
        "claim": normalized_claim,
        "evidence": filtered_evidence,
        "evidence_summary": evidence_summary,
    }


# -------------------------------------------------------------------
# 6. Render functions
# -------------------------------------------------------------------
def render_header() -> None:
    """Render the page header and short product description."""
    st.title(PAGE_TITLE)
    st.write(PAGE_DESCRIPTION)

    with st.container():
        st.info("\n".join(TOP_BANNER_LINES))


def render_verdict_card(result: dict) -> None:
    """Render the primary verdict card."""
    verdict = result["verdict"]
    verdict_title, default_summary = get_verdict_summary(verdict)
    summary = result.get("summary", default_summary)
    what_this_means = result.get("what_this_means", get_user_decision_message(verdict))

    with st.container(border=True):
        st.markdown(f"## {get_verdict_icon(verdict)} {verdict_title}")
        st.caption(summary)
        st.markdown("**What this means for you:**")
        st.write(what_this_means)
        st.caption(get_trusted_sources_label(result.get("evidence", [])))


def render_claim_assessment(result: dict) -> None:
    """Render verdict, key insights, and explanation."""
    st.markdown("## Claim Assessment")

    if result.get("claim"):
        st.caption(f"Claim checked: {result['claim']}")

    render_verdict_card(result)

    st.markdown("### Key insights")
    st.caption("What you should know:")
    st.markdown("\n".join(f"- {item}" for item in result.get("key_insights", [])))

    with st.expander("Explanation", expanded=False):
        st.write(result.get("explanation", "No explanation available."))


def render_input_panel() -> None:
    """Render claim input controls and the claim assessment result."""
    st.markdown("Try an example:")
    chip_columns = st.columns(3, gap="small")

    for column, example_claim in zip(chip_columns, EXAMPLE_CLAIMS):
        with column:
            if st.button(example_claim, use_container_width=True):
                st.session_state.claim_input = example_claim

    st.text_input(
        "Enter a claim to fact-check",
        placeholder="e.g. Vaccines cause infertility",
        key="claim_input",
    )

    st.radio(
        "Source filter",
        SOURCE_FILTERS,
        index=SOURCE_FILTERS.index(st.session_state.source_filter),
        key="source_filter",
    )

    action_col, clear_col = st.columns([3, 1], gap="small")

    with action_col:
        check_clicked = st.button("Check Claim", type="primary", use_container_width=True)

    with clear_col:
        clear_clicked = st.button("Clear", use_container_width=True)

    if check_clicked:
        if not st.session_state.claim_input.strip():
            st.warning("Please enter a claim to check.")
        else:
            with st.spinner("Analyzing claim using trusted sources..."):
                result = get_fact_check_result(
                    st.session_state.claim_input,
                    st.session_state.source_filter,
                )
            st.session_state.result = result
            st.session_state.checked_claim = st.session_state.claim_input.strip()

    if clear_clicked:
        reset_app_state()

    if st.session_state.result:
        render_claim_assessment(st.session_state.result)


def render_evidence_panel() -> None:
    """Render evidence summary and evidence cards."""
    st.markdown("## Evidence Panel")
    st.caption("Evidence retrieved from trusted public health sources.")
    st.caption(f"Showing results from: {st.session_state.source_filter}")

    result = st.session_state.result
    if not result:
        st.info("Evidence will appear here after you check a claim.")
        return

    evidence = result.get("evidence", [])
    st.caption(get_trusted_sources_label(evidence))

    st.markdown("#### Evidence Summary")
    evidence_summary = result.get("evidence_summary", build_evidence_summary(evidence))
    if isinstance(evidence_summary, list):
        st.markdown("\n".join(f"- {item}" for item in evidence_summary))
    else:
        st.write(evidence_summary)

    if not evidence:
        st.info("No supporting evidence is available for this claim in the current demo.")
        return

    source_badges = {"WHO": "[WHO]", "CDC": "[CDC]"}
    for index, item in enumerate(evidence, start=1):
        expander_title = (
            f"Evidence {index} | {item.get('source', 'Unknown')} | "
            f"{item.get('stance', 'Unspecified')}"
        )
        with st.expander(expander_title, expanded=index == 1):
            source = item.get("source", "Unknown")
            st.markdown(f"### {source_badges.get(source, '[Source]')} {source}")
            st.caption(item.get("stance", "Unspecified"))
            st.markdown(f"**Title:** {item.get('title', 'Untitled evidence')}")
            st.markdown("**Retrieved Text**")
            st.write(item.get("text", "No evidence text available."))
            if item.get("url"):
                st.markdown(f"[Open source]({item['url']})")


def render_model_card(model: dict) -> None:
    """Render one model output as a compact card."""
    evidence_sources = model.get("evidence_sources", [])
    evidence_label = ", ".join(evidence_sources) if evidence_sources else "No evidence"
    has_evidence_label = "Yes" if model.get("has_evidence") else "No"

    with st.container(border=True):
        st.markdown(f"#### {model.get('model_name', 'Model')}")
        st.caption(f"Type: {format_model_type(model.get('model_type', 'unknown'))}")
        st.markdown(f"**Verdict:** {model.get('verdict', 'UNKNOWN')}")
        st.markdown(f"**Summary:** {model.get('summary', 'No summary available.')}")
        st.markdown(f"**Has evidence:** {has_evidence_label}")
        st.markdown(f"**Evidence:** {evidence_label}")


def render_model_comparison() -> None:
    """Render the bottom multi-model comparison area."""
    result = st.session_state.result

    with st.expander("Multi-model Comparison", expanded=False):
        st.caption("Final decision based on outputs from 4 models")

        if not result:
            st.write("Model comparison will appear here after checking a claim.")
            return

        final_verdict = result.get("final_verdict", result.get("verdict", "UNKNOWN").upper())
        agreement_summary = result.get("agreement_summary", "Model agreement summary is not available.")
        models = result.get("models", build_default_models(result.get("verdict", "Uncertain")))

        with st.container(border=True):
            st.markdown(f"**Final verdict:** {final_verdict}")
            st.markdown(f"**Agreement summary:** {agreement_summary}")

        st.markdown("#### Model outputs")
        card_slots = [*st.columns(2, gap="large"), *st.columns(2, gap="large")]

        for slot, model in zip(card_slots, models[:4]):
            with slot:
                render_model_card(model)


def render_footer() -> None:
    """Render the bottom disclaimer."""
    st.caption(
        "This demo is for educational purposes only and does not provide medical advice."
    )


# -------------------------------------------------------------------
# 7. Main entry
# -------------------------------------------------------------------
def main() -> None:
    """Build and render the full Streamlit page."""
    configure_page()
    initialize_session_state()

    render_header()

    left_col, right_col = st.columns([2, 1], gap="large")

    with left_col:
        render_input_panel()

    with right_col:
        render_evidence_panel()

    st.divider()
    render_model_comparison()
    st.divider()
    render_footer()


if __name__ == "__main__":
    main()
