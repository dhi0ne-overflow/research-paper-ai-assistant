import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.agents import CitationExtractorAgent, PaperSummarizerAgent, ResearchGapFinderAgent
from src.llm import LlmError, active_llm_label, agent_llm_label
from src.orchestrator import run_research_assistant
from src.pdf_processor import extract_text_from_pdf

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🧩",
    layout="wide",
)

st.title("Multi-Agent Research Assistant")
st.caption("Paper Summarizer · Citation Extractor · Research Gap Finder (parallel)")

with st.expander("How it works", expanded=False):
    st.markdown(
        f"""
        Three **specialist agents** share the same paper text, orchestrated in **parallel**:
        - **Paper summarizer** — structured summary and contributions
        - **Citation extractor** — in-text and bibliography-style references
        - **Research gap finder** — limitations, open questions, conservative inferred gaps
        Configure **`project3/.env`**: set `GROQ_API_KEY` (default provider when present)
        or `GEMINI_API_KEY`, or set `LLM_PROVIDER` to `groq` or `gemini`. See `README.md`
        in this folder. Current backend: **{active_llm_label()}**.
        """
    )

uploaded = st.file_uploader("Upload a research PDF", type=["pdf"])
paste = st.text_area("Or paste paper text", height=200, placeholder="Optional if you upload a PDF…")

if uploaded:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.session_state["paper_path"] = path

run_btn = st.button("Run all agents", type="primary")
paper_text = paste.strip() if paste else ""
if st.session_state.get("paper_path") and not paper_text:
    paper_text = extract_text_from_pdf(st.session_state["paper_path"])


def has_valid_text(text: str) -> bool:
    return bool(text and len(text.strip()) >= 80)


def run_single_agent(agent_name: str, paper: str):
    if agent_name == "paper_summarizer":
        return PaperSummarizerAgent().run(paper)
    if agent_name == "citation_extractor":
        return CitationExtractorAgent().run(paper)
    return ResearchGapFinderAgent().run(paper)


if run_btn:
    if not has_valid_text(paper_text):
        st.warning("Add enough text (upload a PDF or paste text).")
    else:
        with st.spinner("Running agents in parallel…"):
            report = run_research_assistant(paper_text)
        st.session_state["last_report"] = report

with st.expander("Advanced mode: run individual agents", expanded=False):
    st.caption("Optional debugging mode. Run one specialist at a time.")
    col_sum, col_cit, col_gap = st.columns(3)
    run_sum = col_sum.button("Run paper summarizer only", use_container_width=True)
    run_cit = col_cit.button("Run citation extractor only", use_container_width=True)
    run_gap = col_gap.button("Run research gap finder only", use_container_width=True)

    if "individual_runs" not in st.session_state:
        st.session_state["individual_runs"] = {}

    def handle_single_run(agent_name: str, spinner_label: str):
        if not has_valid_text(paper_text):
            st.warning("Add enough text first (upload a PDF or paste text).")
            return
        model_label = agent_llm_label(agent_name)
        try:
            with st.spinner(spinner_label):
                out = run_single_agent(agent_name, paper_text)
            st.session_state["individual_runs"][agent_name] = {
                "ok": True,
                "output": out,
                "error": None,
                "model_used": model_label,
            }
        except LlmError as e:
            st.session_state["individual_runs"][agent_name] = {
                "ok": False,
                "output": None,
                "error": str(e),
                "model_used": model_label,
            }
        except Exception as e:  # noqa: BLE001
            st.session_state["individual_runs"][agent_name] = {
                "ok": False,
                "output": None,
                "error": str(e),
                "model_used": model_label,
            }

    if run_sum:
        handle_single_run("paper_summarizer", "Running paper summarizer…")
    if run_cit:
        handle_single_run("citation_extractor", "Running citation extractor…")
    if run_gap:
        handle_single_run("research_gap_finder", "Running research gap finder…")

    single = st.session_state["individual_runs"]
    st.markdown("### Individual run results")
    tab_sum_one, tab_cit_one, tab_gap_one = st.tabs(
        ["Paper summarizer", "Citation extractor", "Research gap finder"]
    )

    with tab_sum_one:
        s = single.get("paper_summarizer")
        if s and s.get("model_used"):
            st.caption(f"Model: `{s['model_used']}`")
        if s and s.get("ok") and s.get("output"):
            st.markdown(s["output"].text)
        elif s and not s.get("ok"):
            st.error(s.get("error"))
        else:
            st.info("No individual summarizer run yet.")

    with tab_cit_one:
        c = single.get("citation_extractor")
        if c and c.get("model_used"):
            st.caption(f"Model: `{c['model_used']}`")
        if c and c.get("ok") and c.get("output"):
            st.markdown(c["output"].raw_markdown)
            if c["output"].items:
                st.caption(f"Structured items: {len(c['output'].items)}")
        elif c and not c.get("ok"):
            st.error(c.get("error"))
        else:
            st.info("No individual citation extractor run yet.")

    with tab_gap_one:
        g = single.get("research_gap_finder")
        if g and g.get("model_used"):
            st.caption(f"Model: `{g['model_used']}`")
        if g and g.get("ok") and g.get("output"):
            st.markdown(g["output"].text)
        elif g and not g.get("ok"):
            st.error(g.get("error"))
        else:
            st.info("No individual gap finder run yet.")

if "last_report" in st.session_state:
    rpt = st.session_state["last_report"]
    st.subheader("Run results")
    st.metric("Paper length (chars)", f"{rpt.paper_length_chars:,}")

    by_name = {rec.agent: rec for rec in rpt.runs}
    tab_sum, tab_cit, tab_gap = st.tabs(
        [
            "Paper summarizer",
            "Citation extractor",
            "Research gap finder",
        ]
    )

    with tab_sum:
        s = by_name.get("paper_summarizer")
        if s and s.model_used:
            st.caption(f"Model: `{s.model_used}`")
        if s and s.ok and s.output:
            st.markdown(s.output.text)
        elif s and not s.ok:
            st.error(s.error)
        else:
            st.info("No summarizer result.")

    with tab_cit:
        c = by_name.get("citation_extractor")
        if c and c.model_used:
            st.caption(f"Model: `{c.model_used}`")
        if c and c.ok and c.output:
            st.markdown(c.output.raw_markdown)
            if c.output.items:
                st.caption(f"Structured items: {len(c.output.items)}")
        elif c and not c.ok:
            st.error(c.error)
        else:
            st.info("No citation extractor result.")

    with tab_gap:
        g = by_name.get("research_gap_finder")
        if g and g.model_used:
            st.caption(f"Model: `{g.model_used}`")
        if g and g.ok and g.output:
            st.markdown(g.output.text)
        elif g and not g.ok:
            st.error(g.error)
        else:
            st.info("No gap finder result.")

    st.download_button(
        "Download JSON report",
        data=json.dumps(rpt.to_dict(), indent=2, ensure_ascii=False),
        file_name="research_assistant_report.json",
        mime="application/json",
    )
