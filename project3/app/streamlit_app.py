import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.orchestrator import run_research_assistant
from src.pdf_processor import extract_text_from_pdf

st.set_page_config(
    page_title="Research Assistant — Multi-Agent",
    page_icon="🧩",
    layout="wide",
)

st.title("Research Assistant — agentic / multi-agent")
st.caption("Paper Summarizer · Citation Extractor · Research Gap Finder (parallel)")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
        Three **specialist agents** share the same paper text, orchestrated in **parallel**:
        - **Paper summarizer** — structured summary and contributions
        - **Citation extractor** — in-text and bibliography-style references
        - **Research gap finder** — limitations, open questions, conservative inferred gaps
        Set `GEMINI_API_KEY` in `.env` (and optionally `GEMINI_MODEL`). Uses
        `gemini-flash-latest` by default, consistent with other projects in this repo.
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

if run_btn:
    if not paper_text or len(paper_text.strip()) < 80:
        st.warning("Add enough text (upload a PDF or paste text).")
    else:
        with st.spinner("Running agents in parallel…"):
            report = run_research_assistant(paper_text)
        st.session_state["last_report"] = report

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
        if s and s.ok and s.output:
            st.markdown(s.output.text)
        elif s and not s.ok:
            st.error(s.error)
        else:
            st.info("No summarizer result.")

    with tab_cit:
        c = by_name.get("citation_extractor")
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
