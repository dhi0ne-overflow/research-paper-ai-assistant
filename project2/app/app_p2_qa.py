import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.pdf_processor import extract_text_from_pdf
from src.ocr_processor import extract_text_with_ocr
from src.rag_pipeline import build_rag, answer_question
from src.vector_store import load_index, list_papers

st.set_page_config(page_title="Research Paper AI Assistant", layout="centered")

st.title("📄 Research Paper AI Assistant")

# -------- Upload & Process --------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Uploaded!")

    if st.button("Process Paper"):
        with st.spinner("Processing... ⏳"):
            text = extract_text_from_pdf(temp_path)

            if len(text.strip()) < 100:
                text = extract_text_with_ocr(temp_path)

            paper_name = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")

            chunks, index = build_rag(text, paper_name)

            st.session_state["paper"] = paper_name
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index

            st.success(f"Saved as: {paper_name}")
            st.rerun()

# -------- Select Existing Paper --------
papers = list_papers()

if papers:
    selected = st.selectbox("Select Paper", papers)

    if selected:
        index, chunks = load_index(selected)
        if index is not None:
            st.session_state["paper"] = selected
            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
            st.success(f"Loaded: {selected}")

# -------- Q&A --------
if "index" in st.session_state:
    st.markdown("## 💬 Ask Questions")

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Thinking... 🤖"):
                answer, intent = answer_question(
                    question,
                    st.session_state["chunks"],
                    st.session_state["index"]
                )
                st.session_state["last_answer"] = answer
                st.session_state["last_intent"] = intent

    if "last_answer" in st.session_state:
        st.markdown("### 📌 Answer")
        st.write(st.session_state["last_answer"])
        st.markdown(f"**Detected Type:** `{st.session_state['last_intent']}`")

# -------- Reset --------
if st.button("🗑️ Reset All Papers"):
    base = os.path.join("data", "papers")
    if os.path.exists(base):
        import shutil
        shutil.rmtree(base)
    st.session_state.clear()
    st.success("All papers deleted. Start fresh.")