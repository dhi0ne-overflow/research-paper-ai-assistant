import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.pdf_processor import extract_text_from_pdf
from src.ocr_processor import extract_text_with_ocr
from src.summarizer import summarize_paper

# Page config
st.set_page_config(
    page_title="Research Paper AI Assistant",
    page_icon="📄",
    layout="centered"
)

# CSS
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #00ffcc;
}

.subtitle {
    text-align: center;
    color: #cccccc;
    margin-bottom: 25px;
}

.result-text {
    font-size: 16px;
    line-height: 1.7;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📄 Research Paper AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a research paper and get AI insights</div>', unsafe_allow_html=True)

# ✅ Cache (VERY IMPORTANT)
@st.cache_data(show_spinner=False)
def cached_summary(text):
    return summarize_paper(text)

# Clear cache
if st.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.session_state.clear()
    st.success("Cache cleared!")

# Upload
uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

if uploaded_file:
    os.makedirs("data/raw", exist_ok=True)

    file_path = os.path.join("data/raw", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded!")

    if st.button("🚀 Analyze Paper"):

        # Prevent multiple API calls
        if "result" not in st.session_state:

            with st.spinner("Processing... ⏳"):

                text = extract_text_from_pdf(file_path)

                if len(text.strip()) < 100:
                    st.warning("Using OCR...")
                    text = extract_text_with_ocr(file_path)

                result = cached_summary(text)

                st.session_state["result"] = result

        # Show result
        if "result" in st.session_state:
            st.markdown("## 📌 Analysis Result")

            st.markdown(
                f'<div class="result-text">{st.session_state["result"]}</div>',
                unsafe_allow_html=True
            )