import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

from src.chunking import chunk_text
from src.embeddings import embed_text
from src.vector_store import build_index, save_index, retrieve_chunks
from src.groq_utils import classify_question, expand_query, rerank_chunks

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-flash-latest")


def build_rag(text, paper_name):
    chunks = chunk_text(text)
    embeddings = embed_text(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = build_index(embeddings)
    save_index(index, chunks, paper_name)

    return chunks, index


def answer_question(question, chunks, index):
    # 1) intent
    intent = classify_question(question)

    # 2) query expansion
    queries = expand_query(question)

    # 3) retrieve
    all_chunks = []
    for q in queries:
        emb = embed_text([q])
        retrieved = retrieve_chunks(index, emb, chunks, top_k=8)
        all_chunks.extend(retrieved)

    # deduplicate (simple)
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        key = c[:200]
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    # 4) re-rank
    best_chunks = rerank_chunks(question, unique_chunks)  # returns top ~5 texts

    # 5) build context (limit length)
    context = "\n\n".join([c[:500] for c in best_chunks[:5]])

    # 6) intent-aware instruction
    if intent == "dataset":
        instruction = "Focus on identifying dataset or data used."
    elif intent == "methodology":
        instruction = "Explain how the method works."
    elif intent == "limitations":
        instruction = "Focus on limitations and weaknesses."
    else:
        instruction = "Answer clearly and simply."

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the provided context.

{instruction}

Rules:
- Use simple language
- If not found → say "Not found in the paper"
- Do not hallucinate

Context:
{context}

Question:
{question}
"""

    try:
        res = model.generate_content(prompt)
        return res.text, intent
    except Exception as e:
        if "429" in str(e):
            return "⚠️ Rate limit reached. Try again in ~1 minute.", intent
        return f"Error: {str(e)}", intent