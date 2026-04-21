import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def classify_question(question):
    prompt = f"""
Classify the question into ONE category:

dataset
methodology
limitations
general
unknown

Return ONLY the category.

Question:
{question}
"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return res.choices[0].message.content.strip().lower()
    except:
        return "unknown"


def expand_query(question):
    prompt = f"""
Rewrite the question into 3 short search queries.

Keep them concise.

Question:
{question}
"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        lines = res.choices[0].message.content.split("\n")
        queries = [l.strip("- ").strip() for l in lines if l.strip()]

        return queries[:3]

    except:
        return [question]


def rerank_chunks(question, chunks):
    scored = []

    for chunk in chunks[:8]:  # limit to avoid token issues
        prompt = f"""
Score relevance from 1 to 5.

Question:
{question}

Chunk:
{chunk}

Return ONLY a number.
"""

        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            score = int(res.choices[0].message.content.strip())
        except:
            score = 1

        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [c for _, c in scored[:5]]