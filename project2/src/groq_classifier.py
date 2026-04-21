import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"


def classify_question(question):
    prompt = f"""
Classify the following question into ONE category:

Categories:
- dataset (questions about data used)
- methodology (how the system works)
- limitations (weaknesses or drawbacks)
- general (summary or broad understanding)
- unknown (if unclear)

Return ONLY one word.

Question:
{question}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip().lower()

        # Safety fallback
        valid = ["dataset", "methodology", "limitations", "general", "unknown"]

        if result not in valid:
            return "unknown"

        return result

    except Exception as e:
        print("Groq error:", e)
        return "unknown"