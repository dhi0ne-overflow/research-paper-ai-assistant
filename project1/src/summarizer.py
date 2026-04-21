import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ⚠️ IMPORTANT: Use a model that exists in your list_models()
model = genai.GenerativeModel("gemini-flash-latest")


def summarize_paper(text):
    # Reduce token usage
    trimmed_text = text[:7000]

    prompt = f"""
You are a helpful research assistant.

Analyze the research paper and explain it in a simple, student-friendly way.

INTERNAL THINKING (DO NOT SHOW):
- Understand the core idea
- Identify strengths and weaknesses

OUTPUT FORMAT (ONLY SHOW):

1. Clear Summary
- Explain in simple language
- What the paper does
- How it works
- 5–7 sentences

2. Key Contributions
- Bullet points
- Simple explanations

3. Limitations
- Bullet points
- Simple explanations

RULES:
- Avoid complex jargon
- Keep explanation clear and easy
- Do NOT add extra sections

------------------------
Paper Content:
{trimmed_text}
"""

    try:
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        if "429" in str(e):
            return "⚠️ Rate limit reached. Please wait 1 minute and try again."
        else:
            return f"⚠️ Error: {str(e)}"