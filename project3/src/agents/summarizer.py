from dataclasses import dataclass

from src.llm import generate


@dataclass
class SummarizerResult:
    text: str


class PaperSummarizerAgent:
    """Summarizes a research paper in clear, structured sections."""

    name = "paper_summarizer"

    def run(self, paper_text: str) -> SummarizerResult:
        trimmed = paper_text[:24_000]
        prompt = f"""You are the Paper Summarizer agent in a multi-agent research assistant.

Read the research paper excerpt and produce:

1) One-paragraph overview (what problem, what approach, main result).
2) Key contributions (3–5 bullet points).
3) Method / setup (short; focus on what a reader must know).
4) Main results (bullets; include metrics only if present in the text).
5) Limitations mentioned by the authors (if any; else say "Not clearly stated in the excerpt").

Rules: Use only the text below. No external knowledge. Simple language.

---
PAPER:
{trimmed}
"""
        text = generate(prompt, temperature=0.2, agent=self.name)
        return SummarizerResult(text=text)
