from dataclasses import dataclass

from src.llm import generate


@dataclass
class GapAnalysisResult:
    text: str


class ResearchGapFinderAgent:
    """Identifies open problems, limitations, and research gaps implied by the paper."""

    name = "research_gap_finder"

    def run(self, paper_text: str) -> GapAnalysisResult:
        trimmed = paper_text[:24_000]
        prompt = f"""You are the Research Gap Finder agent. Based ONLY on the paper text below, identify:

1) Stated limitations or weaknesses (quote or paraphrase carefully).
2) Gaps the authors acknowledge (open questions, future work).
3) Potential research gaps a reader might notice (what is not evaluated, not compared, or not fully explained) — mark these as "inferred" and keep them conservative.

Format with clear headings. If the excerpt is too short to assess gaps, say so explicitly.

PAPER:
{trimmed}
"""
        text = generate(prompt, temperature=0.25, agent=self.name)
        return GapAnalysisResult(text=text)
