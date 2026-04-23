import json
import re
from dataclasses import dataclass, field
from typing import List

from src.llm import generate


@dataclass
class CitationItem:
    raw: str
    kind: str  # in_text, bibliography, unknown
    context: str = ""


@dataclass
class CitationExtractionResult:
    items: List[CitationItem] = field(default_factory=list)
    raw_markdown: str = ""


def _items_to_display(items: List[CitationItem]) -> str:
    lines = ["| kind | citation | context |", "| --- | --- | --- |"]
    for it in items:
        ctx = (it.context or "-").replace("|", "\\|")
        raw = (it.raw or "-").replace("|", "\\|")
        lines.append(f"| {it.kind} | {raw} | {ctx} |")
    return "\n".join(lines)


def _parse_json_list(raw: str) -> List[dict]:
    # Strip markdown code fences
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    data = json.loads(t)
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


class CitationExtractorAgent:
    """Extracts in-text citations and bibliography-style entries from the paper text."""

    name = "citation_extractor"

    def run(self, paper_text: str) -> CitationExtractionResult:
        trimmed = paper_text[:28_000]
        prompt = f"""You are the Citation Extractor agent. Extract scholarly citations from the paper.

Return ONLY a JSON array (no other text), each element an object with keys:
- "raw" (string): the citation as written, or reference line
- "kind": one of "in_text" | "bibliography" | "unknown"
- "context" (string): 1 short sentence of surrounding context, or "" if a bibliography line

Max 40 items, prioritize distinct works. If the excerpt has a references section, include bibliography items from it.

PAPER:
{trimmed}
"""
        raw = generate(prompt, temperature=0.0)
        items: List[CitationItem] = []
        try:
            for obj in _parse_json_list(raw)[:40]:
                items.append(
                    CitationItem(
                        raw=str(obj.get("raw", "")).strip(),
                        kind=str(obj.get("kind", "unknown") or "unknown").lower(),
                        context=str(obj.get("context", "") or "").strip(),
                    )
                )
        except (json.JSONDecodeError, TypeError, ValueError):
            items = []
        if items:
            display = _items_to_display(items)
        else:
            display = f"*(Could not parse structured citations; raw model output below.)*\n\n```\n{raw}\n```"
        return CitationExtractionResult(items=items, raw_markdown=display)
