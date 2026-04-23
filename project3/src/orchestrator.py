from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agents import CitationExtractorAgent, PaperSummarizerAgent, ResearchGapFinderAgent
from src.llm import LlmError


@dataclass
class AgentRunRecord:
    agent: str
    ok: bool
    output: Any = None
    error: Optional[str] = None

    def _serialize_output(self) -> Any:
        return _output_to_serializable(self.output)


@dataclass
class ResearchAssistantReport:
    """Aggregated output from the multi-agent run."""

    paper_length_chars: int
    runs: List[AgentRunRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_length_chars": self.paper_length_chars,
            "agents": [
                {
                    "agent": r.agent,
                    "ok": r.ok,
                    "error": r.error,
                    "output": r._serialize_output(),
                }
                for r in self.runs
            ],
        }


def _output_to_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    f = getattr(obj, "__dataclass_fields__", None)
    if f:
        d: Dict[str, Any] = {}
        for k in f:
            v = getattr(obj, k)
            if isinstance(v, list):
                d[k] = [_output_to_serializable(x) for x in v]
            elif getattr(v, "__dataclass_fields__", None):
                d[k] = _output_to_serializable(v)
            else:
                d[k] = v
        return d
    if isinstance(obj, (list, tuple)):
        return [_output_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(a): _output_to_serializable(b) for a, b in obj.items()}
    return str(obj)


def _safe_call(name: str, fn, *args, **kwargs) -> AgentRunRecord:
    try:
        out = fn(*args, **kwargs)
        return AgentRunRecord(agent=name, ok=True, output=out)
    except LlmError as e:
        return AgentRunRecord(agent=name, ok=False, error=str(e))
    except Exception as e:  # noqa: BLE001
        return AgentRunRecord(agent=name, ok=False, error=str(e))


def run_research_assistant(
    paper_text: str, *, max_workers: int = 3
) -> ResearchAssistantReport:
    """
    Orchestrator: runs Paper Summarizer, Citation Extractor, and Research Gap Finder
    in parallel (each is an independent agent over the same text).
    """
    text = paper_text or ""
    summarizer = PaperSummarizerAgent()
    citations = CitationExtractorAgent()
    gaps = ResearchGapFinderAgent()
    report = ResearchAssistantReport(paper_length_chars=len(text))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        f_sum = ex.submit(_safe_call, summarizer.name, summarizer.run, text)
        f_cit = ex.submit(_safe_call, citations.name, citations.run, text)
        f_gap = ex.submit(_safe_call, gaps.name, gaps.run, text)
        for fut in (f_sum, f_cit, f_gap):
            report.runs.append(fut.result())

    # Stable order: summarizer, citations, gaps
    order = {PaperSummarizerAgent.name: 0, CitationExtractorAgent.name: 1, ResearchGapFinderAgent.name: 2}
    report.runs.sort(key=lambda r: order.get(r.agent, 99))
    return report
