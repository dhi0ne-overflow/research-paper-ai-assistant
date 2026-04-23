"""Smoke tests (no API keys): orchestration, JSON report, imports."""
import json
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _fake_generate(prompt, temperature=0.2, agent=None):
    if agent == "paper_summarizer":
        return "## Summary\nMock summary ok."
    if agent == "citation_extractor":
        return json.dumps(
            [{"raw": "Doe et al. 2020", "kind": "in_text", "context": "Prior work."}]
        )
    if agent == "research_gap_finder":
        return "## Gaps\nMock gap analysis ok."
    raise AssertionError(f"unexpected agent {agent!r}")


class TestOrchestratorMocked(unittest.TestCase):
    def test_parallel_agents_and_report(self):
        with patch("src.agents.summarizer.generate", side_effect=_fake_generate), patch(
            "src.agents.citations.generate", side_effect=_fake_generate
        ), patch("src.agents.gaps.generate", side_effect=_fake_generate):
            from src.orchestrator import run_research_assistant

            rpt = run_research_assistant("x" * 200)
            self.assertEqual(rpt.paper_length_chars, 200)
            self.assertEqual(len(rpt.runs), 3)
            by = {x.agent: x for x in rpt.runs}
            self.assertTrue(by["paper_summarizer"].ok)
            self.assertIn("Mock summary", by["paper_summarizer"].output.text)
            self.assertTrue(by["citation_extractor"].ok)
            self.assertTrue(by["citation_extractor"].output.items)
            self.assertTrue(by["research_gap_finder"].ok)
            for x in rpt.runs:
                self.assertIsNotNone(x.model_used)
            d = rpt.to_dict()
            self.assertIn("model_used", d["agents"][0])


class TestImports(unittest.TestCase):
    def test_core_modules_import(self):
        import importlib

        importlib.import_module("src.llm")
        importlib.import_module("src.pdf_processor")
        importlib.import_module("src.orchestrator")


if __name__ == "__main__":
    unittest.main()
