"""Run an AgentX research workflow using demo search data by default."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentx.agents import ResearchAgent
from agentx.core.settings import Settings
from agentx.tools.web_search import SearchConfigurationError, SearchProviderError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="What does an AI research assistant do, and why are citations useful?",
    )
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()

    try:
        settings = Settings.from_env()
        result = ResearchAgent(settings=settings).research(
            args.question, max_results=args.max_results
        )
    except (SearchConfigurationError, SearchProviderError) as exc:
        print(f"Research failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Research failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Search provider: {result.search_provider}")
    if result.search_provider == "demo":
        print("Source mode: demo fixtures; this is not live web research.")
    print(f"LLM: {result.llm_provider}/{result.llm_model}")
    print(f"Question: {result.question}\n")
    print(f"Answer: {result.summary}\n")
    print("Citations:")
    if not result.citations:
        print("  No citations (no sources found).")
    for citation in result.citations:
        date = f" ({citation.published_date})" if citation.published_date else ""
        print(f"  [{citation.number}] {citation.title}{date}\n      {citation.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
