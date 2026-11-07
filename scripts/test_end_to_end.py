"""Run the complete AgentX workflow and export its report as PDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentx.core.end_to_end import (
    DocumentUnavailableError,
    EndToEndError,
    run_end_to_end,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Natural-language research question.")
    parser.add_argument(
        "--document-id",
        help="Optional ID already stored in the configured local ChromaDB; enables web + PDF workflow.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="PDF chunks to retrieve (default: 5).")
    parser.add_argument("--persist-directory", help="ChromaDB directory; defaults to configured AgentX storage.")
    parser.add_argument("--report-dir", help="Output directory for the Markdown and PDF reports.")
    args = parser.parse_args()

    try:
        result = run_end_to_end(
            args.question,
            document_id=args.document_id,
            top_k=args.top_k,
            persist_directory=args.persist_directory,
            report_output_dir=args.report_dir,
        )
    except DocumentUnavailableError as exc:
        print(f"Document unavailable: {exc}", file=sys.stderr)
        return 2
    except (EndToEndError, ValueError) as exc:
        print(f"End-to-end smoke test could not complete: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Avoid echoing arbitrary provider exceptions, which might contain
        # request details or credentials.
        print(
            f"End-to-end smoke test failed unexpectedly ({type(exc).__name__}).",
            file=sys.stderr,
        )
        return 1

    errors = result.state.get("errors")
    if isinstance(errors, list) and errors:
        print("Smoke-test result: report exported, but one or more agent steps failed.", file=sys.stderr)
        return 1
    if result.state.get("report_status") == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
