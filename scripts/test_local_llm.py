"""Run a real inference request against the local AgentX Ollama gateway."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

# Make `python scripts/test_local_llm.py` work from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentx.core.settings import Settings
from agentx.llm import (
    LocalModelConnectionError,
    ModelConfigurationError,
    initialize_local_model,
    send_prompt,
)

PROVIDER = "ollama"
MODEL_NAME = "llama3.2:1b"
PROMPT = "Explain what an AI research assistant does in two sentences."


def _is_model_not_found(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    message = str(error).lower()
    return status_code == 404 or (
        "model" in message and ("not found" in message or "pull" in message)
    )


def main() -> int:
    settings = replace(Settings.from_env(), ollama_model=MODEL_NAME)
    try:
        model = initialize_local_model(settings=settings)
        response = send_prompt(PROMPT, model=model, settings=settings)
    except LocalModelConnectionError as exc:
        print(f"Inference failed: could not connect to Ollama: {exc}", file=sys.stderr)
        return 1
    except ModelConfigurationError as exc:
        print(f"Inference failed: AgentX model configuration error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if _is_model_not_found(exc):
            print(
                f"Inference failed: Ollama model '{MODEL_NAME}' was not found. "
                f"Install it explicitly with: ollama pull {MODEL_NAME}",
                file=sys.stderr,
            )
        else:
            print(f"Inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Provider: {PROVIDER}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}")
    print("Response:")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
