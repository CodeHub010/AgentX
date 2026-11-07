"""Environment based configuration for AgentX."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping


class ConfigurationError(ValueError):
    """Raised when AgentX configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings shared by AgentX components."""

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_embedding_model: str = "nomic-embed-text"
    embedding_batch_size: int = 8
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    web_search_provider: str = "demo"
    tavily_api_key: str | None = None
    connect_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Read settings from environment variables and an optional local .env."""
        if environ is None:
            try:
                from dotenv import load_dotenv
            except ImportError:
                pass
            else:
                load_dotenv()
            environ = os.environ

        provider = environ.get("LLM_PROVIDER", "ollama").strip().lower()
        ollama_url = environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        ollama_model = environ.get("OLLAMA_MODEL", "llama3.2:1b").strip()
        embedding_model = environ.get(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
        ).strip()
        try:
            embedding_batch_size = int(environ.get("OLLAMA_EMBEDDING_BATCH_SIZE", "8"))
        except ValueError as exc:
            raise ConfigurationError(
                "OLLAMA_EMBEDDING_BATCH_SIZE must be a positive integer."
            ) from exc
        openai_key = environ.get("OPENAI_API_KEY", "").strip() or None
        openai_model = environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
        web_search_provider = environ.get("WEB_SEARCH_PROVIDER", "demo").strip().lower()
        tavily_api_key = environ.get("TAVILY_API_KEY", "").strip() or None

        connect_timeout = cls._read_timeout(environ, "LLM_CONNECT_TIMEOUT_SECONDS", 10.0)
        request_timeout = cls._read_timeout(environ, "LLM_REQUEST_TIMEOUT_SECONDS", 120.0)

        settings = cls(
            llm_provider=provider,
            ollama_base_url=ollama_url,
            ollama_model=ollama_model,
            ollama_embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            openai_api_key=openai_key,
            openai_model=openai_model,
            web_search_provider=web_search_provider,
            tavily_api_key=tavily_api_key,
            connect_timeout_seconds=connect_timeout,
            request_timeout_seconds=request_timeout,
        )
        settings.validate()
        return settings

    @staticmethod
    def _read_timeout(environ: Mapping[str, str], name: str, default: float) -> float:
        raw = environ.get(name, str(default)).strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigurationError(f"{name} must be a number; received {raw!r}.") from exc
        if not math.isfinite(value) or value <= 0:
            raise ConfigurationError(f"{name} must be greater than zero.")
        return value

    def validate(self, provider: str | None = None) -> None:
        """Validate shared values and the selected provider's configuration."""
        selected = (provider or self.llm_provider).strip().lower()
        if selected not in {"ollama", "openai"}:
            raise ConfigurationError(
                f"Unsupported LLM_PROVIDER {selected!r}; choose 'ollama' or 'openai'."
            )
        for name, value in (
            ("LLM_CONNECT_TIMEOUT_SECONDS", self.connect_timeout_seconds),
            ("LLM_REQUEST_TIMEOUT_SECONDS", self.request_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be greater than zero and finite.")
        if selected == "ollama":
            if not self.ollama_base_url:
                raise ConfigurationError("OLLAMA_BASE_URL must not be empty.")
            if not self.ollama_model:
                raise ConfigurationError("OLLAMA_MODEL must not be empty.")
        elif not self.openai_model:
            raise ConfigurationError("OPENAI_MODEL must not be empty.")
        if not self.ollama_embedding_model:
            raise ConfigurationError("OLLAMA_EMBEDDING_MODEL must not be empty.")
        if self.embedding_batch_size < 1:
            raise ConfigurationError("OLLAMA_EMBEDDING_BATCH_SIZE must be at least 1.")
        if self.web_search_provider not in {"demo", "tavily"}:
            raise ConfigurationError(
                f"Unsupported WEB_SEARCH_PROVIDER {self.web_search_provider!r}; "
                "choose 'demo' or 'tavily'."
            )
        if self.web_search_provider == "tavily" and not self.tavily_api_key:
            raise ConfigurationError(
                "TAVILY_API_KEY is required when WEB_SEARCH_PROVIDER is 'tavily'."
            )
