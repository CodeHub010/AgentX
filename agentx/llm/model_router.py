"""Select a configured local or cloud chat model."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agentx.core.settings import ConfigurationError, Settings


@runtime_checkable
class ChatModel(Protocol):
    """Minimal interface consumed by AgentX agents."""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Run a chat request and return the provider's chat result."""
        ...

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Run a chat request asynchronously."""
        ...


class ModelConfigurationError(ConfigurationError):
    """Raised when the selected model provider cannot be configured."""


def _load_chat_ollama() -> type:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ModelConfigurationError(
            "Ollama support is unavailable. Install dependencies from requirements.txt."
        ) from exc
    return ChatOllama


def _load_chat_openai() -> type:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ModelConfigurationError(
            "OpenAI support is unavailable. Install dependencies from requirements.txt."
        ) from exc
    return ChatOpenAI


def _load_chat_gemini() -> type:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ModelConfigurationError(
            "Gemini support is unavailable. Install langchain-google-genai from requirements.txt."
        ) from exc
    return ChatGoogleGenerativeAI


def _build_timeout(settings: Settings) -> Any:
    """Create explicit connect and total request deadlines."""
    try:
        import httpx
    except ImportError as exc:
        raise ModelConfigurationError(
            "HTTPX is required for configured model timeouts; install requirements.txt."
        ) from exc
    return httpx.Timeout(
        timeout=settings.request_timeout_seconds,
        connect=settings.connect_timeout_seconds,
    )


def create_chat_model(
    provider: str | None = None,
    *,
    settings: Settings | None = None,
) -> ChatModel:
    """Create a chat model explicitly or using ``LLM_PROVIDER`` from settings.

    The return type is provider-neutral. Local Ollama mode does not require an
    OpenAI key; cloud mode validates its key before constructing the client.
    """
    configured = settings or Settings.from_env()
    selected = (provider or configured.llm_provider).strip().lower()
    if selected not in {"ollama", "openai", "gemini"}:
        raise ModelConfigurationError(
            f"Unsupported provider {selected!r}; choose 'ollama', 'openai', or 'gemini'."
        )

    if selected == "openai" and not configured.openai_api_key:
        raise ModelConfigurationError(
            "OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'. "
            "Set the key in your environment or switch to LLM_PROVIDER=ollama."
        )
    if selected == "gemini" and not configured.gemini_api_key:
        raise ModelConfigurationError(
            "GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'. "
            "Set it as an environment variable or a top-level Streamlit secret."
        )
    configured.validate(provider=selected)

    timeout = _build_timeout(configured)
    if selected == "ollama":
        model_class = _load_chat_ollama()
        return model_class(
            model=configured.ollama_model,
            base_url=configured.ollama_base_url,
            keep_alive="5m",
            client_kwargs={"timeout": timeout},
            async_client_kwargs={"timeout": timeout},
        )

    if selected == "gemini":
        model_class = _load_chat_gemini()
        try:
            return model_class(
                model=configured.gemini_model,
                api_key=configured.gemini_api_key,
                timeout=configured.request_timeout_seconds,
                max_retries=2,
            )
        except Exception as exc:
            raise ModelConfigurationError(
                f"Gemini chat client could not be initialized ({type(exc).__name__}). "
                "Check GEMINI_MODEL and provider configuration."
            ) from None

    model_class = _load_chat_openai()
    return model_class(
        model=configured.openai_model,
        api_key=configured.openai_api_key,
        timeout=timeout,
        max_retries=1,
    )


class LocalModelConnectionError(ConnectionError):
    """Raised when the local Ollama service cannot be reached."""


def initialize_local_model(*, settings: Settings | None = None) -> ChatModel:
    """Initialize the configured local Ollama chat model without making a call."""
    return create_chat_model("ollama", settings=settings)


def send_prompt(
    prompt: str,
    *,
    model: ChatModel | None = None,
    settings: Settings | None = None,
) -> str:
    """Send a text prompt to the local model and return its response text."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    configured = settings or Settings.from_env()
    active_model = model or initialize_local_model(settings=configured)
    try:
        response = active_model.invoke(prompt)
    except (ConnectionError, TimeoutError) as exc:
        raise LocalModelConnectionError(
            f"Could not connect to Ollama at {configured.ollama_base_url}. "
            "Confirm Ollama is running and the configured model is available."
        ) from exc
    except Exception as exc:
        try:
            import httpx
        except ImportError:
            raise exc
        if isinstance(exc, httpx.RequestError):
            raise LocalModelConnectionError(
                f"Could not connect to Ollama at {configured.ollama_base_url}. "
                "Confirm Ollama is running and the configured model is available."
            ) from exc
        raise

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
        )
    return str(content)
