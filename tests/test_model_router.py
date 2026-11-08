"""Unit tests for model configuration and provider routing."""

import unittest
from unittest.mock import Mock, patch

from agentx.core.settings import ConfigurationError, Settings
from agentx.llm.model_router import (
    LocalModelConnectionError,
    ModelConfigurationError,
    create_chat_model,
    initialize_local_model,
    send_prompt,
)


class SettingsTests(unittest.TestCase):
    def test_configuration_validation_rejects_non_numeric_timeout(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "LLM_REQUEST_TIMEOUT_SECONDS must be a number"):
            Settings.from_env({"LLM_REQUEST_TIMEOUT_SECONDS": "soon"})

    def test_configuration_validation_rejects_non_positive_timeout(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "greater than zero"):
            Settings.from_env({"LLM_CONNECT_TIMEOUT_SECONDS": "0"})

    def test_configuration_validation_rejects_empty_selected_model(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "OLLAMA_MODEL must not be empty"):
            Settings.from_env({"OLLAMA_MODEL": "  "})


class ModelRouterTests(unittest.TestCase):
    def test_selects_ollama_without_cloud_api_key(self) -> None:
        ollama_class = Mock()
        settings = Settings(openai_api_key=None)
        with patch("agentx.llm.model_router._build_timeout", return_value=object()), patch("agentx.llm.model_router._load_chat_ollama", return_value=ollama_class):
            model = create_chat_model(settings=settings)
        ollama_class.assert_called_once()
        self.assertIs(model, ollama_class.return_value)
        args = ollama_class.call_args.kwargs
        self.assertEqual(args["model"], settings.ollama_model)
        self.assertEqual(args["base_url"], settings.ollama_base_url)
        self.assertIn("timeout", args["client_kwargs"])

    def test_initialize_local_model_uses_configured_endpoint_and_model(self) -> None:
        ollama_class = Mock()
        settings = Settings(ollama_model="llama3.2:1b")
        with patch("agentx.llm.model_router._build_timeout", return_value=object()), patch(
            "agentx.llm.model_router._load_chat_ollama", return_value=ollama_class
        ):
            model = initialize_local_model(settings=settings)
        self.assertIs(model, ollama_class.return_value)
        self.assertEqual(ollama_class.call_args.kwargs["base_url"], "http://localhost:11434")
        self.assertEqual(ollama_class.call_args.kwargs["model"], "llama3.2:1b")
        self.assertEqual(ollama_class.call_args.kwargs["keep_alive"], "5m")

    def test_send_prompt_returns_chat_message_content(self) -> None:
        model = Mock()
        model.invoke.return_value.content = "Local response"
        response = send_prompt("Hello", model=model, settings=Settings())
        self.assertEqual(response, "Local response")
        model.invoke.assert_called_once_with("Hello")

    def test_send_prompt_wraps_connection_errors(self) -> None:
        model = Mock()
        model.invoke.side_effect = ConnectionError("connection refused")
        with self.assertRaisesRegex(LocalModelConnectionError, "http://localhost:11434"):
            send_prompt("Hello", model=model, settings=Settings())
    def test_explicit_cloud_selection_uses_openai(self) -> None:
        openai_class = Mock()
        settings = Settings(openai_api_key="test-placeholder")
        with patch("agentx.llm.model_router._build_timeout", return_value=object()), patch("agentx.llm.model_router._load_chat_openai", return_value=openai_class):
            model = create_chat_model("openai", settings=settings)
        openai_class.assert_called_once()
        self.assertIs(model, openai_class.return_value)
        self.assertEqual(openai_class.call_args.kwargs["api_key"], "test-placeholder")

    def test_missing_cloud_api_key_has_clear_error(self) -> None:
        settings = Settings(llm_provider="openai", openai_api_key=None)
        with self.assertRaisesRegex(ModelConfigurationError, "OPENAI_API_KEY is required"):
            create_chat_model(settings=settings)

    def test_invalid_provider_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "choose 'ollama', 'openai', or 'gemini'"):
            Settings.from_env({"LLM_PROVIDER": "anthropic"})
        with self.assertRaisesRegex(ConfigurationError, "Unsupported provider"):
            create_chat_model("unknown", settings=Settings())


if __name__ == "__main__":
    unittest.main()
