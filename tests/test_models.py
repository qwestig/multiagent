from __future__ import annotations

import unittest

from mkpi_app.models import OPENAI_BASE_URL, OPENROUTER_BASE_URL, OpenAICompatibleAdapter, provider_configuration_hint


class ModelConfigTests(unittest.TestCase):
    def test_openrouter_key_with_openai_base_url_gets_hint(self) -> None:
        hint = provider_configuration_hint("sk-or-v1-test", OPENAI_BASE_URL, "gpt-oss-120b")
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn(OPENROUTER_BASE_URL, hint)

    def test_openrouter_headers_are_added(self) -> None:
        adapter = OpenAICompatibleAdapter(
            model="gpt-oss-120b",
            api_key="sk-or-v1-test",
            base_url=OPENROUTER_BASE_URL,
        )
        headers = adapter._build_headers()
        self.assertIn("HTTP-Referer", headers)
        self.assertIn("X-Title", headers)

    def test_mismatched_key_raises_friendly_error_before_network(self) -> None:
        adapter = OpenAICompatibleAdapter(
            model="gpt-oss-120b",
            api_key="sk-or-v1-test",
            base_url=OPENAI_BASE_URL,
        )
        with self.assertRaises(RuntimeError) as context:
            adapter.generate(
                type(
                    "Request",
                    (),
                    {
                        "system_prompt": "sys",
                        "user_prompt": "user",
                        "temperature": 0.2,
                        "max_tokens": 16,
                    },
                )()
            )
        self.assertIn("OpenRouter", str(context.exception))


if __name__ == "__main__":
    unittest.main()
