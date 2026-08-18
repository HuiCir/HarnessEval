from __future__ import annotations

import asyncio
import http.client
import json
import unittest
from unittest.mock import patch

from benchmark_platform.harnesses.api import ApiConfig, OpenAICompatibleClient


class _Response:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.body


class NativeTransportTests(unittest.TestCase):
    def test_native_messages_and_tool_schema_are_not_rewritten_or_sliced(self) -> None:
        large_argument = "x" * 200_000
        messages = [
            {"role": "system", "content": "policy"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": json.dumps({"query": large_argument}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "complete result"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        observed = {}

        def fake_urlopen(request, timeout):
            observed["payload"] = json.loads(request.data.decode("utf-8"))
            return _Response(
                {
                    "choices": [{"message": {"role": "assistant", "content": "done"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )

        client = OpenAICompatibleClient(
            ApiConfig("https://example.invalid/v1", "secret", "model", transport_retries=0)
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            completion = asyncio.run(
                client.complete_native(messages, tools=tools, tool_choice="required")
            )

        self.assertEqual(completion.content, "done")
        self.assertEqual(observed["payload"]["messages"], messages)
        self.assertEqual(observed["payload"]["tools"], tools)
        self.assertEqual(observed["payload"]["tool_choice"], "required")
        self.assertEqual(
            json.loads(observed["payload"]["messages"][1]["tool_calls"][0]["function"]["arguments"])["query"],
            large_argument,
        )

    def test_remote_disconnect_uses_transport_retry_budget(self) -> None:
        client = OpenAICompatibleClient(
            ApiConfig("https://example.invalid/v1", "secret", "model", transport_retries=1)
        )
        response = _Response(
            {
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )
        with (
            patch(
                "urllib.request.urlopen",
                side_effect=[http.client.RemoteDisconnected("closed"), response],
            ) as urlopen,
            patch("benchmark_platform.harnesses.api.time.sleep") as sleep,
        ):
            completion = asyncio.run(client.complete_native([{"role": "user", "content": "test"}]))

        self.assertEqual(completion.content, "done")
        self.assertEqual(completion.transport_retries, 1)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()


class ReasoningEffortTests(unittest.TestCase):
    """A matched control must send the same reasoning knob the product harness sends."""

    def _payload(self, config: ApiConfig) -> dict:
        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured.update(json.loads(request.data.decode("utf-8")))
            return _Response({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })

        with patch("urllib.request.urlopen", fake_urlopen):
            asyncio.run(OpenAICompatibleClient(config).complete([{"role": "user", "content": "hi"}]))
        return captured

    def _config(self, **overrides) -> ApiConfig:
        return ApiConfig(base_url="https://provider.example/v1", api_key="k", model="m", **overrides)

    def test_reasoning_effort_replaces_temperature(self) -> None:
        payload = self._payload(self._config(reasoning_effort="high"))
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("temperature", payload)

    def test_temperature_is_sent_when_no_reasoning_effort_is_configured(self) -> None:
        payload = self._payload(self._config(temperature=0.0))
        self.assertEqual(payload["temperature"], 0.0)
        self.assertNotIn("reasoning_effort", payload)

