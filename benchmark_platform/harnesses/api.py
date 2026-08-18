from __future__ import annotations

import asyncio
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout_seconds: float = 180.0
    transport_retries: int = 3
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls) -> "ApiConfig":
        base_url = os.getenv("HARNESS_API_BASE", "").strip()
        api_key = os.getenv("HARNESS_API_KEY", "").strip()
        model = os.getenv("HARNESS_MODEL", "").strip()
        if not base_url or not api_key or not model:
            raise RuntimeError("Set HARNESS_API_BASE, HARNESS_API_KEY, and HARNESS_MODEL")
        raw_max = os.getenv("HARNESS_MAX_OUTPUT_TOKENS", "").strip()
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            temperature=float(os.getenv("HARNESS_TEMPERATURE", "0")),
            timeout_seconds=float(os.getenv("HARNESS_API_TIMEOUT_S", "180")),
            transport_retries=max(0, int(os.getenv("HARNESS_API_RETRIES", "3"))),
            max_output_tokens=int(raw_max) if raw_max else None,
            reasoning_effort=os.getenv("HARNESS_REASONING_EFFORT", "").strip() or None,
        )


@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    transport_retries: int
    raw: dict[str, Any]


class OpenAICompatibleClient:
    def __init__(self, config: ApiConfig):
        self.config = config

    @property
    def endpoint(self) -> str:
        if self.config.base_url.endswith("/chat/completions"):
            return self.config.base_url
        return f"{self.config.base_url}/chat/completions"

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        return await asyncio.to_thread(
            self._complete_sync,
            messages,
            temperature=temperature,
            json_mode=json_mode,
        )

    async def complete_native(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
    ) -> Completion:
        """Preserve native chat/tool messages for benchmark-owned simulators."""
        return await asyncio.to_thread(
            self._complete_sync,
            messages,
            temperature=temperature,
            json_mode=False,
            tools=tools,
            tool_choice=tool_choice,
        )

    def _complete_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None,
        json_mode: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
        }
        # A reasoning model rejects or ignores temperature; the product harness omits it for
        # the same reason, so a matched control must omit it here too.
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        else:
            payload["temperature"] = self.config.temperature if temperature is None else temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if self.config.max_output_tokens is not None:
            payload["max_tokens"] = self.config.max_output_tokens
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        retries = 0
        while True:
            request = urllib.request.Request(
                self.endpoint,
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = response.read()
                break
            except urllib.error.HTTPError as exc:
                body = exc.read()
                if exc.code != 429 and exc.code < 500:
                    raise RuntimeError(f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}") from exc
                if retries >= self.config.transport_retries:
                    raise RuntimeError(f"API HTTP {exc.code}: {body.decode('utf-8', errors='replace')}") from exc
            # http.client.RemoteDisconnected subclasses ConnectionResetError and
            # BadStatusLine, neither of which is a urllib.error.URLError, so it escaped
            # this handler and killed the episode outright. Large tool-schema payloads
            # (VitaBench cross-domain sends 66 schemas) make such transient disconnects
            # routine, which is precisely what the retry budget exists for.
            except (TimeoutError, urllib.error.URLError, http.client.HTTPException, ConnectionError) as exc:
                if retries >= self.config.transport_retries:
                    raise RuntimeError(
                        f"API transport failed after {retries} retries: {type(exc).__name__}: {exc}"
                    ) from exc
            retries += 1
            time.sleep(2 ** (retries - 1))

        raw = json.loads(body.decode("utf-8"))
        message = raw["choices"][0]["message"]
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        usage = raw.get("usage") or {}
        return Completion(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            elapsed_seconds=time.perf_counter() - started,
            transport_retries=retries,
            raw=raw,
        )
