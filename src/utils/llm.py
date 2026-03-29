from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    timestamp: float


@dataclass
class LLMClient:
    """Unified wrapper around OpenAI, Anthropic, and Mistral chat APIs."""

    _config: dict[str, Any] = field(default_factory=dict, repr=False)
    _clients: dict[str, Any] = field(default_factory=dict, repr=False)
    usage_log: list[UsageRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        with open(CONFIG_PATH) as f:
            self._config = yaml.safe_load(f)["models"]

    # ------------------------------------------------------------------
    # Client initialisation (lazy)
    # ------------------------------------------------------------------

    def _get_client(self, provider: str) -> Any:
        if provider in self._clients:
            return self._clients[provider]

        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif provider == "mistral":
            from mistralai.client import Mistral
            client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        elif provider == "google":
            from google import genai
            client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        elif provider == "groq":
            from groq import Groq
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._clients[provider] = client
        return client

    def _model_cfg(self, model_name: str) -> dict[str, Any]:
        # Normalize: accept underscores or hyphens (groq_llama -> groq-llama)
        if model_name not in self._config:
            normalized = model_name.replace("_", "-")
            if normalized in self._config:
                model_name = normalized
        if model_name not in self._config:
            raise ValueError(
                f"Model '{model_name}' not found in config. "
                f"Available: {list(self._config.keys())}"
            )
        return self._config[model_name]

    # ------------------------------------------------------------------
    # Core chat
    # ------------------------------------------------------------------

    def chat(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request and return a normalised response dict.

        Returns:
            {
                "content": str,
                "tool_calls": list | None,
                "input_tokens": int,
                "output_tokens": int,
                "cost_usd": float,
                "latency_s": float,
            }
        """
        cfg = self._model_cfg(model_name)
        provider = cfg["provider"]
        client = self._get_client(provider)

        t0 = time.time()

        if provider == "openai":
            result = self._chat_openai(client, cfg, messages, tools)
        elif provider == "anthropic":
            result = self._chat_anthropic(client, cfg, messages, tools)
        elif provider == "mistral":
            result = self._chat_mistral(client, cfg, messages, tools)
        elif provider == "google":
            result = self._chat_google(client, cfg, messages, tools)
        elif provider == "groq":
            result = self._chat_groq(client, cfg, messages, tools)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        latency = time.time() - t0
        cost = self._compute_cost(cfg, result["input_tokens"], result["output_tokens"])
        result["latency_s"] = round(latency, 3)
        result["cost_usd"] = cost

        self.usage_log.append(
            UsageRecord(
                model=model_name,
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                cost_usd=cost,
                latency_s=result["latency_s"],
                timestamp=t0,
            )
        )
        return result

    def chat_with_tools(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        tools: list[dict],
    ) -> dict[str, Any]:
        return self.chat(model_name, messages, tools=tools)

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _chat_openai(
        self,
        client: Any,
        cfg: dict,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            model=cfg["model_id"],
            messages=messages,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return {
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }

    def _chat_anthropic(
        self,
        client: Any,
        cfg: dict,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        # Anthropic expects system prompt separated from messages
        system_text = ""
        filtered_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                filtered_messages.append(m)

        kwargs: dict[str, Any] = dict(
            model=cfg["model_id"],
            messages=filtered_messages,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
        if system_text.strip():
            kwargs["system"] = system_text.strip()
        if tools:
            kwargs["tools"] = self._convert_tools_to_anthropic(tools)

        resp = client.messages.create(**kwargs)

        content_text = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "function": {
                            "name": block.name,
                            "arguments": block.input,
                        },
                    }
                )

        return {
            "content": content_text,
            "tool_calls": tool_calls if tool_calls else None,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }

    def _chat_mistral(
        self,
        client: Any,
        cfg: dict,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            model=cfg["model_id"],
            messages=messages,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = client.chat.complete(**kwargs)
        msg = resp.choices[0].message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return {
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }

    def _chat_google(
        self,
        client: Any,
        cfg: dict,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        from google.genai import types

        # Build contents from messages
        system_text = ""
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

        config = types.GenerateContentConfig(
            temperature=cfg["temperature"],
            max_output_tokens=cfg["max_tokens"],
        )
        if system_text.strip():
            config.system_instruction = system_text.strip()

        resp = client.models.generate_content(
            model=cfg["model_id"],
            contents=contents,
            config=config,
        )

        content_text = resp.text or ""
        input_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0) if resp.usage_metadata else 0
        output_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0) if resp.usage_metadata else 0

        return {
            "content": content_text,
            "tool_calls": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _chat_groq(
        self,
        client: Any,
        cfg: dict,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        # Groq uses OpenAI-compatible API
        kwargs: dict[str, Any] = dict(
            model=cfg["model_id"],
            messages=messages,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return {
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools_to_anthropic(tools: list[dict]) -> list[dict]:
        """Convert OpenAI-style tool definitions to Anthropic format."""
        converted = []
        for tool in tools:
            fn = tool.get("function", tool)
            converted.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return converted

    @staticmethod
    def _compute_cost(cfg: dict, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1000) * cfg.get("cost_per_1k_input_tokens", 0)
        output_cost = (output_tokens / 1000) * cfg.get("cost_per_1k_output_tokens", 0)
        return round(input_cost + output_cost, 6)

    @property
    def total_cost(self) -> float:
        return round(sum(r.cost_usd for r in self.usage_log), 6)

    @property
    def total_tokens(self) -> dict[str, int]:
        return {
            "input": sum(r.input_tokens for r in self.usage_log),
            "output": sum(r.output_tokens for r in self.usage_log),
        }

    def available_models(self) -> list[str]:
        return list(self._config.keys())
