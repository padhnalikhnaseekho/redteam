---
spec-type: service
title: LLM Client Service
last-updated: 2026-05-15
status: CURRENT
---

# LLM Client Service

**Module:** `src/utils/llm.py`  
**Primary class:** `LLMClient`

---

## Purpose

Unified wrapper around direct LLM provider APIs plus Vertex AI Gemini. Provides cost tracking, per-call latency measurement, normalized response format, and a 120-second timeout mechanism. Used by all V3 agents, defenses, and the generator. `CommodityTradingAgent` still uses LangChain providers directly, but can now resolve Vertex-backed Gemini model keys for target-agent runs.

---

## Interface

```python
client = LLMClient()  # loads config/models.yaml at construction

response = client.chat(
    model_name: str,
    messages: list[dict[str, str]],
    tools: list[dict] | None = None,
    timeout: int = 120,
) -> dict[str, Any]
```

**Response dict:**
```python
{
    "content":       str,
    "tool_calls":    list | None,
    "input_tokens":  int,
    "output_tokens": int,
    "cost_usd":      float,
    "latency_s":     float,
}
```

---

## Provider Adapters

| Provider | Client class | Message handling | Tool support |
|---|---|---|---|
| `openai` | `openai.OpenAI` | Standard `{role, content}` | Yes |
| `anthropic` | `anthropic.Anthropic` | System extracted from messages; filtered | Yes (auto-converted from OpenAI format) |
| `mistral` | `mistralai.client.Mistral` | Standard `{role, content}` | Yes |
| `google` | `google.genai.Client` | System extracted; `types.Content` objects | No (returns `tool_calls: None`) |
| `groq` | `groq.Groq` | OpenAI-compatible | Yes |
| `vertex` | `google.genai.Client(vertexai=True, project=..., location=...)` | System extracted; `types.Content` objects | No in `LLMClient`; use `CommodityTradingAgent` for LangChain tool-calling |

**Anthropic tool conversion:** `_convert_tools_to_anthropic()` converts OpenAI-format tool dicts to Anthropic format (`{name, description, input_schema}`).

---

## Model Configuration

Models are defined in `config/models.yaml`:

| Key | Provider | Model ID | Free? |
|---|---|---|---|
| `claude-sonnet` | anthropic | claude-sonnet-4-20250514 | No ($0.003/$0.015 per 1K tokens) |
| `mistral-large` | mistral | mistral-large-latest | No ($0.002/$0.006 per 1K tokens) |
| `vertex-gemini-flash` | vertex | gemini-2.5-flash | Vertex billing |
| `vertex-gemini-pro` | vertex | gemini-2.5-pro | Vertex billing |
| `gemini-flash` | google | gemini-2.0-flash | Yes ($0.0) |
| `groq-llama` | groq | llama-3.3-70b-versatile | Yes ($0.0) |
| `groq-qwen` | groq | qwen/qwen3-32b | Yes ($0.0) |
| `groq-scout` | groq | meta-llama/llama-4-scout-17b-16e-instruct | Yes ($0.0) |

Model name normalization: `model_name.replace("_", "-")` tried if exact key not found.

---

## Cost Tracking

Every call appends a `UsageRecord` to `client.usage_log`:
```python
UsageRecord(model, input_tokens, output_tokens, cost_usd, latency_s, timestamp)
```

Access aggregates:
```python
client.total_cost   # float: sum of all cost_usd
client.total_tokens  # dict: {"input": int, "output": int}
```

Usage records are NOT persisted to disk.

---

## Timeout Mechanism

Each provider call is wrapped in `ThreadPoolExecutor(max_workers=1)`:
```python
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(_call)
    result = future.result(timeout=timeout)  # default 120s
```

If the provider takes longer than `timeout` seconds, `concurrent.futures.TimeoutError` is raised and propagates to the caller. `LLMClient` does not catch this — the caller must handle it.

---

## Secret Management

Direct provider keys are read at first provider use (lazy initialization):
```python
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
```

`os.environ["KEY"]` raises `KeyError` if missing (not `.get()`). `load_dotenv()` is called at module import time (`src/utils/llm.py` module level).

Vertex models do not use provider API keys. They use Application Default Credentials and require a GCP project from `GOOGLE_CLOUD_PROJECT` unless `project_id` is set on the model config. Each Vertex model entry must define `location`.

---

## Limitations

- Google provider does not support tool calling via `LLMClient`
- Vertex Gemini provider does not support tool calling via `LLMClient`
- Vertex partner model adapters are not implemented yet; only `family: gemini` is supported
- No retry logic for rate limits or transient errors
- No request batching — calls are always sequential
- Token counts from Google use `usage_metadata` attribute which may be 0 if provider doesn't populate it
