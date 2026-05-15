# Pass 07 — Cross-Cutting Concerns

**Extraction date:** 2026-05-15  
**Status:** COMPLETE

---

## Logging

| Module | Logger name | Level | Format | Notes |
|---|---|---|---|---|
| `src/agent/trading_agent.py` | `__name__` | `logging.error` for agent failures; `logging.info` for tool mode changes, resets | Unstructured | Uses `logger.error("Agent execution failed: %s", e)` |
| `src/attacks/v8_gcg_adversarial.py` | `__name__` | `logging.info` for GCG progress; `logging.debug` for precomputed suffix use; `logging.warning` for fallback | Unstructured | Logs step, loss, decoded suffix every 50 steps |
| `src/defenses/semantic_filter.py` | `__name__` | `logging.info` for model load | Unstructured | Single info on lazy load |
| `src/defenses/perplexity_filter.py` | `__name__` | `logging.info` for model load | Unstructured | Single info on lazy load |
| `src/defenses/ensemble_defense.py` | `__name__` | `logging.info` for save/load | Unstructured | Classifier persistence |
| `src/defenses/multi_agent.py` | None | None | — | No logging; errors surfaced as flags in DefenseResult |
| `src/evaluation/explainability.py` | `__name__` | `logging.info` for SHAP plot path | Unstructured | — |

**Global logging configuration is NOT set by the library code.** Scripts do not call `logging.basicConfig()` — logs may be invisible without consumer configuration.

---

## Error Handling

| Location | Pattern | What's caught | What's propagated |
|---|---|---|---|
| `CommodityTradingAgent.analyze()` | `try/except Exception` | All exceptions from agent_executor | Sets `result.raw_output = f"ERROR: {e}"` |
| `CommodityTradingAgent._create_llm()` | Import inside branches | `ImportError` (not caught explicitly) | Would propagate if provider SDK missing |
| `RedTeamEvaluator.run_single()` | `try/except Exception` | All exceptions from `_run_agent()` | Returns AttackResult with error output |
| `RedTeamEvaluator._apply_tool_overrides()` | No error handling | — | Would propagate if tool module missing |
| `EnsembleDefense._extract_features()` | `try/except Exception` per defense | Any defense exception | Sets features to defaults (0.5, 0, 0.0) |
| `EnsembleDefense.check_output()` | `try/except Exception` per defense | Any defense exception | Silently skips that defense |
| `MultiAgentDefense.check_output()` | `try/except Exception` | API call failures | Returns `DefenseResult(allowed=True, flags=["reviewer_error:..."], confidence=0.3)` — **fail open** |
| `GCGSuffixGenerator.get()` | `try/except Exception` | Online GCG generation failures | Falls back to precomputed suffix |
| `GCGSuffixGenerator._generate_online()` | `try/except ImportError` for configure_http_backend | Missing huggingface_hub version | Silently continues without SSL config |
| `LLMClient.chat()` | `future.result(timeout=timeout)` | `concurrent.futures.TimeoutError` | Propagates to caller (not caught in LLMClient) |
| `scripts/run_auto_redteam_v3.py` | `try/except Exception` per iteration | Any per-round failure | Logs error, continues to next round |
| `CommodityAttackGenerator.generate_batch()` | `try/except Exception` per attack | Generation failures | Appends `{"error": str(e), ...}` to list |

**Critical gap**: `MultiAgentDefense` is fail-open. A network outage or rate limit error causes all outputs to be allowed, silently degrading to no defense.

---

## Configuration Loading

| Config file | Loaded by | How | Path resolution | Called when |
|---|---|---|---|---|
| `config/models.yaml` | `LLMClient.__post_init__()` | `yaml.safe_load()` | `Path(__file__).resolve().parents[2] / "config" / "models.yaml"` | On every `LLMClient()` instantiation |
| `config/agent_config.yaml` | `OutputValidatorDefense.__init__()` | `yaml.safe_load()` | `Path(__file__).resolve().parents[2] / "config" / "agent_config.yaml"` | On every `OutputValidatorDefense()` instantiation |
| `config/agent_config.yaml` | `HumanInLoopDefense.__init__()` | `yaml.safe_load()` | Same relative path | On every `HumanInLoopDefense()` instantiation |
| `config/agent_config.yaml` | `CommodityTradingAgent._load_config()` | `yaml.safe_load()` | `Path(__file__).resolve().parents[3] / "config" / "agent_config.yaml"` | On agent construction |
| `config/commodities.yaml` | `OutputValidatorDefense._load_commodities()` | `yaml.safe_load()` | `Path(__file__).resolve().parents[2] / "config" / "commodities.yaml"` | On every `OutputValidatorDefense()` instantiation |
| `config/commodities.yaml` | `src/utils/data.py` | `yaml.safe_load()` | Same relative path | When `get_commodity_info()` called |
| `.env` | `src/utils/llm.py` (module level) | `load_dotenv()` | Current working directory / default dotenv search | At module import time |

All config paths use `__file__`-relative resolution, making configs portable regardless of CWD. However, if scripts are moved to a different directory level, path depth (`parents[N]`) would need adjustment.

---

## Secrets Management

| Secret | Env var name | Where read | Required for |
|---|---|---|---|
| Anthropic API key | `ANTHROPIC_API_KEY` | `LLMClient._get_client("anthropic")` via `os.environ["ANTHROPIC_API_KEY"]` | `claude-sonnet` model |
| Mistral API key | `MISTRAL_API_KEY` | `LLMClient._get_client("mistral")` | `mistral-large` model |
| Google API key | `GOOGLE_API_KEY` | `LLMClient._get_client("google")` | `gemini-flash` model |
| Groq API key | `GROQ_API_KEY` | `LLMClient._get_client("groq")` | `groq-llama`, `groq-qwen`, `groq-scout` models |
| OpenAI API key | `OPENAI_API_KEY` | `LLMClient._get_client("openai")` | `gpt-4o` (commented out in models.yaml) |
| LangChain provider keys | `ANTHROPIC_API_KEY` etc. | LangChain provider classes (auto-read from env) | `CommodityTradingAgent` |

`load_dotenv()` is called at `src/utils/llm.py` **module import time** (not inside a function). This means the `.env` file must exist at the time the module is first imported.

Keys are accessed with `os.environ["KEY"]` (raises `KeyError` if missing) rather than `os.environ.get("KEY")` (returns None). A missing key will crash at client initialization time, not at import time.

---

## State Mutation (Module-Level Mutable State)

| Location | Mutable state | Modified by | Reset by | Risk |
|---|---|---|---|---|
| `src/attacks/registry.py` | `_REGISTRY: dict` | `@register` decorator at import | Never (intentional) | Safe: write-once at import |
| `src/attacks/registry.py` | `_discovered: bool` | `_auto_discover()` | Never (module reload) | Safe: idempotent guard |
| `src/agent/tools/price.py` | `_tool_state: dict` | `set_mode()` | `reset_mode()` | RISK: shared across all calls; if reset fails, state leaks |
| `src/agent/tools/news.py` | `_tool_state: dict` | `set_mode()` | `reset_mode()` | RISK: same |
| `src/agent/tools/risk.py` | `_tool_state: dict` | `set_mode()` | `reset_mode()` | RISK: same |
| `src/agent/tools/fundamentals.py` | `_tool_state: dict` | `set_mode()` | `reset_mode()` | RISK: same |
| `src/agent/tools/correlation.py` | `_tool_state: dict` | `set_mode()` | `reset_mode()` | RISK: same |
| `src/agent/tools/position.py` | `_tool_state: dict` | `set_mode()` | `reset_mode()` | RISK: same |
| `src/agent/tools/recommendation.py` | `_submitted_recommendations: list` | `submit_recommendation_impl()` | `clear_recommendations()` | RISK: accumulates across attacks if not cleared |
| `src/attacks/v8_gcg_adversarial.py` | `_generator: Optional[GCGSuffixGenerator]` | `_get_generator()` | Never | Design intent: singleton; safe since GCGConfig is fixed |
| `src/v3/trajectory_defense.py` | `_trajectory_buffer: list` (CLASS ATTRIBUTE) | `check_input()`, `check_output()` | `reset()` | CRITICAL RISK: shared across ALL instances |
| `src/defenses/semantic_filter.py` | `_model`, `_injection_embeddings` | `_load_model()` | Never | Design intent: lazy singleton per instance; safe |
| `src/defenses/perplexity_filter.py` | `_model`, `_tokenizer` | `_load_model()` | Never | Design intent: lazy singleton per instance; safe |

**Critical finding**: `TrajectoryDefense._trajectory_buffer` is a **class attribute**, not an instance attribute. All instances share the same buffer. This is almost certainly a bug — it should be `self._trajectory_buffer: list = []` in `__init__`.

---

## Randomness / Seeding

| Location | Usage | Seeded? | Impact |
|---|---|---|---|
| `src/v3/strategy_db.py` | `random.random()` (epsilon check), `random.choice()`, `random.sample()` | No | Strategy selection is non-deterministic |
| `src/v3/attack_archive.py` | `random.sample()` (exploration pool) | No | Archive exploration is non-deterministic |
| `src/defenses/ensemble_defense.py` | `XGBClassifier(random_state=42)` | Yes (42) | Training is reproducible |
| `src/evaluation/explainability.py` | `XGBClassifier(random_state=42)` | Yes (42) | Training is reproducible |
| `scripts/run_auto_redteam_v3.py` | Via StrategyDB (random module) | No | Run results are non-deterministic |

No global `random.seed()` or `numpy.random.seed()` is set anywhere. Benchmark results will vary between runs for V3 loop scripts.
