---
sdlc-version: "1.0"
stack: custom
sdd-framework: "~/repos/ai-spec-driven-framework"
---

# SDLC-PROFILE — Generic / Custom Stack

> **Read by SDD agents on every session start via SessionStart hook.**
> Fill in all sections to configure SDD for your tech stack.

---

## Source Paths

- app: src/
- protected-paths:
- tests: src/
- migrations:

## Spec Gate

- feature-path: specs/features/active/
- approved-status: SPEC-APPROVED
- in-progress-status: IN-PROGRESS

## Spec Files

- cross-cutting: specs/system/CROSS-CUTTING-PATTERNS.md
- architecture: specs/system/ARCHITECTURE.md
- constitution: specs/system/CONSTITUTION.md
- data-models: specs/system/DATA-MODELS.md
- infrastructure: specs/system/INFRASTRUCTURE.md
- feature-template: specs/features/TEMPLATE.md

## Tech Stack Summary

> Filled in by codebase-spec-extractor on 2026-05-15.
> These are used by generic agents to apply correct patterns quickly.

- language: Python 3.10+ (uses union type syntax `X | Y`; `from __future__ import annotations` in most modules)
- database: none — flat CSV/JSON files via pandas; no SQL or NoSQL backend
- test-framework: pytest (>=7.4); test root is tests/ (exempt from spec gate)
- logger-pattern: standard library `logging` module — `logger = logging.getLogger(__name__)` at module level; NOT print statements (though some scripts use `rich` for console output)
- config-pattern: PyYAML `yaml.safe_load()` — raw dicts, no Pydantic validation on YAML; resolved via `Path(__file__).resolve().parents[N]` so cwd-independent; loaded lazily at class init time
- auth-pattern: none — API keys from environment variables via `python-dotenv`; keys read directly from `os.environ["KEY"]` inside `LLMClient._get_client()`
- result-pattern: flat dict rows appended to `list[dict]`, converted to `pd.DataFrame`, written as CSV+JSON pairs via `pandas.to_csv()` / `pandas.to_json()`; all result columns documented in specs/system/DATA-MODELS.md
- error-pattern: try/except with fallback return values in agent/evaluator code; `AttackResult` with `success=False` and error message in `agent_output` field on exception; no typed exception hierarchy
