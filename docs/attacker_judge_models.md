# Attacker / judge LLM defaults — v9 PAIR, v11 AutoDAN, v12 Crescendo

> Where to look when the iterative attacks (v9 / v11 / v12) seem to be
> silently falling back to precomputed prompts instead of running their
> live LLM-vs-LLM search.

## Default model

| Attack | Internal role | Config field | Default |
|---|---|---|---|
| v9 PAIR | attacker (prompt refiner) | `PAIRConfig.attacker_model` | `vertex-gemini-flash` |
| v9 PAIR | judge (scores target response) | `PAIRConfig.judge_model` | `vertex-gemini-flash` |
| v11 AutoDAN | attacker (genetic mutator) | `AutoDANConfig.attacker_model` | `vertex-gemini-flash` |
| v11 AutoDAN | judge | `AutoDANConfig.judge_model` | `vertex-gemini-flash` |
| v12 Crescendo | strategist (next-turn planner) | `CrescendoConfig.strategist_model` | `vertex-gemini-flash` |
| v12 Crescendo | judge | `CrescendoConfig.judge_model` | `vertex-gemini-flash` |

**Why `vertex-gemini-flash`:** it's fast, free on the GCP project, and
authenticated by workload identity on Cloud Run — no API key in Secret
Manager required. Gemini 2.5 Flash also has strong instruction-following
for the role-play attacker prompts and consistent 1–10 integer outputs
for the judge.

These defaults can be overridden via the dataclass constructor or via the
`--attacker-model` / `--judge-model` / `--strategist-model` flags on the
offline seeder scripts (`scripts/run_pair_generate.py`,
`scripts/run_autodan_generate.py`, `scripts/run_crescendo_generate.py`).

## Credential resolution

Each runner's `_maybe_load_llm()` accepts **either** of two credential
sources, in this order:

1. **Groq** — if `GROQ_API_KEY` is set in the environment, use it. Pre-existing
   laptop flow; nothing changes for users who had this configured.
2. **Vertex AI** — otherwise, probe Application Default Credentials via
   `google.auth.default()`. Resolves automatically against:
   - workload identity on Cloud Run (the deployed worker path)
   - the file written by `gcloud auth application-default login` (laptop dev)

If neither resolves, the runner returns `None` and the attack falls through
to its `PRECOMPUTED_*` offline path — same as before.

## Behaviour matrix

| Environment | Creds present | What runs | Notes |
|---|---|---|---|
| Cloud Run worker | Vertex (workload identity) | Live iterative attack via Gemini Flash | The cloud-native path. No Secret Manager entry needed. |
| Laptop with `gcloud auth ADC` only | Vertex | Live iterative attack via Gemini Flash | Recommended dev setup. |
| Laptop with `GROQ_API_KEY` only | Groq | Live iterative attack via Groq Llama | Backward compat — pre-existing dev setup keeps working. |
| Laptop with both | Groq (first match wins) | Live iterative attack via Groq Llama | Set `--attacker-model vertex-gemini-flash` to force Vertex. |
| CI / no creds | (none) | Offline `PRECOMPUTED_*` fallback | Tests cover this path via monkeypatched `_maybe_load_llm`. |

## When to override the default

- **A/B-testing attacker strength**: try `vertex-gemini-pro` for the attacker
  (slower, more capable) against the same target/judge for ablation.
- **Cost-sensitive runs**: stay on `groq-llama` if Groq free-tier suits and
  you don't want Vertex billing on the project.
- **Judge-attacker decoupling**: deliberately mismatch attacker and judge
  (e.g., `attacker_model=vertex-gemini-flash, judge_model=vertex-gemini-pro`)
  to reduce self-grading bias in PAIR.

## Tests

- `tests/test_attacks.py::TestV9PAIR` (9 tests)
- `tests/test_attacks.py::TestV11AutoDAN` (11 tests)
- `tests/test_attacks.py::TestV12Crescendo` (10 tests)

All three test classes monkeypatch `_maybe_load_llm` to return `None` in their
autouse fixtures — the dual-provider logic above is therefore exercised via
direct import + default-inspection smoke checks rather than via the offline
test path.

## Cloud Run deployment note

The deployed Cloud Run worker (`redteam-demo-worker`) needs no extra
configuration beyond what the existing Terraform sets up — `GOOGLE_CLOUD_PROJECT`
is already exported on the worker, and workload identity grants Vertex AI
access via the worker service account. The first cloud benchmark after this
PR merges will run v9/v11/v12 live against Vertex Gemini Flash automatically.

If you ever want the worker to use Groq instead (for cost comparison or as
a fallback), add `GROQ_API_KEY` to Secret Manager and mount it on the
worker — Groq takes precedence over Vertex in `_maybe_load_llm()`.
