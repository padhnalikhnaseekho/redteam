---
feature: vertex-ai-cloudrun-readiness
status: IN-PROGRESS
stack: python
created: 2026-05-15
security-review: required
---

# Feature: Vertex AI Cloud Run Readiness

> **Status:** `IN-PROGRESS`

## Requirement

Prepare the existing local CommodityRedTeam framework for the GCP-native deployment path documented in `specs/cloud/GCP-VERTEXAI-CLOUDRUN-IMPLEMENTATION-PLAN.md`.

This first implementation slice adds Vertex AI model configuration and Gemini-on-Vertex runtime support, while fixing benchmark correctness issues that would make deployed results misleading.

## Acceptance Criteria

- [x] `config/models.yaml` includes Vertex-backed Gemini model keys for reviewer and target-agent roles.
- [x] `LLMClient.chat()` supports `provider: vertex` for Gemini through Application Default Credentials, without requiring `GOOGLE_API_KEY`.
- [x] `CommodityTradingAgent` can instantiate a Vertex-backed Gemini LangChain model from a `config/models.yaml` key.
- [x] Existing non-Vertex provider config remains backward compatible.
- [x] Tool overrides used by V3/V7-style attacks are applied and reset across all stateful tool modules.
- [x] `GuardrailsDefense.modified_input` is consumed by the evaluator when possible instead of being silently ignored.
- [x] `TrajectoryDefense` state is instance-owned.
- [x] `MultiAgentDefense` does not silently fail open by default on reviewer API errors.
- [x] Generated attacks use the `Severity` enum instead of a raw severity string.
- [x] Unit tests cover the changed contracts without requiring live provider calls.

## Spec Changes Required

| Spec File | Section | Change Description | Done |
|---|---|---|---|
| `specs/cloud/GCP-VERTEXAI-CLOUDRUN-IMPLEMENTATION-PLAN.md` | All | Establish target GCP-native deployment approach. | [x] |
| `specs/system/INFRASTRUCTURE.md` | Environment Variables / LLM selection | Add Vertex provider and ADC path. | [x] |
| `specs/services/llm-client.md` | Provider Adapters | Add Vertex Gemini provider behavior. | [x] |
| `specs/services/trading-agent.md` | LLM selection | Add Vertex-backed Gemini target-agent support. | [x] |
| `specs/system/API-CONTRACTS.md` | LLMClient / Evaluator | Document provider `vertex`, tool override expansion, and guardrail modified input behavior. | [x] |

## Security Gate

- [x] This feature handles API keys, provider credentials, or secrets.
- [x] This feature changes LLM prompt boundaries, attack payload generation, or defense blocking behavior.
- [x] This feature changes benchmark outputs that may contain adversarial prompts.

Mitigations:

- Vertex AI uses ADC and Cloud Run service identity rather than committed credentials.
- Non-Vertex provider secrets remain environment-only.
- Prompt hardening behavior is scoped to a single evaluator run.
- Reviewer-model failures are surfaced in `DefenseResult.flags`.

## Implementation Plan

### Phase 1 - Contracts

- [x] Add Vertex model config keys.
- [x] Add Vertex provider branch to `LLMClient`.
- [x] Add Vertex target-agent branch to `CommodityTradingAgent`.
- [x] Update evaluator behavior for guardrails and all tool overrides.

### Phase 2 - Implementation

- [x] Framework module changes.
- [x] Unit tests for contract behavior.
- [x] Dependency check confirmed `langchain-google-genai` is the correct LangChain package for Gemini on Vertex via `ChatGoogleGenerativeAI(vertexai=True)`.

### Phase 3 - Verification

- [x] `pytest` or targeted tests run.
- [x] YAML config parse verified.
- [x] Spec-sync audit completed.

## Spec-Sync Audit

| Finding | File | Detail |
|---|---|---|
| Old infrastructure spec still listed direct API keys as required | `specs/system/INFRASTRUCTURE.md` | Revised to add Vertex ADC path and direct provider key fallback. |
| LLM service spec lacked `vertex` provider | `specs/services/llm-client.md` | Revised to document Vertex Gemini provider behavior. |
| Trading-agent spec lacked Vertex LangChain support | `specs/services/trading-agent.md` | Revised to document config-key based Vertex model resolution. |
