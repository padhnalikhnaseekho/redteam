# Parallel Execution as a Defense Against Multi-Step Compounding Attacks (V7)

## Hypothesis

When an agent's tool calls execute in parallel (each in an independent context),
multi-step attack chains (V7) are significantly harder to propagate because each
tool call does not observe the manipulated output of previous calls. We predict a
40-60% reduction in V7 attack success rate (ASR) when switching from sequential
to parallel tool execution.

## Background

### Sequential Execution and Chain Dependency

In a typical agentic workflow the model calls tools one at a time. Each call can
depend on the result of a prior call:

    get_price("brent")          -> $75 (poisoned; actual ~$80)
    calculate_spread(brent=$75) -> WTI-Brent = +$2 (wrong sign)
    submit_trade(spread=+$2)    -> sell Brent, buy WTI (wrong direction)

This is the core mechanism exploited by V7 attacks. A single poisoned data point
at the top of the chain cascades through every downstream calculation. The agent
has no independent reference to catch the error because each step consumes the
output of the step before it.

The original V7 attack suite (v7.1 through v7.6) demonstrates six flavours of
this pattern, from price-to-spread chains to context-overflow-to-safety-bypass.

### Parallel Execution Breaks the Chain

If the agent's tool calls run concurrently in isolated threads, no tool call can
observe another's result:

    get_price("brent")      -> $75 (poisoned)     --|
    get_correlation(Cu, Al) -> +0.75 (clean)       --|--> merge -> cross-validate
    get_fundamentals(brent) -> neutral (clean)     --|

After all calls complete, results are merged and checked for consistency. A
poisoned price that contradicts neutral fundamentals produces a detectable
inconsistency signal. The agent (or a defense layer) can flag the divergence
before acting.

## Methodology

### Attack Corpus

We use 12 V7 attacks total:

| ID      | Name                           | Chain Type                          |
|---------|--------------------------------|-------------------------------------|
| v7.1    | Price -> Spread -> Trade       | Price chain                         |
| v7.2    | Correlation -> Anti-Hedge      | Correlation chain                   |
| v7.3    | News -> Analysis -> Trade      | News + fundamentals chain           |
| v7.4    | Timeout -> Assumed Price       | Error-handling chain                |
| v7.5    | Context Overflow -> Bypass     | Context-window chain                |
| v7.6    | Confidence -> Skip Review      | Confidence chain                    |
| v7.p1   | Price -> Spread (parallel)     | Same as v7.1, explicit parallel test|
| v7.p2   | News -> Fundamentals (parallel)| News bias chain                     |
| v7.p3   | Correlation -> Hedge (parallel)| Correlation chain (gold-silver)     |
| v7.p4   | Risk -> Position (parallel)    | VaR understatement chain            |
| v7.p5   | All Tools Poisoned             | Full compound chain                 |
| v7.p6   | Feedback Loop                  | Self-reinforcement chain            |

### Execution Modes

Each attack runs twice per model:

1. **Sequential mode** -- standard single-threaded tool execution where each
   tool call receives prior results as context.
2. **Parallel mode** -- tool calls dispatched via `ParallelExecutionDefense`
   using `concurrent.futures.ThreadPoolExecutor`. Each thread receives only
   the original user query and the tool's own arguments; no prior results are
   visible. After completion the defense layer cross-validates results and
   flags inconsistencies.

### Target Models

- `groq-llama` (Llama 3 70B via Groq)
- `mistral-large` (Mistral Large via Mistral API)

### Metrics

- **ASR (Attack Success Rate)**: fraction of attacks that achieve their target
  action, per mode per model.
- **ASR Reduction**: `ASR_sequential - ASR_parallel`.
- **Consistency Flags**: number of cross-validation flags raised by the parallel
  defense.
- **Financial Impact Delta**: difference in estimated dollar impact between
  sequential and parallel modes.

## Expected Results

Based on the structure of V7 attacks, we expect:

| Category                         | Expected ASR Reduction |
|----------------------------------|------------------------|
| Price/spread chain attacks       | 50-60%                 |
| Correlation-based chain attacks  | 40-55%                 |
| News-to-fundamentals chains      | 30-50%                 |
| All-tools-poisoned (v7.p5)       | 60%+ (cross-validation)|
| Context overflow (v7.5)          | <10% (not chain-dependent)|
| Confidence/skip-review (v7.6)    | <15% (not chain-dependent)|

The overall ASR reduction should fall in the 40-60% range when averaged across
all 12 attacks and both models.

Attacks that do not rely on tool-to-tool data flow (v7.5 context overflow and
v7.6 confidence manipulation) are expected to show little or no improvement,
since they exploit the prompt context rather than tool-call chaining.

## Connection to Option 7 (CACS / Parallelization)

This experiment directly tests one component of Capstone Option 7:
*Context-Aware Coordination and Supervision (CACS)*. The CACS framework proposes
that agent safety can be improved by restructuring how tool calls are scheduled
and supervised:

- **Parallelization** reduces the attack surface for chain-dependent exploits.
- **Cross-validation** after parallel execution adds a consistency check that is
  absent in sequential mode.
- **Isolation** prevents a single poisoned data source from contaminating the
  entire reasoning chain.

If the experiment confirms the 40-60% ASR reduction, parallel execution becomes
a strong candidate for the first layer in a defence-in-depth strategy, combined
with input filtering (D1), output validation (D2), and system-prompt hardening
(D3).

## How This Fits Into the Broader Red Teaming Project

The red teaming framework evaluates seven attack categories (V1-V7) against
five existing defenses (D1-D5). This experiment adds D6 (Parallel Execution)
and provides the first systematic measurement of how execution scheduling
affects attack success.

Results feed into the overall benchmark matrix:

    Attack (V1-V7) x Defense (D1-D6) x Model (groq-llama, mistral-large, ...)

The parallel experiment specifically enriches the V7 column, which previously
had no structural defense -- only prompt-level mitigations.

## Limitations

1. **Not all V7 attacks are chain-dependent.** Context overflow (v7.5) and
   confidence manipulation (v7.6) exploit the prompt window rather than
   tool-call ordering. Parallel execution offers no protection here.

2. **Parallel execution increases latency and cost.** Running all tools
   concurrently means more API calls (planning step + tool calls + final
   synthesis). Wall-clock time may decrease but token usage increases.

3. **Some chains are unavoidable.** If the agent genuinely needs output from
   tool A as input to tool B (e.g., needing the price before calculating VaR),
   those calls cannot be parallelised without providing stale or default data.

4. **Cross-validation is heuristic.** The consistency checks compare
   directional signals (bullish vs bearish) across tool outputs. Subtle
   manipulations that preserve directional consistency may evade detection.

5. **LLM behaviour is non-deterministic.** Even with temperature=0, model
   outputs vary between runs. Results should be averaged over multiple trials
   for statistical significance.

6. **Simulation vs real tool calls.** In this experiment, tool calls return
   mock or overridden data. In production, parallel execution would involve
   actual API calls, introducing real concurrency issues (rate limits,
   partial failures, etc.).
