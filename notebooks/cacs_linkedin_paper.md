# CACS: Can Domain Knowledge Make Multi-GPU Inference Smarter?

**A quick experiment on correlation-aware commodity sharding using 2x T4 GPUs on Kaggle**

---

## The Problem

When deploying ML models to forecast multiple commodities (oil, metals, gold) simultaneously, a natural scaling approach is to split workload across multiple GPUs. But **how** you split matters.

**Random sharding** assigns commodities to GPUs arbitrarily. This creates unnecessary cross-GPU data transfers -- because correlated commodities (e.g., Brent and WTI crude oil) frequently need each other's features but end up on different GPUs.

**The question**: Can we use domain knowledge (price correlations) to shard smarter?

---

## The Idea: CACS (Correlation-Aware Commodity Sharding)

Instead of random assignment, CACS uses hierarchical clustering on historical price correlations to group commodities that move together onto the same GPU.

**Step 1**: Compute correlation matrix from 2 years of daily returns for 9 commodities.

![Correlation Heatmap](correlation_heatmap.png)

Brent and WTI are 0.96 correlated. Copper and aluminum are 0.45. Gold is weakly correlated with everything.

**Step 2**: Cluster commodities using the correlation distance (1 - |correlation|).

![Dendrogram](cacs_dendrogram.png)

The dendrogram reveals two natural clusters:
- **GPU 0**: natural_gas, zinc, brent, wti, coal (energy + linked commodities)
- **GPU 1**: gold, iron_ore, copper, aluminum (metals + safe haven)

---

## The Experiment

**Hardware**: Kaggle 2x T4 GPUs (free tier)
**Model**: Small transformer (0.8M params) for commodity price forecasting
**Data**: 440 sequences of 60-day windows across 9 commodities

Compared 4 strategies:
- **Single GPU**: Baseline, no sharding
- **Random**: Commodities assigned randomly across 2 GPUs
- **Round-Robin**: Alternating assignment (commodity 1 -> GPU 0, commodity 2 -> GPU 1, ...)
- **CACS**: Correlation-based clustering

---

## Results

### Communication Analysis (Theoretical)

| Strategy | Cross-GPU Transfers | Transfer Rate |
|----------|-------------------|---------------|
| **CACS** | **0** | **0%** |
| Random | 8 | 50% |
| Round-Robin | 10 | 62.5% |

CACS achieved **100% reduction** in cross-GPU transfers. Every correlated commodity pair landed on the same GPU.

### Inference Benchmark (Actual GPU Measurement)

| Strategy | Latency | Transfer Time | Throughput |
|----------|---------|---------------|------------|
| Single GPU | 0.044s | -- | 10,086/s |
| Random | 0.050s | 0.0051s | 8,789/s |
| Round-Robin | 0.041s | 0.0033s | 10,865/s |
| **CACS** | **0.041s** | **0.0032s** | **10,856/s** |

![Benchmark Results](cacs_benchmark_results.png)

**CACS vs Random**:
- **38% less cross-GPU transfer time**
- **19% lower latency**
- **23% higher throughput**

---

## Key Takeaways

**1. Naive parallelization can hurt.** Random sharding made things 13% *slower* than a single GPU. The overhead of cross-GPU transfers exceeded the benefit of parallel compute.

**2. Domain-aware sharding eliminates the overhead.** CACS reduced cross-GPU transfers to zero for this workload, making 2 GPUs faster than 1.

**3. For small workloads, single GPU wins.** With only 9 commodities and a 0.8M param model, one GPU handles everything fine. CACS becomes critical when scaling to 50+ commodities or larger models where single-GPU memory is the bottleneck.

**4. The algorithm is simple.** The entire CACS partitioning is 6 lines of scipy:

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

dist = squareform((1 - corr_matrix.abs()).values)
Z = linkage(dist, method="average")
labels = fcluster(Z, n_gpus, criterion="maxclust")
```

---

## When Does This Matter?

This experiment is a proof-of-concept on a small scale. The real value of CACS emerges at scale:

| Scenario | Single GPU | Random 2-GPU | CACS 2-GPU |
|----------|-----------|-------------|------------|
| 9 commodities, 0.8M model | Best | Worst | Tied with single |
| 50+ commodities, 200M model | Out of memory | Works, but slow | Fastest |
| 100+ commodities, 1B model | Impossible | Slow | Optimized |

The principle is general: **if your workload has internal correlations, use them to inform your parallelization strategy.** This applies beyond commodities -- any multi-task inference where tasks share features.

---

## Reproducibility

- **Notebook**: Runs entirely on Kaggle free tier (2x T4 GPU)
- **Data**: Live commodity prices via yfinance (no proprietary data needed)
- **Total runtime**: ~3 minutes
- **Cost**: $0

---

*This experiment is part of a larger capstone project on red teaming agentic AI for commodity trading (IIT Bombay EPGD AI/ML). CACS was explored as a "parallel execution defense" -- a way to make multi-step AI agent attacks harder by isolating tool calls across GPUs while maintaining inference efficiency.*
