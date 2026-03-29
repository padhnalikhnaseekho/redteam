#!/usr/bin/env python3
"""Generate a minimal LinkedIn carousel PDF from CACS experiment results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "cacs_linkedin_carousel_minimal.pdf"

# Colors — minimal palette
DARK = "#1a1a1a"
GRAY = "#888888"
LIGHT_GRAY = "#cccccc"
GREEN = "#22863a"
RED = "#cb2431"
BLUE = "#0366d6"


def new_slide():
    fig = plt.figure(figsize=(10, 5.625), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.625)
    ax.set_facecolor("white")
    ax.axis("off")
    return fig, ax


def slide_title(pages):
    fig, ax = new_slide()

    ax.text(5, 3.8, "Can Domain Knowledge Make", fontsize=28, fontweight="bold",
            color=DARK, ha="center")
    ax.text(5, 3.0, "Multi-GPU Inference Smarter?", fontsize=28, fontweight="bold",
            color=DARK, ha="center")
    ax.text(5, 2.0, "Correlation-Aware Commodity Sharding (CACS)", fontsize=14,
            color=GRAY, ha="center")
    ax.text(5, 1.3, "2x T4 GPUs  |  Kaggle  |  $0 cost  |  3 min runtime", fontsize=11,
            color=LIGHT_GRAY, ha="center")

    pages.savefig(fig)
    plt.close(fig)


def slide_problem(pages):
    fig, ax = new_slide()

    ax.text(0.6, 4.8, "The Problem", fontsize=22, fontweight="bold", color=DARK)

    lines = [
        "Scaling commodity forecasting across GPUs requires splitting workload.",
        "Random sharding puts correlated commodities on different GPUs.",
        "This creates cross-GPU data transfers that slow everything down.",
    ]
    y = 3.8
    for line in lines:
        ax.text(0.9, y, line, fontsize=12, color=DARK)
        y -= 0.6

    ax.text(0.9, 1.4, "Random sharding made 2-GPU inference", fontsize=14, color=DARK)
    ax.text(0.9, 0.8, "13% slower than a single GPU.", fontsize=14, fontweight="bold", color=RED)

    pages.savefig(fig)
    plt.close(fig)


def slide_idea(pages):
    fig, ax = new_slide()

    ax.text(0.6, 4.8, "The Idea", fontsize=22, fontweight="bold", color=DARK)

    ax.text(0.9, 3.9, "Use price correlations to decide which commodities share a GPU.", fontsize=13, color=DARK)

    steps = [
        "1.  Compute correlation matrix from 2 years of daily returns",
        "2.  Cluster commodities by correlation distance",
        "3.  Assign each cluster to a GPU",
    ]
    y = 3.0
    for s in steps:
        ax.text(0.9, y, s, fontsize=12, color=DARK)
        y -= 0.55

    ax.text(0.9, 1.2, "GPU 0:", fontsize=12, fontweight="bold", color=DARK)
    ax.text(2.3, 1.2, "natural_gas, zinc, brent, wti, coal", fontsize=12, color=GREEN)
    ax.text(0.9, 0.7, "GPU 1:", fontsize=12, fontweight="bold", color=DARK)
    ax.text(2.3, 0.7, "gold, iron_ore, copper, aluminum", fontsize=12, color=BLUE)

    pages.savefig(fig)
    plt.close(fig)


def slide_correlation(pages):
    fig, ax = new_slide()

    ax.text(0.6, 4.8, "Correlation Matrix", fontsize=22, fontweight="bold", color=DARK)

    img_path = SCRIPT_DIR / "chart_cell_8.png"
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        img_ax = fig.add_axes([0.05, 0.02, 0.58, 0.82])
        img_ax.imshow(img)
        img_ax.axis("off")

    ax.text(7.0, 3.8, "Brent - WTI:  0.96", fontsize=11, color=DARK)
    ax.text(7.0, 3.3, "Brent - Coal:  0.34", fontsize=11, color=DARK)
    ax.text(7.0, 2.8, "Copper - Aluminum:  0.45", fontsize=11, color=DARK)
    ax.text(7.0, 2.3, "Gold - Brent:  0.04", fontsize=11, color=GRAY)

    ax.text(7.0, 1.4, "Strong clusters exist.", fontsize=12, fontweight="bold", color=DARK)
    ax.text(7.0, 0.9, "CACS exploits them.", fontsize=12, color=GRAY)

    pages.savefig(fig)
    plt.close(fig)


def slide_dendrogram(pages):
    fig, ax = new_slide()

    ax.text(0.6, 4.8, "Clustering Dendrogram", fontsize=22, fontweight="bold", color=DARK)

    img_path = SCRIPT_DIR / "chart_cell_10.png"
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        img_ax = fig.add_axes([0.03, 0.02, 0.94, 0.8])
        img_ax.imshow(img)
        img_ax.axis("off")

    pages.savefig(fig)
    plt.close(fig)


def slide_communication(pages):
    fig, ax = new_slide()

    ax.text(0.6, 4.8, "Communication Analysis", fontsize=22, fontweight="bold", color=DARK)

    ax.text(0.9, 3.9, "Cross-GPU transfers when correlated features are needed:", fontsize=12, color=GRAY)

    # Simple table — no borders
    col_x = [1.5, 4.5, 7.5]
    headers = ["Strategy", "Transfers", "Rate"]
    for i, h in enumerate(headers):
        ax.text(col_x[i], 3.2, h, fontsize=11, fontweight="bold", color=DARK, ha="center")

    ax.plot([0.6, 9.4], [3.0, 3.0], color=LIGHT_GRAY, linewidth=0.5)

    rows = [
        ("CACS", "0", "0%", GREEN),
        ("Random", "8", "50%", RED),
        ("Round-Robin", "10", "62.5%", RED),
    ]
    y = 2.5
    for name, transfers, rate, color in rows:
        ax.text(col_x[0], y, name, fontsize=12, color=DARK, ha="center")
        ax.text(col_x[1], y, transfers, fontsize=12, color=color, ha="center", fontweight="bold")
        ax.text(col_x[2], y, rate, fontsize=12, color=color, ha="center", fontweight="bold")
        y -= 0.55

    ax.text(0.9, 0.7, "CACS: zero cross-GPU transfers.", fontsize=14, fontweight="bold", color=GREEN)

    pages.savefig(fig)
    plt.close(fig)


def slide_benchmark(pages):
    fig, ax = new_slide()

    ax.text(0.6, 4.8, "Benchmark Results", fontsize=22, fontweight="bold", color=DARK)

    img_path = SCRIPT_DIR / "chart_cell_21.png"
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        img_ax = fig.add_axes([0.02, 0.02, 0.96, 0.82])
        img_ax.imshow(img)
        img_ax.axis("off")

    pages.savefig(fig)
    plt.close(fig)


def slide_numbers(pages):
    fig, ax = new_slide()

    ax.text(0.6, 4.8, "CACS vs Random Sharding", fontsize=22, fontweight="bold", color=DARK)

    metrics = [
        ("38%", "less transfer time"),
        ("19%", "lower latency"),
        ("23%", "higher throughput"),
    ]

    for i, (num, label) in enumerate(metrics):
        cx = 1.8 + i * 3
        ax.text(cx, 3.2, num, fontsize=40, fontweight="bold", color=GREEN, ha="center")
        ax.text(cx, 2.4, label, fontsize=13, color=DARK, ha="center")

    ax.text(5, 1.0, "2x T4 GPUs  |  Kaggle free tier  |  $0", fontsize=11, color=LIGHT_GRAY, ha="center")

    pages.savefig(fig)
    plt.close(fig)


def slide_takeaways(pages):
    fig, ax = new_slide()

    ax.text(0.6, 4.8, "Takeaways", fontsize=22, fontweight="bold", color=DARK)

    takeaways = [
        ("Naive parallelization can hurt.", "Random sharding was 13% slower than 1 GPU."),
        ("Domain-aware sharding fixes it.", "CACS eliminated all cross-GPU transfers."),
        ("For small workloads, 1 GPU wins.", "CACS shines at 50+ commodities or larger models."),
        ("The algorithm is 6 lines of code.", "scipy hierarchical clustering on correlations."),
    ]

    y = 3.9
    for title, desc in takeaways:
        ax.text(0.9, y, title, fontsize=13, fontweight="bold", color=DARK)
        ax.text(0.9, y - 0.35, desc, fontsize=11, color=GRAY)
        y -= 0.95

    pages.savefig(fig)
    plt.close(fig)


def slide_code(pages):
    fig, ax = new_slide()

    ax.text(0.6, 4.8, "The Entire Algorithm", fontsize=22, fontweight="bold", color=DARK)

    # Code block — dark background, no border
    ax.add_patch(plt.Rectangle((0.6, 1.0), 8.8, 3.2, facecolor="#f6f8fa", edgecolor="none"))

    code = [
        "from scipy.cluster.hierarchy import linkage, fcluster",
        "from scipy.spatial.distance import squareform",
        "",
        "dist = squareform((1 - corr_matrix.abs()).values)",
        "Z = linkage(dist, method='average')",
        "labels = fcluster(Z, n_gpus, criterion='maxclust')",
    ]

    y = 3.8
    for line in code:
        if line == "":
            y -= 0.3
            continue
        color = BLUE if line.startswith("from") else DARK
        ax.text(1.0, y, line, fontsize=11, fontfamily="monospace", color=color)
        y -= 0.4

    ax.text(0.9, 0.5, "6 lines to make multi-GPU inference domain-aware.", fontsize=12, color=GRAY)

    pages.savefig(fig)
    plt.close(fig)


def slide_closing(pages):
    fig, ax = new_slide()

    ax.text(5, 3.6, "If your workload has internal correlations,", fontsize=18,
            color=DARK, ha="center")
    ax.text(5, 2.8, "use them to inform your parallelization strategy.", fontsize=18,
            fontweight="bold", color=DARK, ha="center")

    ax.text(5, 1.5, "Notebook on Kaggle  |  Free  |  Reproducible", fontsize=11,
            color=LIGHT_GRAY, ha="center")
    ax.text(5, 1.0, "Part of: Red Teaming Agentic AI for Commodity Trading", fontsize=10,
            color=LIGHT_GRAY, ha="center")

    pages.savefig(fig)
    plt.close(fig)


def main():
    print("Generating minimal carousel PDF...")
    with PdfPages(str(OUTPUT)) as pages:
        slide_title(pages)
        slide_problem(pages)
        slide_idea(pages)
        slide_correlation(pages)
        slide_dendrogram(pages)
        slide_communication(pages)
        slide_benchmark(pages)
        slide_numbers(pages)
        slide_takeaways(pages)
        slide_code(pages)
        slide_closing(pages)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
