#!/usr/bin/env python3
"""Generate a LinkedIn carousel PDF from CACS experiment results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "cacs_linkedin_carousel.pdf"

# Colors
BG = "#FFFFFF"
ACCENT = "#0A66C2"  # LinkedIn blue
DARK = "#191919"
GRAY = "#666666"
GREEN = "#2ecc71"
RED = "#e74c3c"
ORANGE = "#f39c12"

def new_slide(fig_width=10, fig_height=5.625):
    """Create a new slide (16:9 aspect ratio)."""
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.625)
    ax.set_facecolor(BG)
    ax.axis("off")
    return fig, ax


def slide_title(pages):
    """Slide 1: Title slide."""
    fig, ax = new_slide()

    # Top accent bar
    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)

    ax.text(5, 4.2, "Can Domain Knowledge Make", fontsize=26, fontweight="bold",
            color=DARK, ha="center", va="center")
    ax.text(5, 3.5, "Multi-GPU Inference Smarter?", fontsize=26, fontweight="bold",
            color=ACCENT, ha="center", va="center")

    ax.text(5, 2.5, "CACS: Correlation-Aware Commodity Sharding", fontsize=14,
            color=GRAY, ha="center", va="center", style="italic")

    ax.text(5, 1.5, "A quick experiment on 2x T4 GPUs (Kaggle, $0 cost)", fontsize=12,
            color=GRAY, ha="center", va="center")

    # Bottom bar
    ax.fill_between([0, 10], [0.3, 0.3], [0, 0], color=ACCENT)

    pages.savefig(fig)
    plt.close(fig)


def slide_problem(pages):
    """Slide 2: The problem."""
    fig, ax = new_slide()

    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)
    ax.text(0.5, 5.46, "THE PROBLEM", fontsize=14, fontweight="bold", color="white", va="center")

    ax.text(5, 4.4, "Scaling commodity forecasting across multiple GPUs", fontsize=18,
            fontweight="bold", color=DARK, ha="center")

    lines = [
        ("Random sharding", "assigns commodities to GPUs arbitrarily"),
        ("Correlated commodities", "end up on different GPUs"),
        ("Cross-GPU transfers", "create overhead that can negate parallelism"),
    ]
    y = 3.5
    for title, desc in lines:
        ax.plot(1.2, y, "s", color=RED, markersize=10)
        ax.text(1.6, y, title, fontsize=13, fontweight="bold", color=DARK, va="center")
        ax.text(1.6, y - 0.35, desc, fontsize=11, color=GRAY, va="center")
        y -= 0.9

    # Key stat
    ax.add_patch(plt.Rectangle((1, 0.4), 8, 0.8, facecolor="#FFF3E0", edgecolor=ORANGE, linewidth=2, zorder=2))
    ax.text(5, 0.8, "Random sharding made 2-GPU inference 13% SLOWER than 1 GPU",
            fontsize=13, fontweight="bold", color=RED, ha="center", va="center", zorder=3)

    pages.savefig(fig)
    plt.close(fig)


def slide_idea(pages):
    """Slide 3: The CACS idea."""
    fig, ax = new_slide()

    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)
    ax.text(0.5, 5.46, "THE IDEA: CACS", fontsize=14, fontweight="bold", color="white", va="center")

    ax.text(5, 4.5, "Use price correlations to shard smarter", fontsize=18,
            fontweight="bold", color=DARK, ha="center")

    # Steps
    steps = [
        ("1", "Compute correlation matrix from 2Y daily returns"),
        ("2", "Cluster commodities by correlation distance"),
        ("3", "Assign each cluster to a GPU"),
    ]
    y = 3.6
    for num, text in steps:
        ax.add_patch(plt.Circle((1.3, y), 0.25, facecolor=ACCENT, edgecolor="none", zorder=2))
        ax.text(1.3, y, num, fontsize=12, fontweight="bold", color="white", ha="center", va="center", zorder=3)
        ax.text(1.9, y, text, fontsize=12, color=DARK, va="center")
        y -= 0.7

    # Result boxes
    ax.add_patch(plt.Rectangle((0.5, 0.5), 4, 1.2, facecolor="#E8F5E9", edgecolor=GREEN, linewidth=2))
    ax.text(2.5, 1.35, "GPU 0", fontsize=11, fontweight="bold", color=DARK, ha="center")
    ax.text(2.5, 0.9, "natural_gas, zinc, brent,\nwti, coal", fontsize=10, color=GRAY, ha="center")

    ax.add_patch(plt.Rectangle((5.5, 0.5), 4, 1.2, facecolor="#E3F2FD", edgecolor=ACCENT, linewidth=2))
    ax.text(7.5, 1.35, "GPU 1", fontsize=11, fontweight="bold", color=DARK, ha="center")
    ax.text(7.5, 0.9, "gold, iron_ore, copper,\naluminum", fontsize=10, color=GRAY, ha="center")

    pages.savefig(fig)
    plt.close(fig)


def slide_correlation(pages):
    """Slide 4: Correlation heatmap."""
    fig, ax = new_slide()

    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)
    ax.text(0.5, 5.46, "CORRELATION MATRIX", fontsize=14, fontweight="bold", color="white", va="center")

    # Load and embed the heatmap image
    img_path = SCRIPT_DIR / "chart_cell_8.png"
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        img_ax = fig.add_axes([0.08, 0.05, 0.55, 0.88])
        img_ax.imshow(img)
        img_ax.axis("off")

    # Annotations on the right
    ax.text(7.2, 4.2, "Key Findings:", fontsize=13, fontweight="bold", color=DARK)
    findings = [
        "Brent-WTI: 0.96 corr",
        "Oil cluster is tight",
        "Gold is the outlier",
        "Metals form a group",
    ]
    y = 3.5
    for f in findings:
        ax.text(7.2, y, f"  {f}", fontsize=11, color=GRAY)
        y -= 0.5

    pages.savefig(fig)
    plt.close(fig)


def slide_dendrogram(pages):
    """Slide 5: Dendrogram."""
    fig, ax = new_slide()

    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)
    ax.text(0.5, 5.46, "CACS CLUSTERING", fontsize=14, fontweight="bold", color="white", va="center")

    img_path = SCRIPT_DIR / "chart_cell_10.png"
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        img_ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])
        img_ax.imshow(img)
        img_ax.axis("off")

    pages.savefig(fig)
    plt.close(fig)


def slide_communication(pages):
    """Slide 6: Communication analysis."""
    fig, ax = new_slide()

    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)
    ax.text(0.5, 5.46, "COMMUNICATION ANALYSIS", fontsize=14, fontweight="bold", color="white", va="center")

    ax.text(5, 4.5, "Cross-GPU transfers when commodities need correlated features",
            fontsize=13, color=GRAY, ha="center")

    # Table
    headers = ["Strategy", "Cross-GPU Transfers", "Transfer Rate"]
    data = [
        ["CACS", "0", "0%"],
        ["Random", "8", "50%"],
        ["Round-Robin", "10", "62.5%"],
    ]
    colors_row = [GREEN, RED, ORANGE]

    y = 3.5
    for i, h in enumerate(headers):
        x = 1.5 + i * 2.8
        ax.text(x, y, h, fontsize=12, fontweight="bold", color=DARK, ha="center")

    y = 3.1
    ax.plot([0.5, 9.5], [y, y], color=GRAY, linewidth=0.5)

    for row_idx, row in enumerate(data):
        y -= 0.6
        for i, val in enumerate(row):
            x = 1.5 + i * 2.8
            weight = "bold" if i == 0 else "normal"
            color = colors_row[row_idx] if i > 0 else DARK
            ax.text(x, y, val, fontsize=13, fontweight=weight, color=color, ha="center")

    # Big stat
    ax.add_patch(plt.Rectangle((2, 0.3), 6, 0.9, facecolor="#E8F5E9", edgecolor=GREEN, linewidth=2))
    ax.text(5, 0.75, "CACS: 100% reduction in cross-GPU transfers",
            fontsize=15, fontweight="bold", color=GREEN, ha="center", va="center")

    pages.savefig(fig)
    plt.close(fig)


def slide_benchmark(pages):
    """Slide 7: Benchmark results chart."""
    fig, ax = new_slide()

    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)
    ax.text(0.5, 5.46, "BENCHMARK RESULTS", fontsize=14, fontweight="bold", color="white", va="center")

    img_path = SCRIPT_DIR / "chart_cell_21.png"
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        img_ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
        img_ax.imshow(img)
        img_ax.axis("off")

    pages.savefig(fig)
    plt.close(fig)


def slide_numbers(pages):
    """Slide 8: Key numbers."""
    fig, ax = new_slide()

    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)
    ax.text(0.5, 5.46, "CACS vs RANDOM SHARDING", fontsize=14, fontweight="bold", color="white", va="center")

    # Three big metrics
    metrics = [
        ("38%", "less transfer time", GREEN),
        ("19%", "lower latency", ACCENT),
        ("23%", "higher throughput", ORANGE),
    ]

    for i, (num, label, color) in enumerate(metrics):
        cx = 1.8 + i * 3
        ax.add_patch(plt.Rectangle((cx - 1.2, 2.2), 2.4, 2.5, facecolor=color, edgecolor="none", alpha=0.1, zorder=1))
        ax.add_patch(plt.Rectangle((cx - 1.2, 2.2), 2.4, 2.5, facecolor="none", edgecolor=color, linewidth=2, zorder=2))
        ax.text(cx, 3.9, num, fontsize=36, fontweight="bold", color=color, ha="center", va="center", zorder=3)
        ax.text(cx, 2.8, label, fontsize=13, color=DARK, ha="center", va="center", zorder=3)

    ax.text(5, 1.2, "Hardware: 2x T4 GPUs (Kaggle free tier)  |  Cost: $0  |  Runtime: 3 min",
            fontsize=11, color=GRAY, ha="center")

    pages.savefig(fig)
    plt.close(fig)


def slide_takeaways(pages):
    """Slide 9: Key takeaways."""
    fig, ax = new_slide()

    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)
    ax.text(0.5, 5.46, "KEY TAKEAWAYS", fontsize=14, fontweight="bold", color="white", va="center")

    takeaways = [
        ("Naive parallelization can hurt.", "Random sharding was 13% slower than 1 GPU."),
        ("Domain-aware sharding fixes it.", "CACS eliminated all cross-GPU transfers."),
        ("For small workloads, 1 GPU wins.", "CACS shines at 50+ commodities / larger models."),
        ("The algorithm is 6 lines of code.", "scipy hierarchical clustering on price correlations."),
    ]

    y = 4.3
    for i, (title, desc) in enumerate(takeaways):
        color = [GREEN, ACCENT, ORANGE, ACCENT][i]
        ax.add_patch(plt.Rectangle((0.8, y - 0.35), 0.06, 0.7, facecolor=color, edgecolor="none"))
        ax.text(1.2, y + 0.05, title, fontsize=13, fontweight="bold", color=DARK, va="center")
        ax.text(1.2, y - 0.3, desc, fontsize=11, color=GRAY, va="center")
        y -= 1.0

    pages.savefig(fig)
    plt.close(fig)


def slide_code(pages):
    """Slide 10: The code."""
    fig, ax = new_slide()

    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)
    ax.text(0.5, 5.46, "THE ENTIRE ALGORITHM", fontsize=14, fontweight="bold", color="white", va="center")

    # Code block
    ax.add_patch(plt.Rectangle((0.5, 1.0), 9, 3.8, facecolor="#1E1E1E", edgecolor=GRAY, linewidth=1))

    code = [
        "from scipy.cluster.hierarchy import linkage, fcluster",
        "from scipy.spatial.distance import squareform",
        "",
        "dist = squareform((1 - corr_matrix.abs()).values)",
        "Z = linkage(dist, method='average')",
        "labels = fcluster(Z, n_gpus, criterion='maxclust')",
    ]

    y = 4.3
    for line in code:
        color = "#569CD6" if line.startswith("from") or line.startswith("import") else "#D4D4D4"
        if "=" in line and not line.startswith("from"):
            color = "#9CDCFE"
        if line == "":
            y -= 0.35
            continue
        ax.text(1.0, y, line, fontsize=11, fontfamily="monospace", color=color, va="center")
        y -= 0.45

    ax.text(5, 0.5, "That's it. 6 lines to make multi-GPU inference domain-aware.",
            fontsize=12, color=GRAY, ha="center", style="italic")

    pages.savefig(fig)
    plt.close(fig)


def slide_closing(pages):
    """Slide 11: Closing."""
    fig, ax = new_slide()

    ax.fill_between([0, 10], [5.625, 5.625], [5.3, 5.3], color=ACCENT)

    ax.text(5, 3.8, "If your workload has internal correlations,", fontsize=18,
            color=DARK, ha="center", va="center")
    ax.text(5, 3.0, "use them to inform your parallelization strategy.", fontsize=18,
            fontweight="bold", color=ACCENT, ha="center", va="center")

    ax.text(5, 1.8, "Notebook: Kaggle (free)  |  Runtime: 3 min  |  Cost: $0",
            fontsize=12, color=GRAY, ha="center")

    ax.text(5, 1.1, "Part of: Red Teaming Agentic AI for Commodity Trading",
            fontsize=11, color=GRAY, ha="center", style="italic")
    ax.text(5, 0.7, "IIT Bombay EPGD AI/ML Capstone",
            fontsize=11, color=GRAY, ha="center", style="italic")

    ax.fill_between([0, 10], [0.3, 0.3], [0, 0], color=ACCENT)

    pages.savefig(fig)
    plt.close(fig)


def main():
    print("Generating LinkedIn carousel PDF...")

    with PdfPages(str(OUTPUT)) as pages:
        slide_title(pages)        # 1
        slide_problem(pages)      # 2
        slide_idea(pages)         # 3
        slide_correlation(pages)  # 4
        slide_dendrogram(pages)   # 5
        slide_communication(pages)# 6
        slide_benchmark(pages)    # 7
        slide_numbers(pages)      # 8
        slide_takeaways(pages)    # 9
        slide_code(pages)         # 10
        slide_closing(pages)      # 11

    print(f"Saved: {OUTPUT}")
    print(f"11 slides, ready to upload to LinkedIn as a document post.")


if __name__ == "__main__":
    main()
