"""Create matched, illustrative market-failure diagrams for an SS111 presentation.

Run: python ai_market_failure_graphs.py
Outputs are written beside this script in the ``output`` directory.
"""

import os
from pathlib import Path

# Keep font and Matplotlib caches writable in restricted environments.
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a desktop display.

import matplotlib.pyplot as plt
import numpy as np


# Illustrative marginal schedules: intercept + slope * quantity.
DEMAND = (100.0, -0.6)  # MPB = MSB
MPC = (20.0, 0.4)
MSC = (40.0, 0.4)

BLUE = "#1764A0"
GRAY = "#727B84"
GREEN = "#176B50"
AMBER = "#CF8B28"
CHARCOAL = "#202B35"
GUIDE = "#929BA3"

X_LIMITS = (0, 112)
Y_LIMITS = (0, 108)
FIGSIZE = (16, 9)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def value(schedule: tuple[float, float], quantity):
    """Evaluate an affine marginal schedule at a scalar or NumPy array."""
    return schedule[0] + schedule[1] * quantity


def intersection(first: tuple[float, float], second: tuple[float, float]):
    """Solve both schedules' equality, returning quantity and common price."""
    quantity = (second[0] - first[0]) / (first[1] - second[1])
    return quantity, value(first, quantity)


def model_values():
    q_market, p_market = intersection(DEMAND, MPC)
    q_optimal, p_consumer = intersection(DEMAND, MSC)
    charge = MSC[0] - MPC[0]
    if not np.isclose(MSC[1], MPC[1]):
        raise ValueError("This diagram assumes a constant marginal external damage.")
    p_producer = value(MPC, q_optimal)
    q_dwl = np.linspace(q_optimal, q_market, 201)
    dwl = np.trapezoid(value(MSC, q_dwl) - value(DEMAND, q_dwl), q_dwl)
    return q_market, p_market, q_optimal, p_consumer, p_producer, charge, dwl


def base_chart(title: str, subtitle: str, *, policy: bool):
    """Make identical axes, curves, colors, and typography for both slides."""
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    ax = fig.add_axes((0.09, 0.235, 0.83, 0.605))
    ax.set_facecolor("white")
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)

    q = np.linspace(*X_LIMITS, 400)
    ax.plot(q, value(DEMAND, q), color=BLUE, lw=3.3, zorder=3)
    ax.plot(q, value(MPC, q), color=GRAY, lw=2.8, zorder=3)
    ax.plot(q, value(MSC, q), color=GREEN, lw=3.5, zorder=4)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(CHARCOAL)
        ax.spines[spine].set_linewidth(1.5)
    ax.set_xticks([0, 20, 40, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", colors=CHARCOAL, labelsize=12, length=5, pad=7)
    ax.set_xlabel("Quantity of AI Compute", fontsize=17, color=CHARCOAL, labelpad=23)
    ax.set_ylabel("Marginal Cost / Benefit", fontsize=17, color=CHARCOAL, labelpad=18)

    fig.text(0.09, 0.947, title, ha="left", va="top",
             fontsize=25, fontweight="bold", color=CHARCOAL)
    fig.text(0.09, 0.892, subtitle, ha="left", va="top",
             fontsize=16, color="#53616C")
    fig.text(0.09, 0.044,
             "Illustrative economic model; not empirical AI industry estimates.",
             ha="left", va="center", fontsize=11, color="#626D76")

    # Direct labels sit away from the crossings. The long policy label sits above its curve.
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.5)
    ax.text(108, 31.8, "Demand (MPB = MSB)", ha="right", va="center",
            fontsize=14, fontweight="bold", color=BLUE, bbox=label_box)
    ax.text(108, 67.8, "MPC", ha="right", va="center",
            fontsize=14, fontweight="bold", color=GRAY, bbox=label_box)
    if policy:
        ax.text(109, 94, "MSC = MPC + Environmental Charge", ha="right", va="center",
                fontsize=14, fontweight="bold", color=GREEN, bbox=label_box)
        ax.text(109, 88.3, "MSC unchanged; firms face MPC + charge",
                ha="right", va="center", fontsize=11.5, color=GREEN,
                bbox=label_box)
    else:
        ax.text(108, 88, "MSC", ha="right", va="center",
                fontsize=14, fontweight="bold", color=GREEN, bbox=label_box)
    return fig, ax


def equilibrium_guides(ax, q_market, p_market, q_optimal, p_consumer):
    """Use the same quantity guides and axis labels on each graph."""
    for q, p in ((q_market, p_market), (q_optimal, p_consumer)):
        ax.vlines(q, 0, p, color=GUIDE, lw=1.5, ls=(0, (3, 4)), zorder=1)
    for q, label in ((q_optimal, r"$Q^*$"), (q_market, r"$Q_{\mathrm{market}}$")):
        ax.annotate(label, xy=(q, 0), xytext=(0, -13), textcoords="offset points",
                    ha="center", va="top", fontsize=16, fontweight="bold",
                    color=CHARCOAL, annotation_clip=False)


def deadweight_loss(ax, q_optimal, q_market, *, policy: bool):
    """Shade the exact area between MSC and demand over the inefficient range."""
    q = np.linspace(q_optimal, q_market, 300)
    ax.fill_between(q, value(DEMAND, q), value(MSC, q),
                    facecolor=AMBER, alpha=0.12 if policy else 0.34,
                    edgecolor="none", zorder=2)
    if policy:
        ax.plot(q, value(DEMAND, q), color=AMBER, lw=1.4, ls="--", zorder=5)
        ax.plot(q, value(MSC, q), color=AMBER, lw=1.4, ls="--", zorder=5)
        label = "Deadweight loss avoided\nafter policy"
    else:
        label = "Deadweight Loss"
    label_position = (77.5, 62) if policy else (85, 64)
    ax.annotate(label, xy=(70, 62), xytext=label_position,
                ha="left", va="center", fontsize=13.5, fontweight="bold",
                color="#9A5F11",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6,
                                shrinkA=3, shrinkB=3), zorder=7)


def plot_part_a(results):
    q_market, p_market, q_optimal, p_consumer, _, _, _ = results
    fig, ax = base_chart(
        "Part A | Negative Externalities in AI Development",
        "Unpriced environmental costs lead to excessive AI compute production",
        policy=False,
    )
    deadweight_loss(ax, q_optimal, q_market, policy=False)
    equilibrium_guides(ax, q_market, p_market, q_optimal, p_consumer)

    ax.scatter([q_market], [p_market], s=125, color=GRAY, edgecolor="white",
               linewidth=1.8, zorder=9)
    ax.scatter([q_optimal], [p_consumer], s=180, color=GREEN, edgecolor="white",
               linewidth=2, zorder=10)
    ax.text(83, 27, "Market equilibrium", color=CHARCOAL, fontsize=13,
            ha="left", va="center")
    ax.annotate("Social optimum", xy=(q_optimal, p_consumer), xytext=(28, 67),
                fontsize=13, color=CHARCOAL, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=CHARCOAL, lw=1.3,
                                shrinkA=3, shrinkB=8))

    x_gap = 24
    ax.annotate("", xy=(x_gap, value(MSC, x_gap)),
                xytext=(x_gap, value(MPC, x_gap)),
                arrowprops=dict(arrowstyle="<->", color=CHARCOAL, lw=1.6))
    ax.text(28, 43, "Marginal External Cost", fontsize=13, color=CHARCOAL,
            ha="left", va="center")
    fig.text(0.09, 0.091,
             r"Unpriced environmental damage causes overproduction: $Q_{\mathrm{market}} > Q^*$.",
             fontsize=15, fontweight="bold", color=CHARCOAL,
             ha="left", va="center")
    return fig


def plot_part_c(results):
    q_market, p_market, q_optimal, p_consumer, p_producer, charge, _ = results
    fig, ax = base_chart(
        "Part C | Correcting the Market Failure",
        "Environmental charges internalize external costs and improve efficiency",
        policy=True,
    )
    deadweight_loss(ax, q_optimal, q_market, policy=True)
    equilibrium_guides(ax, q_market, p_market, q_optimal, p_consumer)

    ax.scatter([q_market], [p_market], s=80, color=GRAY, alpha=0.75,
               edgecolor="white", linewidth=1.3, zorder=8)
    ax.scatter([q_optimal], [p_consumer], s=210, color=GREEN,
               edgecolor="white", linewidth=2, zorder=10)
    ax.text(83, 27, "Initial market outcome", color=CHARCOAL, fontsize=13,
            ha="left", va="center")
    ax.annotate("Efficient outcome", xy=(q_optimal, p_consumer), xytext=(27, 68),
                fontsize=13, color=CHARCOAL, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=CHARCOAL, lw=1.3,
                                shrinkA=3, shrinkB=8))

    # The green schedule is the unchanged MSC and the firms' effective MPC after tax.
    ax.annotate("", xy=(q_optimal, p_consumer - 1),
                xytext=(q_optimal, p_producer + 1),
                arrowprops=dict(arrowstyle="<->", color=CHARCOAL, lw=1.8), zorder=8)
    ax.text(56, 49, f"Corrective charge = {charge:g}", ha="right", va="center",
            fontsize=13, color=CHARCOAL)

    ax.annotate("", xy=(q_optimal, 13), xytext=(q_market, 13),
                arrowprops=dict(arrowstyle="->", color=CHARCOAL, lw=1.8), zorder=8)
    ax.text((q_optimal + q_market) / 2, 18, "Output reduction", ha="center",
            va="bottom", fontsize=13, color=CHARCOAL)

    fig.text(0.09, 0.102,
             "If environmental charges accurately reflect external costs, "
             "output moves toward Q*\n"
             "and deadweight loss is eliminated in this simplified model.",
             fontsize=14.5, fontweight="bold", color=CHARCOAL,
             ha="left", va="center")
    fig.text(0.09, 0.066,
             "Graph illustrates environmental charges; separate electricity tariffs "
             "address infrastructure cost-shifting.",
             fontsize=11, color="#53616C", ha="left", va="center")
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = model_values()
    q_market, p_market, q_optimal, p_consumer, p_producer, charge, dwl = results
    print(f"Market equilibrium: Q = {q_market:g}, price = {p_market:g}")
    print(f"Social optimum / post-policy equilibrium: Q* = {q_optimal:g}")
    print(f"Post-policy consumer price = {p_consumer:g}")
    print(f"Post-policy producer price before charge = {p_producer:g}")
    print(f"Corrective environmental charge = {charge:g} per unit")
    print(f"Original deadweight loss = {dwl:g} illustrative value units")

    for stem, plotting_function in (
        ("part_a_market_failure", plot_part_a),
        ("part_c_policy_solution", plot_part_c),
    ):
        fig = plotting_function(results)
        for suffix in ("png", "svg"):
            path = OUTPUT_DIR / f"{stem}.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white")
            print(f"Saved: {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
