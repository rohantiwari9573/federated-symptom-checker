"""
Re-plot the real results from real_comparison_experiment.py's run (numbers
transcribed verbatim from benchmarks/results/run_log.txt and
comparison_results.csv) using a validated, colorblind-safe palette and
print-legible typography for the conference paper figures.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "results"

# --- palette (validated categorical + sequential-orange ramp) ---
GREEN = "#008300"   # centralized (categorical slot 6)
BLUE = "#2a78d6"    # federated, no DP (categorical slot 1)
ORANGE_RAMP = ["#f7c3a8", "#f19d6e", "#eb6834", "#b84f27"]  # eps 0.5,1,2,5 (light->dark)
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

ROUNDS = list(range(1, 11))
cent = [1.0]*10
fed_nodp = [0.7734, 0.9980, 0.9970, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
eps_curves = {
    0.5: [0.0315, 0.0600, 0.0305, 0.0528, 0.0498, 0.0305, 0.0681, 0.0661, 0.0671, 0.0467],
    1.0: [0.0315, 0.0356, 0.0559, 0.0335, 0.0559, 0.0762, 0.0935, 0.1047, 0.0589, 0.0335],
    2.0: [0.0315, 0.0315, 0.0437, 0.0620, 0.0640, 0.1423, 0.1636, 0.2063, 0.2327, 0.2358],
    5.0: [0.0467, 0.0325, 0.0488, 0.1057, 0.1778, 0.2185, 0.2724, 0.3547, 0.3831, 0.3841],
}
epsilons = [0.5, 1.0, 2.0, 5.0]
final_dp = {e: eps_curves[e][-1] for e in epsilons}
cent_final, fed_final = 1.0, 1.0

# ---------------------------------------------------------------------
# Figure 1: Accuracy vs communication round
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
ax.plot(ROUNDS, cent, color=GREEN, linewidth=2.4, label="Centralized (upper bound)", zorder=5)
ax.plot(ROUNDS, fed_nodp, color=BLUE, linewidth=2.4, label="Federated, FedAvg (no DP)", zorder=4)
for c, eps in zip(ORANGE_RAMP, epsilons):
    ax.plot(ROUNDS, eps_curves[eps], color=c, linewidth=2.0, linestyle="--",
            marker="o", markersize=4, label=f"Federated + DP (ε={eps})")

ax.set_xlabel("Federated communication round")
ax.set_ylabel("Test accuracy")
ax.set_title("Accuracy vs. communication round", color=INK, fontsize=12, pad=10)
ax.set_xlim(0.7, 10.3)
ax.set_ylim(-0.03, 1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=9)
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
          fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "accuracy_vs_rounds.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: Privacy-utility trade-off
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=200)
xs = epsilons
ys = [final_dp[e] for e in epsilons]
ax.plot(xs, ys, color=ORANGE_RAMP[2], linewidth=2.2, zorder=4)
for x, y, c in zip(xs, ys, ORANGE_RAMP):
    ax.scatter([x], [y], color=c, edgecolor=INK, linewidth=0.6, s=60, zorder=5)
ax.axhline(cent_final, color=GREEN, linestyle=":", linewidth=1.8,
           label="Non-private ceiling (Centralized = FedAvg, no DP)")

ax.set_xlabel("Privacy budget ε  (lower = stronger privacy)")
ax.set_ylabel("Final test accuracy")
ax.set_title("Privacy-utility trade-off (FedAvg + DP-SGD)", color=INK, fontsize=12, pad=10)
ax.set_ylim(-0.03, 1.08)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=9)
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10)
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "privacy_utility_tradeoff.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 3: Final accuracy bar comparison
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
labels = ["Centralized", "FedAvg\n(no DP)"] + [f"FedAvg+DP\nε={e}" for e in epsilons]
values = [cent_final, fed_final] + [final_dp[e] for e in epsilons]
colors = [GREEN, BLUE] + ORANGE_RAMP
bars = ax.bar(labels, values, color=colors, width=0.62)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center",
            fontsize=9, color=INK)
ax.set_ylabel("Final test accuracy")
ax.set_title("Final accuracy: centralized vs. federated vs. federated+DP",
             color=INK, fontsize=11, pad=10)
ax.set_ylim(0, 1.12)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8.5)
ax.yaxis.label.set_size(10)
fig.tight_layout()
fig.savefig(OUT_DIR / "final_accuracy_bar.png", bbox_inches="tight")
plt.close(fig)

print("Wrote 3 restyled figures to", OUT_DIR)
