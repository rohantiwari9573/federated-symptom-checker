from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "results" / "architecture.png"

BLUE = "#2a78d6"
GREEN = "#008300"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
LAYER_BG = "#f0efec"
BORDER = "#c3c2b7"

fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 10.9)
ax.set_ylim(0, 10.2)
ax.axis("off")

def layer_band(y0, y1, label, color):
    ax.add_patch(FancyBboxPatch((0.3, y0), 9.4, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                 linewidth=1.2, edgecolor=color, facecolor=LAYER_BG, zorder=1))
    ax.text(0.55, y1 - 0.32, label, fontsize=9.5, color=color, fontweight="bold", va="top")

def box(x, y, w, h, text, color, fontsize=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 linewidth=1.4, edgecolor=color, facecolor="white", zorder=3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=4)

def varrow(x, y0, y1, color=SECONDARY, style="-|>"):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle=style, mutation_scale=12,
                                  linewidth=1.4, color=color, zorder=2))

# Layer 1: Client-side
layer_band(6.9, 9.9, "Layer 1 — Client-Side (Local ML, on-device)", BLUE)
box(0.7, 7.75, 2.75, 1.55, "Symptom MLP\n(131→41)", BLUE)
box(3.65, 7.75, 2.75, 1.55, "Skin CNN\n(MobileNetV3-Small)", BLUE)
box(6.6, 7.75, 2.75, 1.55, "Respiratory CNN\n(mel-spectrogram)", BLUE)
ax.text(5.05, 7.15, "Browser / Android client — symptom entry, image & audio capture, Privacy Dashboard",
        ha="center", fontsize=7.6, color=SECONDARY, style="italic")

varrow(5.05, 6.9, 6.55)

# Layer 2: Federated coordination
layer_band(3.9, 6.75, "Layer 2 — Federated Coordination (per client)", ORANGE)
box(1.4, 4.65, 3.1, 1.5, "Local training\n(clip + Gaussian noise,\nOpacus DP-SGD)", ORANGE)
box(5.5, 4.65, 3.1, 1.5, "Flower client\n(sends noised\nweight update)", ORANGE)
ax.add_patch(FancyArrowPatch((4.55, 5.4), (5.45, 5.4), arrowstyle="-|>", mutation_scale=12,
                              linewidth=1.4, color=SECONDARY, zorder=2))

varrow(5.05, 3.9, 4.55)

# Layer 3: Server
layer_band(0.4, 3.75, "Layer 3 — Central Server (Aggregation)", GREEN)
box(1.4, 1.3, 3.1, 1.5, "Flower server\nFedAvg aggregation\n(FedSymptomStrategy)", GREEN)
box(5.5, 1.3, 3.1, 1.5, "Checkpoint +\nglobal model\nbroadcast", GREEN)
ax.add_patch(FancyArrowPatch((4.55, 2.05), (5.45, 2.05), arrowstyle="-|>", mutation_scale=12,
                              linewidth=1.4, color=SECONDARY, zorder=2))

# broadcast-back arrow (server -> client, dashed, looping on the right, clear of all layer bands)
ax.add_patch(FancyArrowPatch((8.75, 2.05), (10.45, 2.05), connectionstyle="arc3,rad=0", arrowstyle="-",
                              linewidth=1.3, color=SECONDARY, linestyle=(0, (3, 2)), zorder=2))
ax.add_patch(FancyArrowPatch((10.45, 2.05), (10.45, 8.5), connectionstyle="arc3,rad=0", arrowstyle="-",
                              linewidth=1.3, color=SECONDARY, linestyle=(0, (3, 2)), zorder=2))
ax.add_patch(FancyArrowPatch((10.45, 8.5), (9.5, 8.5), arrowstyle="-|>", mutation_scale=11,
                              linewidth=1.3, color=SECONDARY, linestyle=(0, (3, 2)), zorder=2))

ax.text(5.05, 0.75, "Only clipped, noised model updates cross this boundary — raw symptom data, images, and audio never leave the client.",
        ha="center", fontsize=7.4, color=INK, fontweight="bold")

legend_elems = [
    Line2D([0], [0], color=SECONDARY, lw=1.4, label="update flow (client → server)"),
    Line2D([0], [0], color=SECONDARY, lw=1.3, linestyle=(0, (3, 2)), label="global model broadcast (server → client)"),
]
ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=1,
          fontsize=7, frameon=False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
