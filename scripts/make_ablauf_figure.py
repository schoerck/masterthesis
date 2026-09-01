"""
Erzeugt das Ablaufschema der Modellierung (Kapitel 3.2) als PDF.

Fünf Schritte von der Hyperparameter-Suche bis zur kalibrierten Bewertung.
Unter jeder Box zeigt eine Datenleiste (Segmente 2021-23 | 2024 | 2025),
welche Datenmenge der Schritt nutzt: grau = darauf trainiert,
blau = darauf bewertet beziehungsweise angewendet.

Aufruf:  python scripts/make_ablauf_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parents[2] / "Tex" / "img"

TRAIN_FILL = "#B5B5B5"   # grau: darauf trainiert
EVAL_FILL = "#7FC3EA"    # blau: darauf bewertet/angewendet
EMPTY_FILL = "#FFFFFF"
EDGE = "#555555"

plt.rcParams.update({
    "font.size": 7.0,
    "pdf.fonttype": 42,
})

# (Boxtext, Rolle je Segment: t=trainiert, e=bewertet/angewendet, -=ungenutzt)
BOXES = [
    ("Hyperparameter-Suche\nje Kandidaten-\nkonfiguration", "te-"),
    ("Validierungslauf\nScores $\\rightarrow c_0,\\ \\gamma$\nTFT: Epochenzahl", "te-"),
    ("Finales Training", "tt-"),
    ("Prognose\ntäglich rollierend", "--e"),
    ("ACI stündlich,\nBewertung roh\nund kalibriert", "--e"),
]
SEG_W = [3, 1, 1]          # Breitenverhältnis 2021-23 : 2024 : 2025
SEG_LABEL = ["21–23", "24", "25"]
ROLE_FILL = {"t": TRAIN_FILL, "e": EVAL_FILL, "-": EMPTY_FILL}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 1.95))
    ax.set_xlim(0, 108)
    ax.set_ylim(0, 23)
    ax.axis("off")

    w, h, gap, pad = 19.3, 10.5, 2.1, 0.5
    bar_y, bar_h = 2.2, 2.6
    x = 0.9
    edges = []
    for text, roles in BOXES:
        box = FancyBboxPatch((x, 8), w, h, boxstyle=f"round,pad={pad}",
                             linewidth=0.9, edgecolor=EDGE, facecolor="#FAFAFA")
        ax.add_patch(box)
        ax.text(x + w / 2, 8 + h / 2, text, ha="center", va="center", fontsize=6.6)

        seg_total = sum(SEG_W)
        sx = x
        for role, sw, lab in zip(roles, SEG_W, SEG_LABEL):
            seg_w = w * sw / seg_total
            ax.add_patch(Rectangle((sx, bar_y), seg_w, bar_h, linewidth=0.6,
                                   edgecolor=EDGE, facecolor=ROLE_FILL[role]))
            ax.text(sx + seg_w / 2, bar_y + bar_h / 2, lab, ha="center",
                    va="center", fontsize=5.4,
                    color="black" if role != "-" else "#999999")
            sx += seg_w
        edges.append((x - pad, x + w + pad))
        x += w + gap

    for i in range(4):
        x0 = edges[i][1] + 0.05
        x1 = edges[i + 1][0] - 0.05
        arrow = FancyArrowPatch((x0, 13.25), (x1, 13.25), arrowstyle="-|>",
                                mutation_scale=8, linewidth=1.0, color=EDGE)
        ax.add_patch(arrow)

    # Legende unten rechts
    lx = 76.0
    for fill, lab in [(TRAIN_FILL, "darauf trainiert"), (EVAL_FILL, "darauf bewertet/angewendet")]:
        ax.add_patch(Rectangle((lx, 20.6), 2.0, 1.7, linewidth=0.6,
                               edgecolor=EDGE, facecolor=fill))
        ax.text(lx + 2.7, 21.45, lab, ha="left", va="center", fontsize=6.4)
        lx += 3.6 + len(lab) * 1.05

    fig.savefig(OUT_DIR / "methodik_ablauf.pdf", bbox_inches="tight")
    plt.close(fig)
    print("gespeichert:", OUT_DIR / "methodik_ablauf.pdf")


if __name__ == "__main__":
    main()
