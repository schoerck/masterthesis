"""
Erzeugt das XGBoost-Schema fuer Abschnitt 3.2.3 als PDF.

Linkes Feld: ein einzelner Regressionsbaum als Abfolge von Ja-Nein-Abfragen
mit konstanten Blattwerten (schematische Zahlen in MW).
Rechtes Feld: stueckweise konstante Prognose eines Baum-Ensembles entlang
eines Merkmals (schematische Simulation) samt fehlender Extrapolation
ausserhalb des im Training beobachteten Wertebereichs.

Aufruf:  .venv/bin/python scripts/make_xgb_schema.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parents[2] / "Tex" / "img"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#999999"
EDGE = "#555555"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def draw_tree(ax) -> None:
    """Einzelner Beispielbaum: zwei Abfragen, drei Blaetter."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Einzelner Regressionsbaum", fontsize=9)

    def node(x, y, w, h, text, leaf=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.12",
                                    linewidth=0.9, edgecolor=EDGE,
                                    facecolor="#DCEDF7" if leaf else "#FAFAFA"))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.6)

    def arrow(x0, y0, x1, y1, lab):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.9,
                                     color=EDGE, shrinkA=1, shrinkB=1))
        ax.text((x0 + x1) / 2 + (0.55 if x1 > x0 else -0.55),
                (y0 + y1) / 2 + 0.15, lab, ha="center", va="center",
                fontsize=7.0, color="#333333")

    node(5.0, 8.6, 5.4, 1.5, "Strahlungsvorhersage\n$<$ 200 W/m$^2$?")
    node(2.5, 5.4, 3.4, 1.4, "Stunde\n$<$ 7 Uhr?")
    node(7.5, 5.4, 3.4, 1.4, "48.600 MW", leaf=True)
    node(1.6, 2.0, 2.9, 1.4, "38.100 MW", leaf=True)
    node(5.0, 2.0, 2.9, 1.4, "52.400 MW", leaf=True)

    arrow(3.9, 7.9, 2.8, 6.15, "ja")
    arrow(6.1, 7.9, 7.2, 6.15, "nein")
    arrow(1.9, 4.65, 1.7, 2.75, "ja")
    arrow(3.1, 4.65, 4.6, 2.75, "nein")


def draw_ensemble(ax) -> None:
    """Stueckweise konstante Ensemble-Prognose mit Extrapolationsgrenze."""
    rng = np.random.default_rng(7)
    x_tr = rng.uniform(0, 780, 130)
    f = lambda x: 50_000 - 26 * x + 5_000 * np.sin(x / 190)
    y_tr = f(x_tr) + rng.normal(0, 1_900, x_tr.size)

    # Treppenfunktion: Mittelwerte in festen Klassen als Naeherung des Ensembles
    bins = np.linspace(0, 780, 14)
    mids, steps = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x_tr >= lo) & (x_tr < hi)
        if m.any():
            mids.append((lo, hi))
            steps.append(y_tr[m].mean())

    ax.scatter(x_tr, y_tr, s=7, color=GREY, alpha=0.55, linewidths=0,
               label="Trainingsstunden", rasterized=True)
    for (lo, hi), v in zip(mids, steps):
        ax.plot([lo, hi], [v, v], color=BLUE, lw=1.8,
                label="Prognose des Ensembles" if lo == mids[0][0] else None)
    # keine Extrapolation: rechts des Trainingsbereichs bleibt der Randwert
    ax.plot([780, 1_000], [steps[-1], steps[-1]], color=ORANGE, lw=1.8,
            label="außerhalb des Trainingsbereichs")
    ax.axvline(780, color=EDGE, lw=0.8, ls=":")
    ax.axvspan(780, 1_000, color=ORANGE, alpha=0.06)
    ax.text(776, 55_500, "Rand des\nTrainingsbereichs", ha="right",
            va="top", fontsize=7.2, color="#333333")

    ax.set_xlim(0, 1_000)
    ax.set_ylim(24_000, 57_000)
    ax.set_xlabel("Merkmal (schematisch)")
    ax.set_ylabel("Prognose (MW)")
    ax.set_title("Prognose eines Baum-Ensembles", fontsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.legend(frameon=False, loc="lower left", fontsize=7.2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 2.9),
                                   gridspec_kw={"width_ratios": [1.0, 1.35],
                                                "wspace": 0.28})
    draw_tree(ax1)
    draw_ensemble(ax2)
    fig.savefig(OUT_DIR / "methodik_xgb_schema.pdf", bbox_inches="tight")
    plt.close(fig)
    print("gespeichert:", OUT_DIR / "methodik_xgb_schema.pdf")


if __name__ == "__main__":
    main()
