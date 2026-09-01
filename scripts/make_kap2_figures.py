"""
Erzeugt die Schema-Abbildungen für Kapitel 2.3 der Masterarbeit
(Conformal-Zuschlag und Adaptive Conformal Inference).

Beide Abbildungen beruhen auf einer gekennzeichneten schematischen
Simulation mit festem Seed. Die im Fließtext genannten Kennzahlen
werden hier berechnet und auf stdout ausgegeben.

Ausgabe: PDF-Vektorgrafiken in Tex/img/ (deutsche Beschriftung,
farbenblindfreundliche Okabe-Ito-Palette).

Aufruf:  python scripts/make_kap2_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Pfade -----------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = CODE_DIR.parent / "Tex" / "img"

# --- Okabe-Ito-Palette (farbenblindfreundlich) -----------------------------
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
GREY = "#999999"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
})

ZIEL = 0.9  # Nominalniveau der Schemata (passend zu den Literaturbeispielen)


def punkte(ax, x, y, drin):
    """Beobachtungen zeichnen: blau innerhalb, rot außerhalb des Intervalls."""
    ax.plot(x[drin], y[drin], "o", ms=2.4, color=BLUE,
            label="Messwert innerhalb")
    ax.plot(x[~drin], y[~drin], "o", ms=3.4, color=VERMILLION,
            label="Messwert außerhalb")


# ---------------------------------------------------------------------------
# Abbildung 1: Prinzip des Conformal-Zuschlags (CQR), vorher/nachher
# ---------------------------------------------------------------------------
def fig_cqr_schema() -> None:
    rng = np.random.default_rng(3)
    # Synthetische Reihe mit Tagesgang und heteroskedastischem Rauschen.
    n_kal, n_test = 500, 300
    n = n_kal + n_test
    t = np.arange(n)
    trend = 10 + 3 * np.sin(2 * np.pi * t / 48)
    sigma = 1.0 + 0.5 * np.sin(2 * np.pi * t / 48 + 1.0) ** 2
    y = trend + rng.normal(0, sigma)

    # "Rohes" Quantilmodell: kennt den Trend, unterschaetzt die Streuung
    # (typischer Fall eines auf Trainingsdaten optimierten Modells).
    unterschaetzung = 0.7
    lo_roh = trend - 1.6449 * sigma * unterschaetzung
    hi_roh = trend + 1.6449 * sigma * unterschaetzung

    # Conformal-Zuschlag aus den Kalibrierungsdaten (CQR-Score von Romano).
    scores = np.maximum(lo_roh[:n_kal] - y[:n_kal], y[:n_kal] - hi_roh[:n_kal])
    k = int(np.ceil((n_kal + 1) * ZIEL))
    q_hut = np.sort(scores)[k - 1]

    lo_cqr = lo_roh - q_hut
    hi_cqr = hi_roh + q_hut

    # Kennzahlen auf allen Testdaten
    yt = y[n_kal:]
    drin_roh = (yt >= lo_roh[n_kal:]) & (yt <= hi_roh[n_kal:])
    drin_cqr = (yt >= lo_cqr[n_kal:]) & (yt <= hi_cqr[n_kal:])
    print(f"[CQR] Zuschlag q_hut = {q_hut:.2f}")
    print(f"[CQR] Abdeckung roh (Test):  {drin_roh.mean():.1%}")
    print(f"[CQR] Abdeckung CQR (Test):  {drin_cqr.mean():.1%}")

    # Anzeigefenster: die ersten 150 Teststunden (weniger Punkte, klarer Blick)
    w = slice(0, 150)
    ts = np.arange(150)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)

    ax = axes[0]
    ax.fill_between(ts, lo_roh[n_kal:][w], hi_roh[n_kal:][w], color=GREY,
                    alpha=0.45, label="rohes Intervall")
    punkte(ax, ts, yt[w], drin_roh[w])
    ax.set_title("vor der Kalibrierung")
    ax.text(0.03, 0.03, f"Abdeckung {drin_roh.mean():.1%}".replace(".", ","),
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GREY))
    ax.set_xlabel("Zeit (Stunden, schematisch)")
    ax.set_ylabel("Zielgröße (schematisch)")

    ax = axes[1]
    ax.fill_between(ts, lo_cqr[n_kal:][w], hi_cqr[n_kal:][w], color=SKY,
                    alpha=0.4, label="konformalisiertes Intervall")
    punkte(ax, ts, yt[w], drin_cqr[w])
    ax.set_title(r"nach der Kalibrierung (Zuschlag $c$)")
    ax.text(0.03, 0.03, f"Abdeckung {drin_cqr.mean():.1%}".replace(".", ","),
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GREY))
    ax.set_xlabel("Zeit (Stunden, schematisch)")

    # Zuschlag als Massband im rechten Feld markieren
    i_mark = 62
    y_lo = hi_roh[n_kal + i_mark]
    y_hi = hi_cqr[n_kal + i_mark]
    ax.annotate("", xy=(i_mark, y_hi + 0.05), xytext=(i_mark, y_lo - 0.05),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.4,
                                shrinkA=0, shrinkB=0))
    ax.annotate(r"$c$", xy=(i_mark, (y_lo + y_hi) / 2),
                xytext=(i_mark + 12, (y_lo + y_hi) / 2 + 1.8), fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.7))

    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    handles = [h0[0], h1[0], h0[1], h0[2]]
    labels = [l0[0], l1[0], l0[1], l0[2]]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), framealpha=0.9)
    fig.savefig(OUT_DIR / "sdf_cqr_schema.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Abbildung 2: ACI unter Verteilungswandel vs. statische Kalibrierung
# ---------------------------------------------------------------------------
def fig_aci_schema() -> None:
    rng = np.random.default_rng(7)
    n_kal, n_test = 500, 800
    shift_start = 300  # Beginn des Verteilungswandels im Testzeitraum

    # Sichtbare Zeitreihe: bekannter Mittelwert + Rauschen, dessen Streuung
    # ab shift_start allmaehlich auf das 2,2-Fache anwaechst.
    t = np.arange(n_test)
    mu = 10 + 2.5 * np.sin(2 * np.pi * t / 96)
    sigma = np.ones(n_test)
    sigma[shift_start:] = np.linspace(1.0, 2.2, n_test - shift_start)
    y = mu + rng.normal(0, sigma)

    scores_kal = np.abs(rng.normal(0, 1.0, n_kal))
    scores_test = np.abs(y - mu)

    # Statisch: fester Zuschlag aus der Kalibrierungsphase
    k = int(np.ceil((n_kal + 1) * ZIEL))
    q_stat = np.sort(scores_kal)[k - 1]
    drin_stat = scores_test <= q_stat

    # ACI: alpha_t-Update nach Gibbs & Candes, Quantil aus allen bisherigen Scores
    gamma = 0.02
    alpha_ziel = 1 - ZIEL
    alpha_t = alpha_ziel
    hist = list(scores_kal)
    q_aci = np.zeros(n_test)
    for i in range(n_test):
        a = np.clip(alpha_t, 0.001, 0.999)
        q_aci[i] = np.quantile(hist, 1 - a)
        err = scores_test[i] > q_aci[i]
        alpha_t = alpha_t + gamma * (alpha_ziel - float(err))
        hist.append(scores_test[i])
    drin_aci = scores_test <= q_aci

    pre, post = slice(0, shift_start), slice(shift_start, n_test)
    print(f"[ACI] statisch: Abdeckung vor Wandel {drin_stat[pre].mean():.1%}, "
          f"nach Wandel {drin_stat[post].mean():.1%}")
    print(f"[ACI] ACI:      Abdeckung vor Wandel {drin_aci[pre].mean():.1%}, "
          f"nach Wandel {drin_aci[post].mean():.1%}")
    print(f"[ACI] Intervallbreite ACI Ende / Anfang: "
          f"{q_aci[-100:].mean() / q_aci[:100].mean():.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.0), sharex=True, sharey=True)

    for ax, drin, lo, hi, name, farbe, bandname in (
        (axes[0], drin_stat, mu - q_stat, mu + q_stat,
         "statisch kalibriert", GREY, "Intervall (konstant)"),
        (axes[1], drin_aci, mu - q_aci, mu + q_aci,
         "adaptiv kalibriert (ACI)", SKY, "Intervall (mitlaufend)"),
    ):
        ax.fill_between(t, lo, hi, color=farbe, alpha=0.4, label=bandname)
        punkte(ax, t, y, drin)
        ax.axvline(shift_start, color="black", lw=0.8, ls="--")
        ax.set_title(name)
        ax.set_ylabel("Zielgröße\n(schematisch)")
        ax.text(0.015, 0.04,
                (f"Abdeckung vor Wandel {drin[pre].mean():.0%}, "
                 f"nach Wandel {drin[post].mean():.0%}").replace(".", ","),
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GREY))

    axes[0].text(shift_start + 10, axes[0].get_ylim()[1] * 0.97,
                 "Beginn des Verteilungswandels", fontsize=7.5,
                 color="#444444", va="top")
    axes[1].set_xlabel("Zeit (Stunden, schematisch)")

    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    handles = [h0[0], h1[0], h0[1], h0[2]]
    labels = [l0[0], l1[0], l0[1], l0[2]]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05), framealpha=0.9)
    fig.savefig(OUT_DIR / "sdf_aci_schema.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_cqr_schema()
    fig_aci_schema()
    print(f"Abbildungen geschrieben nach {OUT_DIR}")
