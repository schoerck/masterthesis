"""
Erzeugt die EDA-Abbildungen für Kapitel 3.1.3 der Masterarbeit.

Ausgabe: PDF-Vektorgrafiken in Tex/img/ (deutsche Beschriftung,
farbenblindfreundliche Okabe-Ito-Palette, Schriftgrößen passend zur
Einbindung mit width=0.95\textwidth bzw. 0.8\textwidth).

Aufruf:  python scripts/make_eda_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Pfade -----------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[1]
DATA_CSV = CODE_DIR / "data" / "processed" / "combined_data.csv"
OUT_DIR = CODE_DIR.parent / "Tex" / "img"

# --- Okabe-Ito-Palette (farbenblindfreundlich) -----------------------------
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
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
    "pdf.fonttype": 42,  # eingebettete TrueType-Schrift (editierbar, normkonform)
})


def de_thousands(x, _pos) -> str:
    """Achsenformat mit deutschem Tausenderpunkt (40000 -> 40.000)."""
    return f"{x:,.0f}".replace(",", ".")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


# ---------------------------------------------------------------------------
# Abbildung 1: Jahresverlauf 2021-2025 (Tages- und Wochenmittel) + Splits
# ---------------------------------------------------------------------------
def fig_jahresverlauf(df: pd.DataFrame) -> None:
    daily = df["residual_load"].resample("D").mean()
    weekly = df["residual_load"].resample("W").mean()

    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    ax.plot(daily.index, daily.values, color=GREY, linewidth=0.5, alpha=0.6,
            label="Tagesmittel")
    ax.plot(weekly.index, weekly.values, color=BLUE, linewidth=1.4,
            label="Wochenmittel")

    # Split-Grenzen (Train | Validierung | Test)
    for split_date in ["2024-01-01", "2025-01-01"]:
        ax.axvline(pd.Timestamp(split_date, tz="UTC"), color="black",
                   linewidth=0.8, linestyle="--", alpha=0.6)
    y_annot = ax.get_ylim()[1]
    for label, x in [("Training", "2022-07-01"), ("Validierung", "2024-07-01"),
                     ("Test", "2025-07-01")]:
        ax.text(pd.Timestamp(x, tz="UTC"), y_annot, label,
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Residuallast in MW")
    ax.yaxis.set_major_formatter(FuncFormatter(de_thousands))
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax.margins(x=0.01)

    fig.savefig(OUT_DIR / "eda_jahresverlauf.pdf")
    plt.close(fig)
    print(f"  eda_jahresverlauf.pdf  ({len(daily)} Tage, {len(weekly)} Wochen)")


# ---------------------------------------------------------------------------
# Abbildung 2: Mittlerer Tagesgang nach Jahreszeit und Wochentagstyp
# (Uhrzeit in Lokalzeit, damit der PV-Mittagseinbruch bei 12 Uhr liegt)
# ---------------------------------------------------------------------------
def fig_tagesgang(df: pd.DataFrame) -> None:
    local = df.copy()
    local.index = local.index.tz_convert("Europe/Berlin")
    local["hour"] = local.index.hour

    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Frühling", 4: "Frühling", 5: "Frühling",
                  6: "Sommer", 7: "Sommer", 8: "Sommer",
                  9: "Herbst", 10: "Herbst", 11: "Herbst"}
    local["season"] = local.index.month.map(season_map)

    weekday = local[(local.index.dayofweek < 5) & (local["is_holiday"] == 0)]
    weekend = local[(local.index.dayofweek >= 5) | (local["is_holiday"] == 1)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)

    season_colors = {"Winter": BLUE, "Frühling": GREEN,
                     "Sommer": ORANGE, "Herbst": VERMILLION}
    for season, color in season_colors.items():
        prof = local[local["season"] == season].groupby("hour")["residual_load"].mean()
        ax1.plot(prof.index, prof.values, color=color, label=season)
    ax1.set_title("nach Jahreszeit")
    ax1.set_ylabel("Residuallast in MW")
    ax1.legend(frameon=False, ncol=2, loc="lower left")

    for sub, color, label in [(weekday, BLUE, "Werktag"),
                              (weekend, ORANGE, "Wochenende/Feiertag")]:
        prof = sub.groupby("hour")["residual_load"].mean()
        ax2.plot(prof.index, prof.values, color=color, label=label)
    ax2.set_title("nach Wochentagstyp")
    ax2.legend(frameon=False, loc="lower left")

    for ax in (ax1, ax2):
        ax.set_xlabel("Uhrzeit (Lokalzeit)")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.margins(x=0.02)
    ax1.yaxis.set_major_formatter(FuncFormatter(de_thousands))

    fig.savefig(OUT_DIR / "eda_tagesgang.pdf")
    plt.close(fig)
    print(f"  eda_tagesgang.pdf      (Werktage: {len(weekday)} h, "
          f"Wochenende/Feiertag: {len(weekend)} h)")


# ---------------------------------------------------------------------------
# Abbildung 3: Autokorrelationsfunktion bis Lag 200 h
# ---------------------------------------------------------------------------
def fig_acf(df: pd.DataFrame, max_lag: int = 200) -> None:
    x = df["residual_load"].to_numpy()
    x = x - x.mean()
    denom = np.dot(x, x)
    acf = np.array([1.0] + [np.dot(x[:-k], x[k:]) / denom
                            for k in range(1, max_lag + 1)])

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    lags = np.arange(max_lag + 1)
    ax.fill_between(lags, 0, acf, color=BLUE, alpha=0.25, linewidth=0)
    ax.plot(lags, acf, color=BLUE, linewidth=1.2)

    for lag, label in [(24, "24 h"), (168, "168 h")]:
        ax.axvline(lag, color=VERMILLION, linewidth=0.9, linestyle="--", alpha=0.8)
        ax.text(lag, 1.02, label, ha="center", va="bottom",
                fontsize=8, color=VERMILLION)

    ax.set_xlabel("Zeitversatz in Stunden")
    ax.set_ylabel("Autokorrelation")
    ax.set_xticks([0, 24, 48, 96, 144, 168, 200])
    ax.set_ylim(min(0, acf.min() * 1.1), 1.08)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f"{v:.2f}".replace(".", ",")))
    ax.margins(x=0.01)

    fig.savefig(OUT_DIR / "eda_acf.pdf")
    plt.close(fig)
    print(f"  eda_acf.pdf            (ACF(24) = {acf[24]:.3f}, "
          f"ACF(168) = {acf[168]:.3f})")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    print(f"Daten: {len(data)} Stunden, erzeuge Abbildungen nach {OUT_DIR}")
    fig_jahresverlauf(data)
    fig_tagesgang(data)
    fig_acf(data)
    print("Fertig.")
