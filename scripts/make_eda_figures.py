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

    # Jahreszeiten hinterlegen: Winter (Dez-Feb) blau, Sommer (Jun-Aug) orange
    for year in range(2021, 2026):
        ax.axvspan(pd.Timestamp(f"{year}-06-01", tz="UTC"),
                   pd.Timestamp(f"{year}-09-01", tz="UTC"),
                   color=ORANGE, alpha=0.10, linewidth=0, zorder=0)
        start = pd.Timestamp(f"{year - 1}-12-01", tz="UTC")
        if start < daily.index[0]:
            start = daily.index[0]
        ax.axvspan(start, pd.Timestamp(f"{year}-03-01", tz="UTC"),
                   color=SKY, alpha=0.14, linewidth=0, zorder=0)
    ax.axvspan(pd.Timestamp("2025-12-01", tz="UTC"), daily.index[-1],
               color=SKY, alpha=0.14, linewidth=0, zorder=0)

    ax.plot(weekly.index, weekly.values, color=BLUE, linewidth=1.3,
            label="Wochenmittel")

    # Linearer Trend ueber die Tagesmittel
    t = (daily.index - daily.index[0]).days.to_numpy(dtype=float)
    slope, intercept = np.polyfit(t, daily.values, 1)
    ax.plot(daily.index, intercept + slope * t, color=VERMILLION,
            linewidth=1.6, linestyle="-", label="Linearer Trend")
    print(f"  Trend: {slope * 365.25:+.0f} MW pro Jahr")

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
def _acf(series: pd.Series, max_lag: int) -> np.ndarray:
    x = series.to_numpy(dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[:-k], x[k:]) / denom
                             for k in range(1, max_lag + 1)])


def fig_acf(df: pd.DataFrame, max_lag: int = 200) -> None:
    # Residuallast plus ihre drei Komponenten: Last und PV tragen die
    # Kalenderperiodik (24/168 h), Wind faellt aperiodisch ab — das
    # begruendet Lag-Wahl und Wettervorhersagen als Eingangsgroessen.
    reihen = [
        ("Residuallast", "residual_load", BLUE, 1.2),
        ("Netzlast", "total_load", "black", 1.1),
        ("Solar", "solar", ORANGE, 1.1),
        ("Wind", "wind_total", SKY, 1.1),
    ]
    acfs = {name: _acf(df[col], max_lag) for name, col, _c, _w in reihen}

    fig, (ax_r, ax_k) = plt.subplots(2, 1, figsize=(5.2, 4.4), sharex=True)
    lags = np.arange(max_lag + 1)

    # Oberes Feld: Residuallast allein
    r = acfs["Residuallast"]
    ax_r.fill_between(lags, 0, r, color=BLUE, alpha=0.25, linewidth=0)
    ax_r.plot(lags, r, color=BLUE, linewidth=1.2)
    ax_r.set_title("Residuallast", fontsize=9)

    # Unteres Feld: die drei Komponenten gemeinsam
    for name, _col, color, width in reihen[1:]:
        ax_k.plot(lags, acfs[name], color=color, linewidth=width, label=name)
    ax_k.set_title("Komponenten", fontsize=9)
    ax_k.legend(frameon=False, ncol=3, loc="upper center",
                bbox_to_anchor=(0.5, -0.42), fontsize=8)

    for ax in (ax_r, ax_k):
        for lag, label in [(24, "24 h"), (168, "168 h")]:
            ax.axvline(lag, color=VERMILLION, linewidth=0.9,
                       linestyle="--", alpha=0.8)
            if ax is ax_r:
                ax.text(lag, 1.02, label, ha="center", va="bottom",
                        fontsize=8, color=VERMILLION)
        ax.set_ylabel("Autokorrelation")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _p: f"{v:.2f}".replace(".", ",")))
        ax.margins(x=0.01)

    ax_r.set_ylim(min(0, r.min() * 1.1), 1.12)
    ymin_k = min(acfs[n].min() for n in ("Netzlast", "Solar", "Wind"))
    ax_k.set_ylim(ymin_k * 1.15, 1.05)
    ax_k.set_xlabel("Zeitversatz in Stunden")
    ax_k.set_xticks([0, 24, 48, 96, 144, 168, 200])
    fig.tight_layout()

    fig.savefig(OUT_DIR / "eda_acf.pdf", bbox_inches="tight")
    plt.close(fig)
    r = acfs["Residuallast"]
    print(f"  eda_acf.pdf            (Residuallast: ACF(24) = {r[24]:.3f}, "
          f"ACF(168) = {r[168]:.3f})")
    for name in ("Netzlast", "Solar", "Wind"):
        a = acfs[name]
        print(f"    {name:<12} ACF(12) = {a[12]:.3f}, ACF(24) = {a[24]:.3f}, "
              f"ACF(48) = {a[48]:.3f}, ACF(168) = {a[168]:.3f}")


# ---------------------------------------------------------------------------
# Abbildung 4: Beispielwochen Winter vs. Sommer — Mechanik der Residuallast
# ---------------------------------------------------------------------------
def fig_beispielwochen(df: pd.DataFrame) -> None:
    local = df.copy()
    local.index = local.index.tz_convert("Europe/Berlin")

    weeks = [("Winterwoche (16.--22. Januar 2023)", "2023-01-16", "2023-01-23"),
             ("Sommerwoche (10.--16. Juli 2023)", "2023-07-10", "2023-07-17")]

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.6), sharey=True)
    for ax, (title, start, end) in zip(axes, weeks):
        w = local.loc[start:end]
        # Gestapelte Zerlegung: Netzlast = Residuallast + Wind + Solar.
        # Basis ist die Residuallast (kann negativ werden), darauf Wind,
        # darauf Solar — die Stapel-Oberkante entspricht der Netzlast.
        top_wind = w["residual_load"] + w["wind_total"]
        top_solar = top_wind + w["solar"]
        ax.fill_between(w.index, 0, w["residual_load"], color=GREY,
                        alpha=0.55, linewidth=0, label="Residuallast")
        ax.fill_between(w.index, w["residual_load"], top_wind, color=SKY,
                        alpha=0.55, linewidth=0, label="Wind")
        ax.fill_between(w.index, top_wind, top_solar, color=ORANGE,
                        alpha=0.6, linewidth=0, label="Solar")
        ax.plot(w.index, w["total_load"], color="black", linewidth=1.0,
                label="Netzlast")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(title.replace("--", "–"))
        ax.set_ylabel("Leistung in MW")
        ax.yaxis.set_major_formatter(FuncFormatter(de_thousands))
        ax.margins(x=0)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d.%m.",
                                                          tz=w.index.tz))
    axes[1].legend(frameon=False, ncol=4, loc="upper right")

    fig.savefig(OUT_DIR / "eda_beispielwochen.pdf")
    plt.close(fig)
    print("  eda_beispielwochen.pdf")


# ---------------------------------------------------------------------------
# Abbildung 5: Verteilung der Residuallast (Histogramm)
# ---------------------------------------------------------------------------
def fig_verteilung(df: pd.DataFrame) -> None:
    x = df["residual_load"]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.hist(x, bins=90, color=BLUE, alpha=0.85, linewidth=0)
    ax.axvline(0, color=VERMILLION, linewidth=1.0, linestyle="--")
    ax.axvline(x.mean(), color="black", linewidth=0.9, linestyle=":",
               label=f"Mittelwert")
    ax.set_xlabel("Residuallast in MW")
    ax.set_ylabel("Stunden")
    ax.xaxis.set_major_formatter(FuncFormatter(de_thousands))
    ax.yaxis.set_major_formatter(FuncFormatter(de_thousands))
    ax.legend(frameon=False)
    ax.margins(x=0.01)

    fig.savefig(OUT_DIR / "eda_verteilung.pdf")
    plt.close(fig)
    skew = float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    print(f"  eda_verteilung.pdf     (Mittel {x.mean():.0f}, Median {x.median():.0f}, Schiefe {skew:.2f})")


# ---------------------------------------------------------------------------
# Abbildung 7: Ergebnis der Wetteraggregation (3 nationale Zeitreihen)
# ---------------------------------------------------------------------------
def fig_wetteraggregate(df: pd.DataFrame) -> None:
    weekly = df[["weather_temperature", "weather_wind_100m",
                 "weather_radiation"]].resample("W").mean()

    fig, axes = plt.subplots(3, 1, figsize=(6.3, 3.8), sharex=True)
    panels = [("weather_temperature", "Temperatur\nin °C", BLUE),
              ("weather_wind_100m", "Wind 100 m\nin km/h", SKY),
              ("weather_radiation", "Strahlung\nin W/m²", ORANGE)]
    for ax, (col, label, color) in zip(axes, panels):
        ax.plot(weekly.index, weekly[col], color=color, linewidth=1.1)
        ax.set_ylabel(label, fontsize=8)
        ax.margins(x=0.01)
    axes[-1].set_xlabel("")

    fig.savefig(OUT_DIR / "eda_wetteraggregate.pdf")
    plt.close(fig)
    print("  eda_wetteraggregate.pdf")


# ---------------------------------------------------------------------------
# Abbildung 8: Schema — zeitliche Struktur der Eingaben (Gantt-Stil,
# ausschliesslich senkrechte Pfeile)
# ---------------------------------------------------------------------------
def fig_schema() -> None:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.set_xlim(-238, 46)
    ax.set_ylim(-4.6, 8.0)
    ax.axis("off")

    PAST_FC, FUT_FC = "#cfe3f2", "#fbe3c8"
    BAR_H = 0.62

    # Hintergrund: Vergangenheit / Horizont
    ax.axvspan(-168, 0, ymin=0.42, ymax=0.97, color=PAST_FC, alpha=0.25,
               linewidth=0)
    ax.axvspan(0, 24, ymin=0.42, ymax=0.97, color=FUT_FC, alpha=0.35,
               linewidth=0)

    # Zeitachse oben
    y_ax = 7.2
    ax.annotate("", xy=(40, y_ax), xytext=(-175, y_ax),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.9))
    for x, lab in [(-168, "$t_0-168\\,$h"), (-24, "$t_0-24\\,$h"),
                   (0, "$t_0$"), (24, "$t_0+24\\,$h")]:
        ax.plot([x, x], [y_ax - 0.14, y_ax + 0.14], color="black",
                linewidth=0.8)
        ax.text(x, y_ax + 0.32, lab, ha="center", fontsize=7.5)
    # t0-Linie durch den Eingabebereich
    ax.plot([0, 0], [1.6, y_ax - 0.4], color="black", linewidth=1.0,
            linestyle="--", alpha=0.7)

    # Eingabezeilen (Gantt): (y, Label, Balken oder Marker)
    rows = [
        (6.0, "Residuallast (Target-History)",        ("bar", -168, 0, BLUE)),
        (5.0, "Energie-Lags (Past Covariates)",       ("dots", [-168, -24], BLUE)),
        (4.0, "Wetter-Istwerte (Past Covariates)",    ("bar", -168, 0, SKY)),
        (3.0, "Kalender, Feiertage (Future Covariates)", ("bar", 0, 24, ORANGE)),
        (2.0, "Wettervorhersagen (Future Covariates)", ("bar", 0, 24, VERMILLION)),
    ]
    for y, label, spec in rows:
        ax.text(-176, y, label, ha="right", va="center", fontsize=8)
        if spec[0] == "bar":
            _, x0, x1, color = spec
            ax.add_patch(plt.Rectangle((x0, y - BAR_H / 2), x1 - x0, BAR_H,
                                       facecolor=color, edgecolor="black",
                                       linewidth=0.5))
        else:
            _, xs, color = spec
            for x in xs:
                ax.plot(x, y, marker="s", markersize=7, color=color,
                        markeredgecolor="black", markeredgewidth=0.5)
            ax.text(-96, y, "Werte von $t_0-24\\,$h und $t_0-168\\,$h",
                    ha="center", va="center", fontsize=7, style="italic")

    def box(xc, y, w, h, text, fc, weight="normal"):
        ax.add_patch(FancyBboxPatch((xc - w / 2, y), w, h,
                                    boxstyle="round,pad=0.06",
                                    facecolor=fc, edgecolor="black",
                                    linewidth=0.7))
        ax.text(xc, y + h / 2, text, ha="center", va="center", fontsize=8,
                weight=weight)

    def varrow(x, y1, y2):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=0.9,
                                     color="black"))

    # Modell- und Ausgabe-Box, ausschliesslich senkrechte Pfeile
    box(-72, -1.3, 190, 1.05,
        "Prognosemodell:  Regression | XGBoost | TFT", "#e8e8e8",
        weight="bold")
    box(-72, -3.6, 190, 1.3,
        "5 Quantile je Stunde des Horizonts\n"
        "$\\hat{q}^{(0{,}1)} \\dots \\hat{q}^{(0{,}9)}$ für "
        "$t_0{+}1, \\dots, t_0{+}24$", "#dff0e2")
    varrow(-100, 1.55, -0.18)   # Vergangenheits-Eingaben -> Modell
    varrow(12, 1.55, -0.18)     # Zukunfts-Eingaben -> Modell
    varrow(-72, -1.38, -2.22)   # Modell -> Ausgabe

    fig.savefig(OUT_DIR / "eda_schema_kovariaten.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  eda_schema_kovariaten.pdf")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    print(f"Daten: {len(data)} Stunden, erzeuge Abbildungen nach {OUT_DIR}")
    fig_jahresverlauf(data)
    fig_tagesgang(data)
    fig_acf(data)
    fig_beispielwochen(data)
    fig_verteilung(data)
    fig_wetteraggregate(data)
    fig_schema()
    # Zusatzinfo fuer den Text: Saisonverteilung der negativen Stunden
    neg = data[data["residual_load"] < 0]
    print("  Negative Stunden nach Monat:",
          dict(neg.groupby(neg.index.month).size()))
    print("Fertig.")
