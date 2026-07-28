"""
Automatische Notebook-Generierung nach Modellläufen.

Erstellt ein nummeriertes, zeitgestempeltes Jupyter-Notebook mit:
  - Laufinfo (Run-ID, Datum, Target, Horizont)
  - Datenbasis & Quellen
  - Feature-Design (Kovariaten + Bundesland-Gewichte)
  - Modellparameter
  - Ergebnisvergleich aller drei Modelle
  - Visualisierungen (letzte Woche, Metriken, Fehleranalyse)
  - Reproduzierbarkeit

Dateiname-Schema: run_001_20260419_1923_residual_load_day_ahead.ipynb
"""

import base64
import logging
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Kein Display nötig
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import nbformat
import pandas as pd

logger = logging.getLogger(__name__)

# Modellfarben (konsistent über alle Plots)
_MODEL_COLORS = {
    "regression": "#2196F3",   # Blau
    "tft":        "#4CAF50",   # Grün
    "xgboost":    "#FF9800",   # Orange
}
_MODEL_LABELS = {
    "regression": "Regression",
    "tft":        "TFT",
    "xgboost":    "XGBoost",
}


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _md(source: str) -> nbformat.NotebookNode:
    """Erstellt eine Markdown-Zelle."""
    return nbformat.v4.new_markdown_cell(source.strip())


def _next_run_number(notebooks_dir: Path) -> int:
    """Gibt die nächste freie Laufnummer zurück (1-basiert)."""
    existing = list(notebooks_dir.glob("run_*.ipynb"))
    if not existing:
        return 1
    numbers = []
    for p in existing:
        m = re.match(r"run_(\d+)_", p.name)
        if m:
            numbers.append(int(m.group(1)))
    return max(numbers, default=0) + 1


def _load_result(results_dir: Path, model: str, target: str, horizon: str) -> pd.Series | None:
    """Lädt eine Evaluations-CSV und gibt die erste Zeile als Series zurück."""
    path = results_dir / f"{model}_{target}_{horizon}_evaluation.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df.iloc[0]
    except Exception:
        return None


def _load_forecast(forecast_dir: Path, model: str, target: str, horizon: str) -> pd.DataFrame | None:
    """Lädt eine Forecast-CSV.

    Bevorzugt die ACI-kalibrierte Variante (`*_forecast_calibrated.csv`),
    falls vorhanden, und mappt q10_cal80/q90_cal80 → q10/q90, sodass alle
    nachgelagerten Plot- und Statistik-Funktionen unverändert mit den
    *ehrlichen* (80 %-kalibrierten) Bändern arbeiten. Originalwerte bleiben
    als q10_uncal/q90_uncal in den Diagnose-Spalten erhalten — wer die rohen
    Bänder explizit braucht, kann sie dort weiterhin lesen.
    """
    calibrated = forecast_dir / f"{model}_{target}_{horizon}_forecast_calibrated.csv"
    raw        = forecast_dir / f"{model}_{target}_{horizon}_forecast.csv"

    path = calibrated if calibrated.exists() else raw
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    except Exception:
        return None

    # Wenn wir die kalibrierte Version geladen haben: q10/q90 auf die
    # kalibrierten Werte umlenken, Originale als q10_uncal/q90_uncal behalten.
    if path is calibrated and "q10_cal80" in df.columns and "q90_cal80" in df.columns:
        df = df.rename(columns={"q10": "q10_uncal", "q90": "q90_uncal"})
        df["q10"] = df["q10_cal80"]
        df["q90"] = df["q90_cal80"]
    return df


def _img_b64(path: Path, alt: str = "") -> str:
    """
    Liest eine PNG-Datei und gibt einen base64-kodierten Markdown-Bildtag zurück.
    So sind Plots direkt im Notebook eingebettet — keine externen Dateien nötig.
    Gibt einen Hinweistext zurück wenn die Datei nicht existiert.
    """
    if not path.exists():
        return f"*Grafik nicht vorhanden: `{path.name}`*"
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{data}" alt="{alt}" style="max-width:100%;" />'
    except Exception as exc:
        return f"*Grafik konnte nicht eingebettet werden: {exc}*"


def _fmt(val, decimals=1) -> str:
    """Formatiert einen numerischen Wert für Markdown-Tabellen."""
    try:
        return f"{float(val):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _delta(val_a, val_b) -> str:
    """Zeigt relative Verbesserung (val_b vs. val_a), negativ = besser."""
    try:
        a, b = float(val_a), float(val_b)
        if a == 0:
            return "—"
        pct = (b - a) / abs(a) * 100
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.1f}%"
    except (TypeError, ValueError):
        return "—"


# ─────────────────────────────────────────────────────────────────────────────
# Plot-Generierung
# ─────────────────────────────────────────────────────────────────────────────

def _generate_run_plots(
    results_dir: Path,
    forecast_dir: Path,
    target: str,
    horizon: str,
    models: list[str],
    run_plots_dir: Path,
) -> dict[str, Path]:
    """
    Generiert Vergleichs-Plots aus den gespeicherten Forecast-CSVs.

    Returns
    -------
    dict[str, Path]
        Mapping von Plot-Name → Dateipfad (nur vorhandene Plots).
    """
    run_plots_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}

    # Forecast-Daten laden
    forecasts: dict[str, pd.DataFrame] = {}
    for m in models:
        df = _load_forecast(forecast_dir, m, target, horizon)
        if df is not None and len(df) > 0:
            forecasts[m] = df

    if not forecasts:
        logger.warning("Keine Forecast-CSVs gefunden — Plots werden übersprungen.")
        return plots

    # ── Plot 1: Letzte Woche — alle Modelle untereinander ────────────────────
    try:
        n = len(forecasts)
        fig, axes = plt.subplots(n, 1, figsize=(16, 5 * n), sharex=False)
        if n == 1:
            axes = [axes]

        for ax, (m, df) in zip(axes, forecasts.items()):
            df_week = df.tail(168).copy()
            color = _MODEL_COLORS.get(m, "#666666")

            ax.plot(
                df_week["timestamp"], df_week["actual"] / 1000,
                color="black", linewidth=2.5, label="Ist-Werte", zorder=5,
            )
            ax.plot(
                df_week["timestamp"], df_week["prediction"] / 1000,
                color=color, linewidth=2, linestyle="--", label="Median-Prognose",
            )
            if "q10" in df_week.columns and "q90" in df_week.columns:
                ax.fill_between(
                    df_week["timestamp"],
                    df_week["q10"] / 1000,
                    df_week["q90"] / 1000,
                    alpha=0.25, color=color, label="80 %-Konfidenzintervall",
                )

            mae_w = np.mean(np.abs(df_week["actual"].values - df_week["prediction"].values))
            ax.set_title(
                f"{_MODEL_LABELS.get(m, m.upper())} — Letzte 7 Tage Testperiode "
                f"| MAE = {mae_w:,.0f} MW",
                fontsize=13, fontweight="bold",
            )
            ax.set_ylabel("Residuallast [GW]", fontsize=11)
            ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d.%m."))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout(pad=1.5)
        p = run_plots_dir / "01_letzte_woche_alle_modelle.png"
        plt.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close()
        plots["letzte_woche"] = p
        logger.info("Plot gespeichert: %s", p)
    except Exception as exc:
        logger.warning("Plot 1 (letzte Woche) fehlgeschlagen: %s", exc)
        plt.close("all")

    # ── Plot 2: Metriken-Vergleich (MAE / RMSE / Pinball) ────────────────────
    try:
        metrics_data: dict[str, pd.Series] = {}
        for m in models:
            r = _load_result(results_dir, m, target, horizon)
            if r is not None:
                metrics_data[m] = r

        if metrics_data:
            model_names = list(metrics_data.keys())
            bar_colors = [_MODEL_COLORS.get(m, "#999") for m in model_names]
            x = np.arange(len(model_names))

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            def _bar_with_labels(ax, vals, title, ylabel, fmt=".0f"):
                valid_vals = [v for v in vals if not np.isnan(v)]
                bars = ax.bar(x, vals, color=bar_colors, edgecolor="black",
                              linewidth=0.8, width=0.55)
                for bar, val in zip(bars, vals):
                    if not np.isnan(val):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.015,
                            f"{val:{fmt}}",
                            ha="center", va="bottom", fontsize=12, fontweight="bold",
                        )
                ax.set_title(title, fontsize=13, fontweight="bold")
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_xticks(x)
                ax.set_xticklabels([_MODEL_LABELS.get(m, m.upper()) for m in model_names], fontsize=11)
                ax.grid(True, alpha=0.3, axis="y")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if valid_vals:
                    ax.set_ylim(0, max(valid_vals) * 1.18)

            mae_vals = [float(metrics_data[m].get("MAE", np.nan)) for m in model_names]
            rmse_vals = [float(metrics_data[m].get("RMSE", np.nan)) for m in model_names]

            _bar_with_labels(axes[0], mae_vals, "MAE ↓", "MAE [MW]")
            _bar_with_labels(axes[1], rmse_vals, "RMSE ↓", "RMSE [MW]")

            # Dritter Plot: Pinball Loss wenn vorhanden, sonst R²
            has_pinball = any("pinball_mean" in metrics_data[m].index for m in model_names)
            if has_pinball:
                pb_vals = [float(metrics_data[m].get("pinball_mean", np.nan)) for m in model_names]
                _bar_with_labels(axes[2], pb_vals, "Pinball Loss ↓", "Pinball Loss [MW]")
            else:
                r2_vals = [float(metrics_data[m].get("R2", np.nan)) for m in model_names]
                _bar_with_labels(axes[2], r2_vals, "R² ↑", "R²", fmt=".3f")

            horizon_label = "Day-Ahead (24 h)" if horizon == "day_ahead" else "Week-Ahead (168 h)"
            plt.suptitle(
                f"Modellvergleich: {target.replace('_', ' ').title()} | {horizon_label}",
                fontsize=14, fontweight="bold", y=1.02,
            )
            plt.tight_layout()
            p = run_plots_dir / "02_metriken_vergleich.png"
            plt.savefig(str(p), dpi=150, bbox_inches="tight")
            plt.close()
            plots["metriken"] = p
            logger.info("Plot gespeichert: %s", p)
    except Exception as exc:
        logger.warning("Plot 2 (Metriken) fehlgeschlagen: %s", exc)
        plt.close("all")

    # ── Plot 3: Fehlerverteilung + MAE nach Tageszeit ─────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for m, df in forecasts.items():
            color = _MODEL_COLORS.get(m, "#999")
            label = _MODEL_LABELS.get(m, m.upper())
            errors_gw = (df["actual"].values - df["prediction"].values) / 1000

            # Histogramm
            axes[0].hist(
                errors_gw, bins=80, alpha=0.5, color=color,
                label=label, edgecolor="none", density=True,
            )

            # MAE nach Stunde
            df_h = df.copy()
            df_h["hour"] = df_h["timestamp"].dt.hour
            df_h["abs_error"] = np.abs(df_h["actual"] - df_h["prediction"])
            hourly = df_h.groupby("hour")["abs_error"].mean() / 1000
            axes[1].plot(
                hourly.index, hourly.values,
                marker="o", markersize=4, linewidth=2, color=color, label=label,
            )

        axes[0].axvline(x=0, color="black", linewidth=1.5, linestyle="--", label="Null-Fehler")
        axes[0].set_xlabel("Prognosefehler [GW]", fontsize=12)
        axes[0].set_ylabel("Dichte", fontsize=12)
        axes[0].set_title("Fehlerverteilung (gesamter Testzeitraum)", fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        axes[1].set_xlabel("Stunde des Tages", fontsize=12)
        axes[1].set_ylabel("Ø MAE [GW]", fontsize=12)
        axes[1].set_title("MAE nach Tageszeit", fontsize=13, fontweight="bold")
        axes[1].set_xticks(range(0, 24, 2))
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        p = run_plots_dir / "03_fehleranalyse.png"
        plt.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close()
        plots["fehler"] = p
        logger.info("Plot gespeichert: %s", p)
    except Exception as exc:
        logger.warning("Plot 3 (Fehleranalyse) fehlgeschlagen: %s", exc)
        plt.close("all")

    # ── Plot 4: Boxplot der Fehler je Modell ─────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(9, 5))

        error_data = []
        error_labels = []
        box_colors_list = []
        for m, df in forecasts.items():
            errors = (df["actual"].values - df["prediction"].values) / 1000
            error_data.append(errors)
            error_labels.append(_MODEL_LABELS.get(m, m.upper()))
            box_colors_list.append(_MODEL_COLORS.get(m, "#999"))

        bp = ax.boxplot(
            error_data,
            labels=error_labels,
            patch_artist=True,
            notch=False,
            medianprops={"color": "black", "linewidth": 2.5},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
        )
        for patch, color in zip(bp["boxes"], box_colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        ax.axhline(y=0, color="black", linewidth=1.2, linestyle="--", label="Null-Fehler")
        ax.set_ylabel("Prognosefehler [GW]", fontsize=12)
        ax.set_title("Fehlerverteilung je Modell (Boxplot, gesamter Testzeitraum)",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        p = run_plots_dir / "04_fehler_boxplot.png"
        plt.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close()
        plots["boxplot"] = p
        logger.info("Plot gespeichert: %s", p)
    except Exception as exc:
        logger.warning("Plot 4 (Boxplot) fehlgeschlagen: %s", exc)
        plt.close("all")

    # ── Plot 5: Scatter Ist vs. Vorhersage je Modell ──────────────────────────
    try:
        n = len(forecasts)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (m, df) in zip(axes, forecasts.items()):
            color = _MODEL_COLORS.get(m, "#666")
            # Sample max 3000 Punkte für Performance
            df_s = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df
            actual_gw = df_s["actual"].values / 1000
            pred_gw = df_s["prediction"].values / 1000

            ax.scatter(actual_gw, pred_gw, alpha=0.25, s=8, color=color, rasterized=True)

            # 45°-Linie
            lim_min = min(actual_gw.min(), pred_gw.min()) * 0.97
            lim_max = max(actual_gw.max(), pred_gw.max()) * 1.03
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1.5,
                    label="Perfekte Prognose")

            # R²
            ss_res = np.sum((actual_gw - pred_gw) ** 2)
            ss_tot = np.sum((actual_gw - np.mean(actual_gw)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            ax.set_title(
                f"{_MODEL_LABELS.get(m, m.upper())}\nR² = {r2:.3f}",
                fontsize=13, fontweight="bold",
            )
            ax.set_xlabel("Ist-Werte [GW]", fontsize=11)
            ax.set_ylabel("Vorhersage [GW]", fontsize=11)
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle("Ist vs. Vorhersage (Scatter)", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        p = run_plots_dir / "05_scatter_ist_vorhersage.png"
        plt.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close()
        plots["scatter"] = p
        logger.info("Plot gespeichert: %s", p)
    except Exception as exc:
        logger.warning("Plot 5 (Scatter) fehlgeschlagen: %s", exc)
        plt.close("all")

    return plots


# ─────────────────────────────────────────────────────────────────────────────
# Notebook-Zellen
# ─────────────────────────────────────────────────────────────────────────────

def _cell_title(run_id: str, run_no: int, ts: datetime,
                target: str, horizon: str, models: list[str]) -> nbformat.NotebookNode:
    horizon_label = "Day-Ahead (24h)" if horizon == "day_ahead" else "Week-Ahead (168h)"
    models_str = " · ".join(_MODEL_LABELS.get(m, m.upper()) for m in models)
    return _md(f"""
# Residuallast-Prognose — Lauf #{run_no:03d}

| | |
|---|---|
| **Lauf-ID** | `{run_id}` |
| **Datum** | {ts.strftime("%d. %B %Y, %H:%M Uhr")} |
| **Target** | `{target}` |
| **Horizont** | {horizon_label} |
| **Modelle** | {models_str} |

---
""")


def _cell_datenbasis(config: dict) -> nbformat.NotebookNode:
    prep = config.get("preprocessing", {})
    smard_cfg   = config["data"]["smard"]
    weather_cfg = config["data"]["weather"]

    # Splits dynamisch aus den Config-Datumsgrenzen ableiten
    import pandas as pd
    train_start = pd.Timestamp(smard_cfg["start_date"])
    train_end   = pd.Timestamp(prep["train_end_date"])
    val_start   = train_end + pd.Timedelta(days=1)
    val_end     = pd.Timestamp(prep["val_end_date"])
    test_start  = val_end + pd.Timedelta(days=1)
    test_end    = pd.Timestamp(smard_cfg["end_date"])

    # Ungefähre Stundenzahlen je Split (24 h × Tagedifferenz)
    train_hours = (train_end - train_start).days * 24 + 24
    val_hours   = (val_end   - val_start).days   * 24 + 24
    test_hours  = (test_end  - test_start).days  * 24 + 24

    def _fmt(ts):
        return ts.strftime("%b %Y").replace(
            "Jan", "Jan").replace("Feb", "Feb").replace("Mar", "Mär")\
            .replace("May", "Mai").replace("Oct", "Okt").replace("Dec", "Dez")

    return _md(f"""
## 1. Datenbasis

### Quellen

| Datentyp | Quelle | Zeitraum | Auflösung |
|---|---|---|---|
| Energie (Residuallast, Last, Wind, Solar) | SMARD API, Bundesnetzagentur | {smard_cfg["start_date"]} – {smard_cfg["end_date"]} | stündlich |
| Wetter-Istwerte (ERA5-Reanalyse) | Open-Meteo Historical Weather API | {weather_cfg["start_date"]} – {weather_cfg["end_date"]} | stündlich |
| Wetter-Vorhersagen | Open-Meteo Historical Forecast API — archivierte Prognose des jeweils **jüngsten Modelllaufs** vor dem Zeitpunkt (Lead-Time typ. 0–12 h, *keine* Vortages-Prognose) | {weather_cfg["start_date"]} – {weather_cfg["end_date"]} | stündlich |
| Bundesland-Gewichte Bevölkerung | Destatis, Tabelle 12411-0010 | Stand 2023 | Jahreswert |
| Bundesland-Gewichte Wind/Solar | Bundesnetzagentur MaStR | Stand 2024 | — |
| Feiertage | Python-Paket `holidays` | — | — |

### Datensplit (chronologisch nach vollen Kalenderjahren)

| Split | ca. Stunden | Zeitraum |
|---|---|---|
| **Train** | ~{train_hours:,} | {_fmt(train_start)} – {_fmt(train_end)} ({train_end.year - train_start.year + 1} Jahre) |
| **Validation** | ~{val_hours:,} | {_fmt(val_start)} – {_fmt(val_end)} (1 Jahr) |
| **Test** | ~{test_hours:,} | {_fmt(test_start)} – {_fmt(test_end)} (1 Jahr) |

---
""")


def _cell_feature_design() -> nbformat.NotebookNode:
    return _md("""
## 2. Feature-Design

### Past Covariates (9 Variablen) — nur historisch bekannt

| Variable | Beschreibung | Quelle |
|---|---|---|
| `total_load_lag24` | Gesamtlast vor 24 h | SMARD (Bundesnetzagentur) |
| `total_load_lag168` | Gesamtlast vor 1 Woche | SMARD (Bundesnetzagentur) |
| `solar_lag24` | Solareinspeisung vor 24 h | SMARD (Bundesnetzagentur) |
| `solar_lag168` | Solareinspeisung vor 1 Woche | SMARD (Bundesnetzagentur) |
| `wind_total_lag24` | Windeinspeisung vor 24 h | SMARD (Bundesnetzagentur) |
| `wind_total_lag168` | Windeinspeisung vor 1 Woche | SMARD (Bundesnetzagentur) |
| `weather_temperature` | Temperatur 2 m, **bevölkerungsgewichtet** | ERA5 / Open-Meteo |
| `weather_wind_100m` | Windgeschw. 100 m, **windkapazitätsgewichtet** | ERA5 / Open-Meteo |
| `weather_radiation` | Globalstrahlung, **PV-kapazitätsgewichtet** | ERA5 / Open-Meteo |

### Future Covariates (11 Variablen)

| Variable | Beschreibung | Herkunft |
|---|---|---|
| `hour_sin` / `hour_cos` | Tageszeit (zyklisch) | Kalender |
| `month_sin` / `month_cos` | Jahreszeit (zyklisch) | Kalender |
| `dow_sin` / `dow_cos` | Wochentag (zyklisch) | Kalender |
| `is_weekend` | Samstag/Sonntag = 1 | Kalender |
| `is_holiday` | Deutscher Feiertag = 1 | Kalender |
| `weather_temperature_forecast` | Temperatur-**Vorhersage**, bevölkerungsgewichtet | Open-Meteo Historical Forecast API — jüngster Modelllauf vor dem Zeitpunkt (Lead-Time typ. 0–12 h) |
| `weather_wind_100m_forecast` | Wind-**Vorhersage** 100 m, windkapazitätsgewichtet | s.o. |
| `weather_radiation_forecast` | Strahlungs-**Vorhersage**, PV-kapazitätsgewichtet | s.o. |

Empirische Forecast-Fehler gegenüber ERA5-Ist (2021–2025, deutschlandweit aggregiert):
Temperatur MAE 0,67 °C · Wind 100 m MAE 3,77 m/s · Strahlung MAE 14,5 W/m².

---
""")


def _cell_modellparameter(config: dict) -> nbformat.NotebookNode:
    tft = config["models"]["tft"]
    xgb = config["models"]["xgboost"]
    reg = config["models"]["regression"]

    # Muss mit den Defaults in build_xgboost_model / build_regression_model
    # übereinstimmen (jeweils min(72, lags), bei lags=168 also 72).
    # Wird hier nicht aus der Config gelesen, weil die Config diesen
    # Wert bewusst nicht überschreibt — die Modell-Builder sind die
    # Single Source of Truth für Covariate-Lookbacks.
    lags_past_cov = 72

    return _md(f"""
## 3. Modellparameter

Alle drei Modelle nutzen dieselbe Feature-Basis (Target-History 168 h, 9 Past- und
11 Future-Covariates) und prognostizieren dieselben 5 Quantile `{tft["quantiles"]}`.

| Modell | Paradigma | Target History (alle 168 h) | Probabilistik |
|---|---|---|---|
| Regression | Linear | ✓ | native Quantile (sklearn `QuantileRegressor`) |
| XGBoost | Gradient Boosted Trees | ✓ | native Quantile (`reg:quantileerror`) |
| TFT | Deep Learning (Transformer + LSTM) | ✓ | native Quantile (`QuantileRegression` Likelihood) |

### Temporal Fusion Transformer (TFT) — Deep Learning

| Parameter | Wert | Bedeutung |
|---|---|---|
| `input_chunk_length` | {tft["input_chunk_length"]} | Lookback-Fenster für Target + Past Covariates (1 Woche) |
| `output_chunk_length` | 24 | Prognosehorizont (Day-Ahead) |
| `hidden_size` | {tft["hidden_size"]} | Interne Repräsentationsdimension |
| `lstm_layers` | {tft["lstm_layers"]} | LSTM-Schichten für Sequence Encoding |
| `num_attention_heads` | {tft["num_attention_heads"]} | Multi-Head-Attention |
| `dropout` | {tft["dropout"]} | Regularisierung |
| `learning_rate` | {tft["learning_rate"]} | Adam-Startlernrate (ReduceLROnPlateau-Scheduler) |
| `batch_size` | {tft["batch_size"]} | Minibatch-Größe |
| `n_epochs` | {tft["n_epochs"]} (max) | Early Stopping nach {tft["early_stopping_patience"]} Epochen ohne Verbesserung |
| Likelihood | `QuantileRegression` | Quantile: {tft["quantiles"]} |

### XGBoost Gradient Boosting — Tree-based Ensemble

| Parameter | Wert | Bedeutung |
|---|---|---|
| `lags` | {xgb["lags"]} | Target-Lags (1 Woche, = Target-Anteil des TFT-Lookbacks) |
| `lags_past_covariates` | {lags_past_cov} | Past-Covariate-Lags (3 Tage, = Regression) |
| `lags_future_covariates` | (0, H) | Future Covariates im Forecast-Horizont |
| `n_estimators` | {xgb["n_estimators"]} | Bäume pro Quantil × Horizont-Schritt |
| `max_depth` | {xgb["max_depth"]} | Maximale Baumtiefe (Feature-Interaktionen) |
| `learning_rate` | {xgb["learning_rate"]} | Boosting-Schrittweite |
| `subsample` | {xgb["subsample"]} | Zeilen-Subsampling pro Baum (Stochastic GB) |
| `colsample_bytree` | {xgb["colsample_bytree"]} | Feature-Subsampling pro Baum (Dekorrelation) |
| `min_child_weight` | {xgb["min_child_weight"]} | Mindest-Hessian pro Blatt (Regularisierung) |
| `reg_lambda` | {xgb["reg_lambda"]} | L2-Regularisierung |
| `tree_method` | `{xgb["tree_method"]}` | Histogramm-basierte Splits (stabil auf CPU) |
| `multi_models` | {xgb["multi_models"]} | Direkte Mehrschritt-Strategie (= Regression) |
| Objective | `reg:quantileerror` | Native Quantile-Regression (XGBoost ≥ 2.0) |
| Likelihood | quantile (nativ) | Quantile: {xgb["quantiles"]} |

### Lineare Quantile-Regression (Baseline)

| Parameter | Wert | Bedeutung |
|---|---|---|
| `lags` | {reg["lags"]} | Target-Lags (1 Woche, = TFT / XGBoost) |
| `lags_past_covariates` | {lags_past_cov} | Past-Covariate-Lags (3 Tage, = XGBoost) |
| `lags_future_covariates` | (0, H) | Future Covariates im Forecast-Horizont |
| `multi_models` | True | Direkte Mehrschritt-Strategie (= XGBoost) |
| Likelihood | quantile (nativ) | Quantile: {reg["quantiles"]} |
| `alpha` | 0.01 | L1-Regularisierungsstärke des `QuantileRegressor` |
| `solver` | `highs` | Linear-Programming-Solver |

---
""")


def _cell_ergebnisse(results_dir: Path, target: str, horizon: str) -> nbformat.NotebookNode:
    models = ["regression", "tft", "xgboost"]
    results = {m: _load_result(results_dir, m, target, horizon) for m in models}

    horizon_label = "Day-Ahead (24h)" if horizon == "day_ahead" else "Week-Ahead (168h)"

    # ── Punktprognose-Tabelle ─────────────────────────────────────────────────
    pt_rows = []
    mae_values = {
        m: float(results[m]["MAE"])
        for m in models
        if results[m] is not None and "MAE" in results[m].index
    }
    best_mae = min(mae_values.values(), default=None)

    for m in models:
        r = results[m]
        label = _MODEL_LABELS.get(m, m.upper())
        if r is None:
            pt_rows.append(f"| {label} | — | — | — | *nicht vorhanden* |")
            continue
        mae  = float(r["MAE"])
        rmse = float(r["RMSE"]) if "RMSE" in r.index else float("nan")
        r2   = float(r["R2"]) if "R2" in r.index else float("nan")
        n    = int(r.get("n_datapoints", 0))
        best = " ⭐" if best_mae is not None and abs(mae - best_mae) < 1 else ""
        r2_str = _fmt(r2, 3) if not np.isnan(r2) else "—"
        pt_rows.append(
            f"| {label}{best} | {_fmt(mae, 0)} MW | {_fmt(rmse, 0)} MW "
            f"| {r2_str} | {n:,} |"
        )

    pt_table = "\n".join(pt_rows)

    # ── Probabilistik-Tabelle ─────────────────────────────────────────────────
    # ACI-Kalibrierung (falls schon gelaufen): bevorzugt die kalibrierten
    # Werte für Coverage/Breite anzeigen, plus Original-Werte als Vergleich.
    conformal_path = (
        results_dir / f"all_models_{target}_{horizon}_conformal_calibration.csv"
    )
    aci_info: dict[str, dict] = {}
    if conformal_path.exists():
        try:
            aci_df = pd.read_csv(conformal_path)
            for _, row in aci_df.iterrows():
                aci_info[str(row["model"])] = {
                    "cov_uncal":   float(row["test_coverage_uncalibrated"]),
                    "cov_cal":     float(row["test_coverage_calibrated"]),
                    "width_uncal": float(row["test_width_uncalibrated"]),
                    "width_cal":   float(row["test_width_calibrated_mean"]),
                }
        except Exception:
            aci_info = {}

    prob_rows = []
    for m in models:
        r = results[m]
        label = _MODEL_LABELS.get(m, m.upper())
        if r is None or "pinball_mean" not in r.index:
            prob_rows.append(f"| {label} | — | — | — | — | — |")
            continue
        pb = _fmt(r["pinball_mean"], 0)
        aci = aci_info.get(m)
        if aci:
            cov_un = f"{aci['cov_uncal']*100:.1f} %"
            cov_ca = f"{aci['cov_cal']*100:.1f} %"
            # Häkchen wenn 78–82 %
            if 0.78 <= aci["cov_cal"] <= 0.82:
                cov_ca += " ✓"
            w_un = f"{aci['width_uncal']:,.0f} MW"
            w_ca = f"{aci['width_cal']:,.0f} MW"
        else:
            # Rückfall auf die rohen Eval-Werte, wenn ACI noch nicht gelaufen
            cov_un = (
                f"{float(r['coverage_80'])*100:.1f} %"
                if "coverage_80" in r.index else "—"
            )
            cov_ca = "—"
            w_un = (
                _fmt(r.get("interval_width_80", float("nan")), 0) + " MW"
                if "interval_width_80" in r.index else "—"
            )
            w_ca = "—"
        prob_rows.append(
            f"| {label} | {pb} | {cov_un} | {cov_ca} | {w_un} | {w_ca} |"
        )

    prob_table = "\n".join(prob_rows)

    # ── Automatische Interpretation ───────────────────────────────────────────
    valid = {
        m: results[m]
        for m in models
        if results[m] is not None and "MAE" in results[m].index
    }
    if valid:
        best_m  = min(valid, key=lambda m: float(valid[m]["MAE"]))
        worst_m = max(valid, key=lambda m: float(valid[m]["MAE"]))
        best_mae_val  = float(valid[best_m]["MAE"])
        worst_mae_val = float(valid[worst_m]["MAE"])
        delta_pct = (worst_mae_val - best_mae_val) / worst_mae_val * 100
        best_label  = _MODEL_LABELS.get(best_m, best_m.upper())
        worst_label = _MODEL_LABELS.get(worst_m, worst_m.upper())
        interpretation = (
            f"> **Bestes Modell (MAE):** {best_label} mit MAE = {best_mae_val:,.0f} MW  \n"
            f"> Das beste Modell erreicht {delta_pct:.1f} % niedrigeren MAE als das schlechteste ({worst_label})."
        )
        # R²-Aussage ergänzen, falls vorhanden
        r2_vals = {
            m: float(valid[m]["R2"])
            for m in valid
            if "R2" in valid[m].index and not np.isnan(float(valid[m]["R2"]))
        }
        if r2_vals:
            best_r2_m = max(r2_vals, key=r2_vals.get)
            best_r2_label = _MODEL_LABELS.get(best_r2_m, best_r2_m.upper())
            interpretation += (
                f"  \n> **Höchstes R²:** {best_r2_label} mit R² = "
                f"{r2_vals[best_r2_m]:.3f} — erklärt rund "
                f"{r2_vals[best_r2_m]*100:.0f} % der Varianz der Residuallast."
            )
    else:
        interpretation = "> *Noch keine Ergebnisse vorhanden.*"

    return _md(f"""
## 4. Modellergebnisse — {horizon_label}

**Target:** `{target}` = Gesamtlast − Wind − Solar

### Punktprognose-Metriken

| Modell | MAE ↓ | RMSE ↓ | R² ↑ | Datenpunkte |
|---|---|---|---|---|
{pt_table}

### Probabilistische Metriken

| Modell | Pinball Loss ↓ | Cov 80 % (vor ACI) | Cov 80 % (nach ACI) | Ø Breite (vor) | Ø Breite (nach) |
|---|---|---|---|---|---|
{prob_table}

### Interpretation

{interpretation}

---
""")


def _cell_conformal(
    results_dir: Path,
    forecast_dir: Path,
    target: str,
    horizon: str,
    run_plots_dir: Path,
) -> nbformat.NotebookNode | None:
    """
    Section 6: Adaptive Conformal Inference (ACI, Gibbs & Candès 2021).

    Liefert eine Markdown-Cell mit:
      - Erläuterung des Verfahrens
      - Vergleichstabelle Cov/Breite vor vs. nach Kalibrierung
      - q̂-Verlaufs-Plot (base64-eingebettet) als Diagnose

    Gibt None zurück, falls die nötige conformal_calibration.csv fehlt
    (z.B. wenn `calibrate` noch nicht gelaufen ist) — Pipeline läuft
    rückwärtskompatibel weiter.
    """
    import base64
    from io import BytesIO

    summary_path = (
        results_dir
        / f"all_models_{target}_{horizon}_conformal_calibration.csv"
    )
    if not summary_path.exists():
        # ACI noch nicht durchgelaufen → Section überspringen.
        return None

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        return None

    # ── q̂-Verlaufs-Plot über alle Modelle ────────────────────────
    img_b64: str | None = None
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(13, 4.5), dpi=120)
        model_colors = {
            "regression": "#2ca02c",
            "tft":        "#1f77b4",
            "xgboost":    "#d62728",
        }

        for model_name in summary_df["model"].unique():
            cal_path = forecast_dir / (
                f"{model_name}_{target}_{horizon}_forecast_calibrated.csv"
            )
            if not cal_path.exists():
                continue
            df = pd.read_csv(cal_path, parse_dates=["timestamp"])
            if "q_hat_80" not in df.columns:
                continue
            df = df.sort_values("timestamp")
            ax.plot(
                df["timestamp"], df["q_hat_80"] / 1000,
                color=model_colors.get(model_name, "#666666"),
                linewidth=1.4,
                label=model_name.upper(),
                alpha=0.9,
            )

        ax.set_title(
            "Adaptive q̂ über die Testperiode",
            fontsize=12, fontweight="bold", pad=10,
        )
        ax.set_ylabel("Adaptiver Offset q̂ [GW]", fontsize=11)
        ax.set_xlabel("Zeit", fontsize=11)
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        # Auch als Datei für externe Nutzung speichern.
        run_plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = run_plots_dir / "05_aci_qhat_verlauf.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("ascii")
        plt.close()
    except Exception as exc:
        logger.warning("ACI-q̂-Plot fehlgeschlagen: %s", exc)
        img_b64 = None

    # ── Markdown-Inhalt ───────────────────────────────────────────
    lines = ["## 5. Probabilistische Kalibrierung — Adaptive Conformal Inference (ACI)\n\n"]
    lines.append(
        "Die rohen Modell-Bänder sind zu schmal — die Ist-Werte liegen seltener "
        "im 80 %-Band als angestrebt. ACI korrigiert das mit einem Zuschlag **q̂** "
        "(in MW), der das Band verbreitert:\n\n"
        "> kalibriertes Band = [ q10 − q̂ , q90 + q̂ ]\n\n"
        "Der Startwert von q̂ wird auf der Validierungsperiode bestimmt, indem "
        "gemessen wird, wie weit die Ist-Werte typischerweise außerhalb des Bands "
        "lagen. Über die Testperiode wird q̂ anschließend fortlaufend nachjustiert: "
        "nach jeder Stunde etwas vergrößert, wenn der Ist-Wert außerhalb lag, und "
        "etwas verkleinert, wenn er innerhalb lag. Da die Anpassung nach oben "
        "stärker gewichtet ist als nach unten, stellt sich q̂ genau auf das Niveau "
        "ein, bei dem 80 % der Werte im Band liegen — auch wenn sich der "
        "Testzeitraum (2025) von Training und Validierung (2021–2024) unterscheidet. "
        "In jede Anpassung fließt nur bereits beobachtete Vergangenheit ein.\n\n"
    )
    lines.append(
        "**Kernberechnung**\n\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "# Konformitätsscores auf der Validierung: wie weit lag der Ist-Wert\n"
        "# außerhalb des rohen 80 %-Bands?  (positiv = außerhalb)\n"
        "scores = np.maximum(q_lower_val - y_val, y_val - q_upper_val)\n"
        "\n"
        "alpha = 0.2                             # 80 %-Band  ->  1 - alpha = 0.8\n"
        "gamma = 0.005 * scores.std()            # Schrittweite der Anpassung\n"
        "q_hat = np.quantile(scores, 1 - alpha)  # Startwert aus der Validierung\n"
        "\n"
        "# Online über die Testperiode: q̂ nach jeder Stunde nachjustieren\n"
        "for t in range(len(y_test)):\n"
        "    lower = q_lower_test[t] - q_hat      # kalibriertes Band =\n"
        "    upper = q_upper_test[t] + q_hat      #   [q10 - q̂,  q90 + q̂]\n"
        "    ausserhalb = (y_test[t] < lower) or (y_test[t] > upper)\n"
        "    q_hat += gamma * ((1.0 if ausserhalb else 0.0) - alpha)\n"
        "```\n"
        "\n"
        "> Liegt der Ist-Wert außerhalb, wächst q̂ um `gamma·0,8`; liegt er drin, "
        "schrumpft es um `gamma·0,2`. Das Gleichgewicht dieser beiden Schritte "
        "liegt genau bei 80 % Coverage.\n\n"
    )
    lines.append("### 5.1 Vergleich vor vs. nach Kalibrierung (80 %-Band)\n\n")

    # Vergleichstabelle aufbauen (Cov Val ausgelassen — für den Vergleich nicht nötig)
    lines.append(
        "| Modell | Cov Test (vor) | Cov Test (nach ACI) "
        "| Ø Breite (vor) | Ø Breite (nach) | q̂-Range |\n"
        "|---|---|---|---|---|---|\n"
    )
    for _, row in summary_df.sort_values("model").iterrows():
        cov_un  = row["test_coverage_uncalibrated"]
        cov_cal = row["test_coverage_calibrated"]
        w_un    = row["test_width_uncalibrated"]
        w_cal   = row["test_width_calibrated_mean"]
        q_min   = row["q_hat_min"]
        q_max   = row["q_hat_max"]
        # Coverage 79–81 % als Erfolg markieren
        marker = " ✓" if 0.78 <= cov_cal <= 0.82 else ""
        lines.append(
            f"| **{str(row['model']).upper()}** | "
            f"{cov_un:.1%} | "
            f"{cov_cal:.1%}{marker} | "
            f"{w_un:,.0f} MW | "
            f"{w_cal:,.0f} MW | "
            f"[{q_min:,.0f}, {q_max:,.0f}] |\n"
        )
    lines.append("\n")

    # q̂-Verlaufs-Plot einbetten
    if img_b64:
        lines.append("### 5.2 Adaptiver Verlauf von q̂ über das Testjahr\n\n")
        lines.append(
            f'<img src="data:image/png;base64,{img_b64}" '
            'alt="q̂-Verlauf über die Test-Periode" '
            'style="max-width:100%;" />\n\n'
        )

    lines.append("---\n")

    return _md("".join(lines))


# Referenzwerte des ersten vollständigen Laufs (Iteration 1):
# 70/15/15-Ratio-Split, Test = Apr–Dez 2025 (6.528 h), OHNE Wetter-Forecasts,
# Default-Hyperparameter. Quelle: all_models_residual_load_day_ahead_comparison.csv
# des Laufs vom 21.04.2026 — als Konstante festgehalten, weil die Artefakte
# bei späteren Läufen überschrieben werden.
ITERATION_1_BASELINE = {
    "regression": {"MAE": 6762, "pinball": 2493, "cov": 0.728, "width": 23218},
    "tft":        {"MAE": 6003, "pinball": 2081, "cov": 0.764, "width": 18081},
    "xgboost":    {"MAE": 6210, "pinball": 2281, "cov": 0.563, "width": 13203},
}


def _cell_entwicklung(
    results_dir: Path,
    target: str,
    horizon: str,
) -> nbformat.NotebookNode | None:
    """
    Section 6: Entwicklung der Ergebnisse über die Methodik-Iterationen.

    Stellt drei Stufen gegenüber:
      1. Baseline      — Ratio-Split, ohne Wetter-Forecasts (Konstante)
      2. + Jahres-Splits & Wetter-Forecasts — live aus den Eval-CSVs
      3. + ACI-Kalibrierung — live aus der Conformal-CSV
    """
    models = ["regression", "tft", "xgboost"]

    # Stufe 2: aktuelle Eval-CSVs
    current = {}
    for m in models:
        r = _load_result(results_dir, m, target, horizon)
        if r is None or "MAE" not in r.index:
            continue
        current[m] = {
            "MAE":     float(r["MAE"]),
            "pinball": float(r["pinball_mean"]) if "pinball_mean" in r.index else float("nan"),
            "cov":     float(r["coverage_80"]) if "coverage_80" in r.index else float("nan"),
            "width":   float(r["interval_width_80"]) if "interval_width_80" in r.index else float("nan"),
        }
    if not current:
        return None

    # Stufe 3: ACI-Kalibrierung
    aci = {}
    conformal_path = results_dir / f"all_models_{target}_{horizon}_conformal_calibration.csv"
    if conformal_path.exists():
        try:
            cal_df = pd.read_csv(conformal_path)
            for _, row in cal_df.iterrows():
                aci[str(row["model"])] = {
                    "cov":   float(row["test_coverage_calibrated"]),
                    "width": float(row["test_width_calibrated_mean"]),
                }
        except Exception:
            aci = {}

    lines = ["## 6. Entwicklung der Ergebnisse über die Iterationen\n\n"]
    lines.append(
        "| Stufe | Datenbasis / Methodik |\n|---|---|\n"
        "| **1 — Baseline** | 70/15/15-Split (Test: Apr–Dez 2025), ohne Wetter-Forecasts |\n"
        "| **2 — Jahres-Splits + Wetter-Forecasts** | volle Kalenderjahre "
        "(Test: Jan–Dez 2025), 3 Forecast-Features als Future Covariates |\n"
        "| **3 — + ACI-Kalibrierung** | wie 2, zusätzlich Adaptive Conformal "
        "Inference auf das 80 %-Band |\n\n"
    )

    for m in models:
        if m not in current:
            continue
        label = _MODEL_LABELS.get(m, m.upper())
        b = ITERATION_1_BASELINE.get(m, {})
        c = current[m]
        a = aci.get(m, {})

        mae_delta = (
            f" ({(c['MAE'] - b['MAE']) / b['MAE'] * 100:+.0f} %)"
            if b.get("MAE") else ""
        )
        pb_delta = (
            f" ({(c['pinball'] - b['pinball']) / b['pinball'] * 100:+.0f} %)"
            if b.get("pinball") and not np.isnan(c["pinball"]) else ""
        )

        lines.append(f"**{label}**\n\n")
        lines.append(
            "| Stufe | MAE | Pinball Loss | Cov 80 % | Ø Breite |\n"
            "|---|---|---|---|---|\n"
        )
        if b:
            lines.append(
                f"| 1 Baseline | {b['MAE']:,} MW | {b['pinball']:,} "
                f"| {b['cov']:.1%} | {b['width']:,} MW |\n"
            )
        lines.append(
            f"| 2 + Splits/Forecasts | {c['MAE']:,.0f} MW{mae_delta} "
            f"| {c['pinball']:,.0f}{pb_delta} "
            f"| {c['cov']:.1%} | {c['width']:,.0f} MW |\n"
        )
        if a:
            lines.append(
                f"| 3 + ACI | {c['MAE']:,.0f} MW | {c['pinball']:,.0f} "
                f"| **{a['cov']:.1%}** | {a['width']:,.0f} MW |\n"
            )
        lines.append("\n")

    lines.append(
        "*Stufe 1 wurde auf einer kürzeren Testperiode (Apr–Dez 2025) evaluiert — "
        "die Deltas zu Stufe 2 enthalten daher neben dem Forecast-Effekt auch den "
        "Periodeneffekt. ACI (Stufe 3) verändert nur die Bänder, nicht die "
        "Punktprognose.*\n\n"
    )
    lines.append("---\n")
    return _md("".join(lines))


def _cell_visualisierungen(
    forecast_dir: Path,
    target: str,
    horizon: str,
    models: list[str],
    run_plots_dir: Path,
    run_id: str,
    existing_plots: dict[str, Path],
) -> nbformat.NotebookNode:
    """
    Baut die Visualisierungszelle mit base64-eingebetteten Bildern.
    Alle Plots sind direkt im Notebook gespeichert — keine externen Dateien nötig.
    """
    lines = ["## 7. Visualisierungen\n\n"]

    # ── 7.1 Detaillierte Wochenansicht — aus kalibrierten Forecast-CSVs ──────
    lines.append("### 7.1 Wochenansicht mit kalibriertem 80 %-Band\n\n")

    any_calibrated = any(
        (Path("output/forecasts") / f"{m}_{target}_{horizon}_forecast_calibrated.csv").exists()
        for m in models
    )
    if any_calibrated:
        lines.append(
            "> Oben: Ist (schwarz), Median (gestrichelt), ACI-kalibriertes 80 %-Band. "
            "Unten: stündlicher Fehler (Ist − Median).\n\n"
        )
    else:
        lines.append(
            "> Oben: Ist + 80 %-Band. Unten: stündlicher Fehler.\n\n"
        )

    # Vorab Daten aller Modelle für dieselbe Woche laden und gemeinsame
    # Fehler-Skala bestimmen. So vergleichbare Y-Achsen über alle Plots.
    week_data: dict[str, pd.DataFrame] = {}
    for m in models:
        df_full = _load_forecast(forecast_dir, m, target, horizon)
        if df_full is None or "q10" not in df_full.columns:
            continue
        df_sorted = df_full.sort_values("timestamp").reset_index(drop=True)
        week_start_idx = min(336, max(len(df_sorted) - 168, 0))
        week_data[m] = df_sorted.iloc[week_start_idx : week_start_idx + 168].copy()

    if week_data:
        # Maximaler absoluter Fehler über alle Modelle (in GW), als
        # symmetrisches ylim für die Fehler-Subplots.
        max_abs_err_gw = max(
            float(
                np.max(np.abs(
                    df["actual"].values - df["prediction"].values
                )) / 1000.0
            )
            for df in week_data.values()
        )
        # Etwas Padding, auf glatte Zahl runden
        err_ylim = float(np.ceil(max_abs_err_gw * 1.1))
    else:
        err_ylim = 10.0

    for m in models:
        label = _MODEL_LABELS.get(m, m.upper())
        if m not in week_data:
            lines.append(f"**{label}**\n\n*Forecast-Daten nicht vorhanden.*\n\n")
            continue

        df_week = week_data[m]
        color = _MODEL_COLORS.get(m, "#666666")
        try:
            fig, (ax_main, ax_err) = plt.subplots(
                2, 1, figsize=(16, 6.5), dpi=130, sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )

            # --- Oben: Wochenverlauf + Bänder ----------------------------------
            ax_main.plot(
                df_week["timestamp"], df_week["actual"] / 1000,
                color="black", linewidth=2.5, label="Ist-Werte", zorder=5,
            )
            ax_main.plot(
                df_week["timestamp"], df_week["prediction"] / 1000,
                color=color, linewidth=2, linestyle="--", label="Median-Prognose",
            )
            ax_main.fill_between(
                df_week["timestamp"],
                df_week["q10"] / 1000,
                df_week["q90"] / 1000,
                alpha=0.25, color=color,
                label="80 %-Band" + (" (ACI-kalibriert)" if any_calibrated else ""),
            )
            if "q10_uncal" in df_week.columns and "q90_uncal" in df_week.columns:
                ax_main.plot(
                    df_week["timestamp"], df_week["q10_uncal"] / 1000,
                    color="#c0392b", linewidth=1.8, linestyle="--", alpha=0.95,
                    label="80 %-Band (roh, unkalibriert)",
                )
                ax_main.plot(
                    df_week["timestamp"], df_week["q90_uncal"] / 1000,
                    color="#c0392b", linewidth=1.8, linestyle="--", alpha=0.95,
                )

            mae_w = float(np.mean(np.abs(
                df_week["actual"].values - df_week["prediction"].values
            )))
            ax_main.set_title(
                f"{label} — Detailansicht einer Woche im Testset "
                f"| MAE = {mae_w:,.0f} MW",
                fontsize=13, fontweight="bold",
            )
            ax_main.set_ylabel("Residuallast [GW]", fontsize=11)
            ax_main.legend(fontsize=9, loc="upper left", framealpha=0.9, ncol=2)
            ax_main.grid(True, alpha=0.3)
            ax_main.spines["top"].set_visible(False)
            ax_main.spines["right"].set_visible(False)

            # --- Unten: Stündlicher Fehler als Balken --------------------------
            errors_gw = (
                df_week["actual"].values - df_week["prediction"].values
            ) / 1000.0
            bar_colors = [color if e <= 0 else "#888888" for e in errors_gw]
            ax_err.bar(
                df_week["timestamp"], errors_gw,
                width=0.035, color=bar_colors, alpha=0.85, edgecolor="none",
            )
            ax_err.axhline(0, color="black", linewidth=0.8)
            ax_err.set_ylim(-err_ylim, err_ylim)  # einheitliche Skala!
            ax_err.set_ylabel("Fehler\n(Ist − Median) [GW]", fontsize=10)
            ax_err.set_xlabel("Zeit", fontsize=10)
            ax_err.grid(True, alpha=0.3, axis="y")
            ax_err.spines["top"].set_visible(False)
            ax_err.spines["right"].set_visible(False)
            ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d.%m."))
            ax_err.xaxis.set_major_locator(mdates.DayLocator())

            plt.tight_layout()

            # Plot speichern + base64-einbetten
            run_plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = run_plots_dir / f"07_week_detail_{m}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            lines.append(f"**{label}**\n\n")
            lines.append(_img_b64(plot_path, f"{label} Week Detail (kalibriert)"))
            lines.append("\n\n")
        except Exception as exc:
            logger.warning("Week-Plot %s fehlgeschlagen: %s", m, exc)
            plt.close("all")
            lines.append(f"**{label}**\n\n*Plot fehlgeschlagen: {exc}*\n\n")

    # ── 7.2 Fehleranalyse ─────────────────────────────────────────────────────
    lines.append("### 7.2 Fehleranalyse\n\n")
    lines.append("> Links: Fehlerverteilung gesamter Testzeitraum. "
                 "Rechts: Ø MAE nach Tageszeit.\n\n")
    if "fehler" in existing_plots:
        lines.append(_img_b64(existing_plots["fehler"], "Fehleranalyse"))
    else:
        lines.append("*Plot nicht vorhanden.*")
    lines.append("\n\n")

    # ── 7.3 Scatter ──────────────────────────────────────────────────────────
    lines.append("### 7.3 Ist vs. Vorhersage (Scatter)\n\n")
    if "scatter" in existing_plots:
        lines.append(_img_b64(existing_plots["scatter"], "Scatter Ist vs Vorhersage"))
    else:
        lines.append("*Plot nicht vorhanden.*")
    lines.append("\n\n")

    lines.append("---\n")
    return _md("".join(lines))


def _cell_uncertainty_xai(
    forecast_dir: Path,
    target: str,
    horizon: str,
    run_plots_dir: Path,
    model: str = "tft",
) -> nbformat.NotebookNode | None:
    """
    Section 8: Unsicherheits-Analyse (XAI).

    Untersucht, unter welchen Bedingungen das Modell unsicher wird — also
    wann das 80 %-Band breit ist bzw. der adaptive Zuschlag q̂ groß wird.
    Methode: Streudiagramme + Korrelation zwischen der Bandbreite und
    Wetter-/Kalender-Größen (Sonne, Wind, Temperatur, Uhrzeit, Wochentag).

    Nutzt ausschließlich vorhandene Artefakte (kalibrierte Forecast-CSV +
    aufbereitete Daten) — kein Neutraining. Gibt None zurück, wenn die
    nötigen Dateien fehlen.
    """
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt

    cal_path = forecast_dir / f"{model}_{target}_{horizon}_forecast_calibrated.csv"
    combined_path = Path("data/processed/combined_data.csv")
    if not (cal_path.exists() and combined_path.exists()):
        return None

    df = pd.read_csv(cal_path, parse_dates=["timestamp"])
    if "q90_cal80" not in df.columns or "q_hat_80" not in df.columns:
        return None

    # Wetter-Istwerte beijoinen
    comb = pd.read_csv(combined_path, index_col=0, parse_dates=True)
    if comb.index.tz is not None:
        comb.index = comb.index.tz_localize(None)
    wcols = ["weather_radiation", "weather_wind_100m", "weather_temperature"]
    comb = comb[[c for c in wcols if c in comb.columns]]

    df = df.set_index("timestamp").join(comb, how="left").reset_index()
    df = df.dropna(subset=wcols)
    if df.empty:
        return None

    # Abgeleitete Größen
    df["width_gw"] = (df["q90_cal80"] - df["q10_cal80"]) / 1000.0   # 80%-Bandbreite [GW]
    df["qhat_gw"] = df["q_hat_80"] / 1000.0
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["radiation"] = df["weather_radiation"]
    df["wind"] = df["weather_wind_100m"]
    df["temp"] = df["weather_temperature"]

    def _corr(x, y):
        try:
            r = float(np.corrcoef(x, y)[0, 1])
            return r if np.isfinite(r) else 0.0
        except Exception:
            return 0.0

    def _fig_to_b64(fig, name):
        run_plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(run_plots_dir / name, dpi=150, bbox_inches="tight")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
        return b64

    SCAT = dict(s=4, alpha=0.12, color="#1f77b4", edgecolors="none")
    weekday_labels = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    month_labels = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

    imgs = {}

    # ── Plot A: Bandbreite vs. 5 Einflussgrößen ──────────────────────────
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=120)
        # Sonne / Wind / Temperatur als Scatter mit Korrelation
        for ax, (col, xlabel, unit) in zip(
            axes[0],
            [("radiation", "Sonneneinstrahlung", "W/m²"),
             ("wind", "Windgeschw. 100 m", "m/s"),
             ("temp", "Temperatur", "°C")],
        ):
            ax.scatter(df[col], df["width_gw"], **SCAT)
            r = _corr(df[col].values, df["width_gw"].values)
            ax.set_title(f"{xlabel}   (r = {r:+.2f})", fontsize=12, fontweight="bold")
            ax.set_xlabel(f"{xlabel} [{unit}]", fontsize=10)
            ax.set_ylabel("80 %-Bandbreite [GW]", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
        # Uhrzeit (Mittel je Stunde)
        h = df.groupby("hour")["width_gw"].mean()
        axes[1, 0].plot(h.index, h.values, marker="o", color="#d62728", lw=2)
        axes[1, 0].set_title("Ø Bandbreite nach Uhrzeit", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Stunde", fontsize=10)
        axes[1, 0].set_ylabel("80 %-Bandbreite [GW]", fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].spines[["top", "right"]].set_visible(False)
        # Wochentag (Mittel je Tag)
        w = df.groupby("dow")["width_gw"].mean()
        axes[1, 1].bar([weekday_labels[i] for i in w.index], w.values, color="#2ca02c")
        axes[1, 1].set_title("Ø Bandbreite nach Wochentag", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("80 %-Bandbreite [GW]", fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis="y")
        axes[1, 1].spines[["top", "right"]].set_visible(False)
        axes[1, 2].axis("off")
        fig.suptitle(
            f"Unsicherheit ({model.upper()}): Wann wird das 80 %-Band breit?",
            fontsize=14, fontweight="bold", y=1.00,
        )
        fig.tight_layout()
        imgs["scatter"] = _fig_to_b64(fig, "08_unsicherheit_scatter.png")
    except Exception as exc:
        logger.warning("XAI-Scatter-Plot fehlgeschlagen: %s", exc)

    # ── Plot B: Bandbreite nach Stunde & Monat ───────────────────────────
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5), dpi=120)
        h = df.groupby("hour")["width_gw"].mean()
        ax1.plot(h.index, h.values, marker="o", color="#1f77b4", lw=2)
        ax1.set_title("Ø 80 %-Bandbreite nach Stunde", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Stunde", fontsize=10); ax1.set_ylabel("Bandbreite [GW]", fontsize=10)
        ax1.grid(True, alpha=0.3); ax1.spines[["top", "right"]].set_visible(False)
        m = df.groupby("month")["width_gw"].mean()
        ax2.plot([month_labels[i - 1] for i in m.index], m.values,
                 marker="o", color="#ff7f0e", lw=2)
        ax2.set_title("Ø 80 %-Bandbreite nach Monat", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Bandbreite [GW]", fontsize=10)
        ax2.grid(True, alpha=0.3); ax2.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        imgs["hourmonth"] = _fig_to_b64(fig, "08_bandbreite_stunde_monat.png")
    except Exception as exc:
        logger.warning("XAI-Stunde/Monat-Plot fehlgeschlagen: %s", exc)

    # ── Plot C: adaptiver Zuschlag q̂ vs. Wetter/Kalender ────────────────
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=120)
        for ax, (col, xlabel, unit) in zip(
            axes[:2],
            [("radiation", "Sonneneinstrahlung", "W/m²"),
             ("temp", "Temperatur", "°C")],
        ):
            ax.scatter(df[col], df["qhat_gw"], **{**SCAT, "color": "#9467bd"})
            r = _corr(df[col].values, df["qhat_gw"].values)
            ax.set_title(f"q̂ vs. {xlabel}   (r = {r:+.2f})", fontsize=12, fontweight="bold")
            ax.set_xlabel(f"{xlabel} [{unit}]", fontsize=10)
            ax.set_ylabel("q̂ [GW]", fontsize=10)
            ax.grid(True, alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
        mq = df.groupby("month")["qhat_gw"].mean()
        axes[2].plot([month_labels[i - 1] for i in mq.index], mq.values,
                     marker="o", color="#9467bd", lw=2)
        axes[2].set_title("Ø q̂ nach Monat", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("q̂ [GW]", fontsize=10)
        axes[2].grid(True, alpha=0.3); axes[2].spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        imgs["qhat"] = _fig_to_b64(fig, "08_qhat_features.png")
    except Exception as exc:
        logger.warning("XAI-q̂-Plot fehlgeschlagen: %s", exc)

    if not imgs:
        return None

    # ── Markdown ─────────────────────────────────────────────────────────
    lines = [f"## 8. Unsicherheits-Analyse ({model.upper()})\n\n"]
    lines.append(
        "Unter welchen Bedingungen wird das Modell unsicher — also wann wird "
        "das 80 %-Band breit? Die Korrelation *r* misst den linearen "
        "Zusammenhang zwischen Bandbreite und der jeweiligen Größe "
        "(nahe 0 = kein Zusammenhang, Betrag nahe 1 = starker Zusammenhang).\n\n"
    )
    if "scatter" in imgs:
        lines.append("### 8.1 Bandbreite vs. Einflussgrößen\n\n")
        lines.append(
            f'<img src="data:image/png;base64,{imgs["scatter"]}" '
            'alt="Unsicherheit vs Features" style="max-width:100%;" />\n\n'
        )
    if "hourmonth" in imgs:
        lines.append("### 8.2 Bandbreite nach Stunde und Monat\n\n")
        lines.append(
            f'<img src="data:image/png;base64,{imgs["hourmonth"]}" '
            'alt="Bandbreite nach Stunde/Monat" style="max-width:100%;" />\n\n'
        )
    if "qhat" in imgs:
        lines.append("### 8.3 Adaptiver Zuschlag q̂ vs. Wetter/Kalender\n\n")
        lines.append(
            "> q̂ ist der von ACI laufend angepasste Kalibrierungs-Zuschlag "
            "(siehe Kapitel 5). Große Werte markieren Phasen, in denen das "
            "Modell besonders korrigiert werden musste.\n\n"
        )
        lines.append(
            f'<img src="data:image/png;base64,{imgs["qhat"]}" '
            'alt="q̂ vs Features" style="max-width:100%;" />\n\n'
        )
    lines.append("---\n")
    return _md("".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Funktion
# ─────────────────────────────────────────────────────────────────────────────

def generate_run_notebook(
    config: dict,
    target: str,
    horizon: str,
    models_run: list[str],
    results_dir: Path,
    notebooks_dir: Path,
    forecast_dir: Path | None = None,
) -> Path:
    """
    Erstellt ein nummeriertes, zeitgestempeltes Jupyter-Notebook.

    Parameters
    ----------
    config : dict
        Pipeline-Konfiguration (für Modellparameter + Splits).
    target : str
        Zielvariable (z.B. "residual_load").
    horizon : str
        Horizont (z.B. "day_ahead").
    models_run : list[str]
        Welche Modelle in diesem Lauf trainiert wurden.
    results_dir : Path
        Verzeichnis mit den Evaluations-CSVs.
    notebooks_dir : Path
        Zielverzeichnis für das Notebook.
    forecast_dir : Path, optional
        Verzeichnis mit den Forecast-CSVs. Fallback: results_dir/../forecasts.

    Returns
    -------
    Path
        Pfad zum erzeugten Notebook.
    """
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    if forecast_dir is None:
        forecast_dir = results_dir.parent / "forecasts"

    ts      = datetime.now()
    run_no  = _next_run_number(notebooks_dir)
    run_id  = f"run_{run_no:03d}_{ts.strftime('%Y%m%d_%H%M')}"
    filename = f"{run_id}_{target}_{horizon}.ipynb"
    out_path = notebooks_dir / filename

    # Plots generieren (in run-spezifischem Unterordner)
    run_plots_dir = results_dir.parent / "plots" / "runs" / run_id
    existing_plots = _generate_run_plots(
        results_dir=results_dir,
        forecast_dir=forecast_dir,
        target=target,
        horizon=horizon,
        models=models_run,
        run_plots_dir=run_plots_dir,
    )

    # Notebook zusammenbauen
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

    cells = [
        _cell_title(run_id, run_no, ts, target, horizon, models_run),
        _cell_datenbasis(config),
        _cell_feature_design(),
        _cell_modellparameter(config),
        _cell_ergebnisse(results_dir, target, horizon),
    ]

    # Section 5: Conformal Calibration — nur wenn das `calibrate`-Command
    # bereits gelaufen ist (sonst rückwärtskompatibel übersprungen).
    conformal_cell = _cell_conformal(
        results_dir=results_dir,
        forecast_dir=forecast_dir,
        target=target,
        horizon=horizon,
        run_plots_dir=run_plots_dir,
    )
    if conformal_cell is not None:
        cells.append(conformal_cell)

    # Section 6: Entwicklung über die Methodik-Iterationen
    entwicklung_cell = _cell_entwicklung(results_dir, target, horizon)
    if entwicklung_cell is not None:
        cells.append(entwicklung_cell)

    cells.append(
        _cell_visualisierungen(
            forecast_dir=forecast_dir,
            target=target,
            horizon=horizon,
            models=models_run,
            run_plots_dir=run_plots_dir,
            run_id=run_id,
            existing_plots=existing_plots,
        )
    )

    # Section 8: Unsicherheits-XAI — nur wenn kalibrierte Forecasts +
    # aufbereitete Daten vorhanden sind (sonst übersprungen).
    xai_cell = _cell_uncertainty_xai(
        forecast_dir=forecast_dir,
        target=target,
        horizon=horizon,
        run_plots_dir=run_plots_dir,
        model="tft",
    )
    if xai_cell is not None:
        cells.append(xai_cell)

    cells.append(
        _md(f"""
## 9. Reproduzierbarkeit

```bash
# Diesen Lauf reproduzieren (Training + Kalibrierung + Notebook)
python main.py run --model all --target {target} --horizon {horizon}
python main.py calibrate --model all --target {target} --horizon {horizon}
```

| | |
|---|---|
| **Konfigurationsdatei** | `config/config.yaml` |
| **Random Seed** | `{config["models"].get("seed", 42)}` |
| **Plots** | `output/plots/runs/{run_id}/` |
| **Ergebnisse** | `output/results/` |
| **Forecasts** | `output/forecasts/` |
| **Generiert** | {ts.strftime("%d.%m.%Y %H:%M:%S")} |
""")
    )
    nb.cells = cells

    nbformat.write(nb, out_path)
    logger.info("Notebook gespeichert: %s", out_path)
    logger.info("Run-Plots gespeichert: %s (%d Plots)", run_plots_dir, len(existing_plots))
    print(f"\n📓 Notebook: {out_path}")
    print(f"📊 Plots:    {run_plots_dir} ({len(existing_plots)} Grafiken)")
    return out_path
