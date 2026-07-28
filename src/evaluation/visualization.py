"""
Visualisierung
Plots für Ergebnisse, Vorhersagen und Vergleiche.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from darts import TimeSeries


def setup_plot_style():
    """Setzt den globalen Plot-Stil für die Thesis."""
    plt.rcParams.update({
        "figure.figsize": (14, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "font.family": "serif",
    })
    sns.set_palette("colorblind")


def plot_forecast_vs_actual(
    actual: TimeSeries,
    predicted: TimeSeries,
    title: str = "Vorhersage vs. Ist-Werte",
    quantile_lower: np.ndarray | None = None,
    quantile_upper: np.ndarray | None = None,
    save_path: str | None = None,
):
    """Plottet Vorhersage gegen tatsächliche Werte mit optionalem Konfidenzintervall."""
    fig, ax = plt.subplots(figsize=(14, 6))

    actual_df = actual.to_dataframe()
    pred_df = predicted.to_dataframe()

    ax.plot(actual_df.index, actual_df.values, label="Ist-Werte", color="black", linewidth=1)
    ax.plot(pred_df.index, pred_df.values, label="Vorhersage", color="tab:blue", linewidth=1)

    if quantile_lower is not None and quantile_upper is not None:
        ax.fill_between(
            pred_df.index,
            quantile_lower.flatten(),
            quantile_upper.flatten(),
            alpha=0.3,
            color="tab:blue",
            label="80% Konfidenzintervall",
        )

    ax.set_xlabel("Zeit")
    ax.set_ylabel("Residuallast [MW]")
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "MAE",
    title: str | None = None,
    save_path: str | None = None,
):
    """Balkendiagramm zum Modellvergleich."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = comparison_df[metric].unstack(level="horizon")
    df_plot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)

    ax.set_ylabel(metric)
    ax.set_title(title or f"Modellvergleich: {metric}")
    ax.set_xlabel("")
    ax.legend(title="Horizont")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_residuals(
    actual: TimeSeries,
    predicted: TimeSeries,
    model_name: str = "",
    save_path: str | None = None,
):
    """Residuenanalyse (Fehlerverteilung und zeitlicher Verlauf)."""
    residuals = actual.values().flatten() - predicted.values().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Zeitlicher Verlauf
    axes[0].plot(residuals, linewidth=0.5)
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"Residuen über Zeit — {model_name}")
    axes[0].set_ylabel("Fehler [MW]")

    # Histogramm
    axes[1].hist(residuals, bins=50, edgecolor="black", linewidth=0.5)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title("Fehlerverteilung")
    axes[1].set_xlabel("Fehler [MW]")

    # QQ-Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("QQ-Plot")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_seasonal_error(
    actual: TimeSeries,
    predicted: TimeSeries,
    model_name: str = "",
    save_path: str | None = None,
):
    """Zeigt MAE aufgeschlüsselt nach Stunde und Monat."""
    actual_df = actual.to_dataframe()
    pred_df = predicted.to_dataframe()

    errors = np.abs(actual_df.values.flatten() - pred_df.values.flatten())
    idx = actual_df.index

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Nach Stunde
    hourly_mae = pd.Series(errors, index=idx).groupby(idx.hour).mean()
    axes[0].bar(hourly_mae.index, hourly_mae.values, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Stunde des Tages")
    axes[0].set_ylabel("MAE [MW]")
    axes[0].set_title(f"MAE nach Tageszeit — {model_name}")

    # Nach Monat
    monthly_mae = pd.Series(errors, index=idx).groupby(idx.month).mean()
    axes[1].bar(monthly_mae.index, monthly_mae.values, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Monat")
    axes[1].set_ylabel("MAE [MW]")
    axes[1].set_title(f"MAE nach Monat — {model_name}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
