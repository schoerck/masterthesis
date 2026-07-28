"""
Evaluation & Metriken
MAE, RMSE, R², Pinball Loss, CRPS und Diebold-Mariano Test.

Bewusst kein MAPE: Die Residuallast geht an PV-starken Mittagen gegen null
(teils negativ) — die Division durch Fast-Null-Istwerte bläht den Mittelwert
strukturell auf. Relative Einordnung übernimmt R².
"""

import logging

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, rmse, r2_score

logger = logging.getLogger(__name__)


def compute_point_metrics(
    actual: TimeSeries,
    predicted: TimeSeries,
) -> dict[str, float]:
    """Berechnet Punktprognose-Metriken.

    R² (Bestimmtheitsmaß) misst den erklärten Varianzanteil: 1 = perfekt,
    0 = nicht besser als der Mittelwert, < 0 = schlechter als der Mittelwert.
    """
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "R2": r2_score(actual, predicted),
    }


def pinball_loss(
    actual: np.ndarray,
    predicted_quantiles: np.ndarray,
    quantiles: list[float],
) -> dict[str, float]:
    """
    Berechnet den Pinball Loss für probabilistische Vorhersagen.

    Parameters
    ----------
    actual : np.ndarray
        Tatsächliche Werte (shape: [T]).
    predicted_quantiles : np.ndarray
        Vorhergesagte Quantile (shape: [T, Q]).
    quantiles : list[float]
        Liste der Quantile (z.B. [0.1, 0.5, 0.9]).

    Returns
    -------
    dict[str, float]
        Pinball Loss pro Quantil und Durchschnitt.
    """
    results = {}
    losses = []

    for i, q in enumerate(quantiles):
        pred = predicted_quantiles[:, i]
        error = actual - pred
        loss = np.where(error >= 0, q * error, (q - 1) * error)
        avg_loss = np.mean(loss)
        results[f"pinball_q{q}"] = avg_loss
        losses.append(avg_loss)

    results["pinball_mean"] = np.mean(losses)
    return results


def coverage_probability(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Berechnet die empirische Coverage eines Prognoseintervalls."""
    covered = np.sum((actual >= lower) & (actual <= upper))
    return covered / len(actual)


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Durchschnittliche Breite des Prognoseintervalls."""
    return np.mean(upper - lower)


def diebold_mariano_test(
    actual: np.ndarray,
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    horizon: int = 1,
) -> dict[str, float]:
    """
    Diebold-Mariano Test zum Vergleich zweier Prognosemodelle.

    H0: Beide Modelle haben gleiche Prognosegenauigkeit.
    H1: Modell 1 ist signifikant besser/schlechter als Modell 2.

    Parameters
    ----------
    actual : np.ndarray
        Tatsächliche Werte.
    pred_1, pred_2 : np.ndarray
        Vorhersagen der beiden Modelle.
    horizon : int
        Prognosehorizont (für Autokorrelations-Korrektur).

    Returns
    -------
    dict
        DM-Statistik und p-Wert.
    """
    from scipy import stats

    e1 = actual - pred_1
    e2 = actual - pred_2
    d = e1**2 - e2**2  # Differenz der quadrierten Fehler

    n = len(d)
    d_mean = np.mean(d)

    # Autokovarianz-Korrektur nach Harvey, Leybourne & Newbold (1997)
    gamma = []
    for k in range(horizon):
        gamma.append(np.mean((d[k:] - d_mean) * (d[: n - k] - d_mean)))

    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n

    if var_d <= 0:
        logger.warning("Varianz <= 0 im DM-Test. Ergebnis nicht aussagekräftig.")
        return {"dm_statistic": np.nan, "p_value": np.nan}

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)

    return {"dm_statistic": dm_stat, "p_value": p_value}


def evaluate_model(
    actual: TimeSeries,
    predicted: TimeSeries,
    model_name: str,
    horizon: str = "24h",
    quantile_predictions: np.ndarray | None = None,
    quantiles: list[float] | None = None,
) -> dict:
    """
    Vollständige Evaluation eines Modells.

    Returns
    -------
    dict
        Alle Metriken für dieses Modell.
    """
    results = {"model": model_name, "horizon": horizon}

    # Punktprognose-Metriken
    point_metrics = compute_point_metrics(actual, predicted)
    results.update(point_metrics)

    # Probabilistische Metriken
    if quantile_predictions is not None and quantiles is not None:
        actual_np = actual.values().flatten()
        pb = pinball_loss(actual_np, quantile_predictions, quantiles)
        results.update(pb)

        # Coverage für 80% Intervall (Q10 bis Q90)
        if 0.1 in quantiles and 0.9 in quantiles:
            idx_10 = quantiles.index(0.1)
            idx_90 = quantiles.index(0.9)
            cov = coverage_probability(
                actual_np,
                quantile_predictions[:, idx_10],
                quantile_predictions[:, idx_90],
            )
            width = interval_width(
                quantile_predictions[:, idx_10],
                quantile_predictions[:, idx_90],
            )
            results["coverage_80"] = cov
            results["interval_width_80"] = width

    logger.info(f"[{model_name} | {horizon}] MAE={results['MAE']:.2f}, RMSE={results['RMSE']:.2f}")
    return results


def create_comparison_table(results_list: list[dict]) -> pd.DataFrame:
    """Erstellt eine Vergleichstabelle aus einer Liste von Ergebnis-Dicts."""
    df = pd.DataFrame(results_list)
    df = df.set_index(["model", "horizon"])
    return df.round(4)
