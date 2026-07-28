"""
Optuna-Objective: ein Trial = Modell bauen → auf Train trainieren →
auf Validation bewerten (mittlerer Pinball Loss) → Score zurückgeben.

Wichtig (Felix' Workflow): Die Bewertung erfolgt ausschließlich auf dem
VALIDATION-Set. Das Test-Set bleibt während der gesamten HPO unangetastet
und wird erst nach der Finalisierung (final-train) ein einziges Mal verwendet.

Das Objective ist bewusst vom Rest entkoppelt: Es bekommt das Training als
Callback übergeben (vermeidet zirkuläre Importe mit main.py) und nutzt nur
die Modell-Bausteine aus model_def.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from darts import TimeSeries

from src.evaluation.metrics import pinball_loss

logger = logging.getLogger(__name__)


def evaluate_pinball_on_validation(
    model,
    datasets: dict,
    horizon_hours: int,
    quantiles: list[float],
    quantile_extractor_fn: Callable,
    eval_stride: int = 72,
    num_samples: int = 500,
) -> float:
    """
    Rolling-Backtest auf der VALIDATION-Periode → mittlerer Pinball Loss
    über alle Quantile.

    Parameters
    ----------
    model
        Bereits trainiertes Darts-Modell.
    datasets : dict
        Standard-Dataset-Dict (target_train_scaled, target_val_scaled,
        target_val, past_cov_full, future_cov_full, target_scaler).
    horizon_hours : int
        Prognosehorizont (z.B. 24).
    quantiles : list[float]
        Die fünf Quantile [0.1, 0.25, 0.5, 0.75, 0.9].
    quantile_extractor_fn : callable
        Modellspezifische Funktion, die aus einer Sample-TimeSeries die
        Quantil-Matrix [T, Q] zieht (model_def["quantile_fn"]).
    eval_stride : int
        Schrittweite des Backtests. Größer als der Horizont (z.B. 72 = alle
        3 Tage) spart Rechenzeit während der HPO, ohne die Pinball-Schätzung
        wesentlich zu verschlechtern. Für die finale Bewertung wird stride=24
        (volle Auflösung) verwendet.
    num_samples : int
        Monte-Carlo-Samples der Quantilverteilung.

    Returns
    -------
    float
        Mittlerer Pinball Loss (kleiner = besser).
    """
    target_train_scaled = datasets["target_train_scaled"]
    target_val_scaled = datasets["target_val_scaled"]
    target_val_unscaled = datasets["target_val"]
    past_cov_full = datasets["past_cov_full"]
    future_cov_full = datasets["future_cov_full"]
    scaler = datasets["target_scaler"]
    val_start = target_val_scaled.start_time()

    backtest_preds = model.historical_forecasts(
        series=target_train_scaled.append(target_val_scaled),
        past_covariates=past_cov_full,
        future_covariates=future_cov_full,
        start=val_start,
        forecast_horizon=horizon_hours,
        stride=eval_stride,
        retrain=False,
        last_points_only=False,
        num_samples=num_samples,
        verbose=False,
    )

    all_actual: list[np.ndarray] = []
    all_quant: list[np.ndarray] = []

    for pred_scaled in backtest_preds:
        if pred_scaled.n_samples <= 1:
            continue
        pred_median = pred_scaled.quantile(0.5)
        pred_unscaled = scaler.inverse_transform(pred_median)
        pred_idx = pred_unscaled.time_index
        actual_slice = target_val_unscaled.slice(pred_idx[0], pred_idx[-1])
        if len(actual_slice) == 0:
            continue

        min_len = min(len(actual_slice), len(pred_unscaled))
        q_preds = quantile_extractor_fn(pred_scaled, quantiles)
        # Quantile zurückskalieren (in MW)
        for i in range(q_preds.shape[1]):
            q_ts = TimeSeries.from_times_and_values(
                pred_scaled.time_index[:min_len],
                q_preds[:min_len, i],
            )
            q_preds[:min_len, i] = scaler.inverse_transform(q_ts).values().flatten()

        all_actual.append(actual_slice[:min_len].values().flatten())
        all_quant.append(q_preds[:min_len])

    if not all_actual:
        raise RuntimeError(
            "Keine Validation-Vorhersagen für die Pinball-Berechnung erzeugt."
        )

    actual_concat = np.concatenate(all_actual)
    quant_concat = np.concatenate(all_quant, axis=0)
    pb = pinball_loss(actual_concat, quant_concat, quantiles)
    return float(pb["pinball_mean"])


def build_objective(
    model_name: str,
    model_def: dict,
    datasets: dict,
    horizon_hours: int,
    quantiles: list[float],
    train_callback: Callable,
    eval_stride: int = 72,
):
    """
    Erzeugt die Optuna-Objective-Funktion für ein Modell.

    Parameters
    ----------
    model_name : str
        "regression" | "xgboost" | "tft".
    model_def : dict
        Eintrag aus get_model_defs() — liefert builder, quantile_fn,
        config (Basis-Hyperparameter).
    datasets : dict
        Dataset-Dict für das gewählte Target.
    horizon_hours : int
        Prognosehorizont.
    quantiles : list[float]
        Quantile.
    train_callback : callable
        Funktion `model -> trainiertes_model`. Kapselt die modellspezifische
        Trainingslogik (z.B. Regression ohne, TFT mit Validation-Set).
        Wird von main.py reingereicht, um zirkuläre Importe zu vermeiden.
    eval_stride : int
        Backtest-Stride für die Validation-Bewertung.

    Returns
    -------
    callable
        Eine `objective(trial) -> float`-Funktion für study.optimize().
    """
    from src.optimization.search_spaces import SEARCH_SPACES

    suggest_fn = SEARCH_SPACES[model_name]
    base_config = dict(model_def["config"])
    builder = model_def["builder"]
    quantile_fn = model_def["quantile_fn"]

    def objective(trial) -> float:
        # 1. Hyperparameter vorschlagen + mit Basis-Config mischen
        hp = suggest_fn(trial)
        model_cfg = {
            **base_config,
            "output_chunk_length": horizon_hours,
            **hp,
        }

        # 2. Modell bauen
        model = builder(**model_cfg)

        # 3. Auf TRAIN trainieren (modellspezifisch via Callback)
        model = train_callback(model)

        # 4. Auf VALIDATION bewerten → Pinball Loss
        score = evaluate_pinball_on_validation(
            model=model,
            datasets=datasets,
            horizon_hours=horizon_hours,
            quantiles=quantiles,
            quantile_extractor_fn=quantile_fn,
            eval_stride=eval_stride,
        )

        logger.info(
            "[Trial %d] %s | HP=%s | Pinball(val)=%.4f",
            trial.number, model_name, hp, score,
        )
        return score

    return objective
