"""
Regressionsbaseline
Lineare Quantil-Regression als probabilistische Mehrschritt-Baseline.

Warum probabilistisch?
──────────────────────
Für einen sauberen Vergleich müssen alle drei Modelle (Regression, XGBoost,
TFT) im selben Paradigma evaluiert werden. TFT und XGBoost liefern nativ
Quantile über `likelihood="quantile"`; die Regression wird hier analog
konfiguriert (Darts `LinearRegressionModel(likelihood="quantile", ...)`),
sodass Pinball-Loss, Coverage und Intervallbreite für alle Modelle auf
derselben Basis berechnet werden.
"""

import logging

import numpy as np
from darts import TimeSeries
from darts.models import LinearRegressionModel

logger = logging.getLogger(__name__)


def build_regression_model(
    output_chunk_length: int = 24,
    lags: int = 168,
    lags_past_covariates: int | None = None,
    lags_future_covariates: tuple | None = None,
    quantiles: list[float] | None = None,
    random_state: int = 42,
    alpha: float = 0.01,
    solver: str = "highs",
    **kwargs,  # zusätzliche Parameter ignorieren (Signatur-Kompatibilität)
) -> LinearRegressionModel:
    """
    Erstellt ein LinearRegressionModel mit nativer Quantile-Regression.

    Parameters
    ----------
    output_chunk_length : int
        Prognosehorizont.
    lags : int
        Anzahl Lags der Zielvariable (Target-History im Feature-Vektor).
    lags_past_covariates : int
        Past-Covariate-Lags. Default: min(72, lags). 72 Stunden (3 Tage)
        decken den aktuellen Wetterzustand plus seine jüngste Trajektorie ab
        — identisch zur XGBoost-Konfiguration für einen fairen
        Paradigmenvergleich.
    lags_future_covariates : tuple
        (min_lag, max_lag) für Future Covariates.
    quantiles : list[float]
        Probabilistische Quantile. Default [0.1, 0.25, 0.5, 0.75, 0.9]
        → identisch zu TFT & XGBoost (fairer Pinball-Loss-Vergleich).
    random_state : int
        Seed für Reproduzierbarkeit.

    Returns
    -------
    LinearRegressionModel
        Ungetrainiertes Darts-Modell mit Quantile-Regression.
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Past Covariates: 72h Lookback deckt den aktuellen Wetterzustand plus
    # seine jüngste Trajektorie ab. Das wöchentliche Muster wird durch die
    # 168 Target-Lags abgedeckt. Identisch zur XGBoost-Konfiguration, damit
    # Regression und XGBoost sich ausschließlich im Paradigma unterscheiden.
    if lags_past_covariates is None:
        lags_past_covariates = min(72, lags)
    if lags_future_covariates is None:
        lags_future_covariates = (0, output_chunk_length)

    # multi_models=True ⇒ direkte Mehrschritt-Strategie: pro Horizont-Schritt
    # (1h, 2h, …, 24h) wird ein eigenes lineares Modell trainiert (× Quantile).
    # Jedes Modell sieht ausschließlich echte Trainingsdaten (keine eigenen
    # Predictions als Input) → keine Fehler-Akkumulation über den Horizont.
    # Identisch zur XGBoost-Konfiguration (siehe xgboost_model.py), damit
    # sich beide tabellarischen Modelle ausschließlich im Paradigma
    # unterscheiden (linear vs. tree-based Ensemble).
    #
    # Kritisch: sklearn.linear_model.QuantileRegressor hat als Default
    # `alpha=1.0` — das ist eine sehr starke L1-Regularisierung, die bei
    # unseren ~500 Features (168 Target-Lags + 3×72 Wetter-Lags + Future-
    # Covariates) den Grossteil der Koeffizienten auf Null schrumpft und
    # die Vorhersage praktisch flach macht. Die ungleiche Behandlung
    # gegenüber der OLS-Punktprognose (`LinearRegression`, komplett
    # unregularisiert) würde die Ergebnisse verzerren. Wir setzen daher
    # `alpha=0.01`: quasi-OLS, nur so viel L1, dass der HiGHS-LP-Solver
    # numerisch stabil bleibt — vergleichbar zur alten Point-Forecast-
    # Qualität, jetzt über alle fünf Quantile.
    model = LinearRegressionModel(
        output_chunk_length=output_chunk_length,
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=lags_future_covariates,
        likelihood="quantile",
        quantiles=quantiles,
        random_state=random_state,
        multi_models=True,
        # Durchgereicht an sklearn.linear_model.QuantileRegressor.
        # alpha/solver sind parametrierbar (Default = bewährte Werte), damit
        # die Hyperparameter-Optimierung (Optuna) sie variieren kann.
        alpha=alpha,
        solver=solver,
    )

    logger.info(
        "Regression erstellt: output=%d, lags_target=%d, lags_past_cov=%d, "
        "quantiles=%s, multi_models=True",
        output_chunk_length,
        lags,
        lags_past_covariates,
        quantiles,
    )
    return model


def train_regression(
    model: LinearRegressionModel,
    target_train: TimeSeries,
    past_cov_train: TimeSeries | None = None,
    future_cov_train: TimeSeries | None = None,
) -> LinearRegressionModel:
    """Trainiert das Regressionsmodell."""
    logger.info("Trainiere Regressionsmodell ...")
    model.fit(
        series=target_train,
        past_covariates=past_cov_train,
        future_covariates=future_cov_train,
    )
    logger.info("Regressionsmodell trainiert.")
    return model


def predict_regression(
    model: LinearRegressionModel,
    n: int,
    series: TimeSeries,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
    num_samples: int = 500,
) -> TimeSeries:
    """
    Erstellt probabilistische Vorhersage (Quantile-Samples).

    Parameters
    ----------
    model : LinearRegressionModel
        Trainiertes Modell mit Quantile-Likelihood.
    n : int
        Anzahl Vorhersageschritte.
    series : TimeSeries
        Historische Daten bis zum Vorhersagezeitpunkt.
    past_covariates, future_covariates : TimeSeries, optional
        Kovariaten.
    num_samples : int
        Anzahl Monte-Carlo-Samples aus der Quantile-Verteilung.

    Returns
    -------
    TimeSeries
        Probabilistische Vorhersage mit `num_samples` Samples pro Zeitschritt.
    """
    return model.predict(
        n=n,
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        num_samples=num_samples,
    )


def get_quantile_predictions(
    prediction: TimeSeries,
    quantiles: list[float] | None = None,
) -> np.ndarray:
    """Extrahiert Quantil-Vorhersagen. Shape: [T, Q]."""
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    return np.column_stack([
        prediction.quantile(q).values().flatten() for q in quantiles
    ])
