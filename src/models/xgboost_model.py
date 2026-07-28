"""
XGBoost Gradient Boosting (Darts XGBModel)
Tree-based Ensemble als drittes Modellparadigma im Vergleich zu Regression und TFT.

Warum XGBoost statt LightGBM?
─────────────────────────────
XGBoost und LightGBM sind konzeptionell sehr nah (beide Gradient Boosted
Decision Trees auf Histogramm-Basis, beide unterstützen native Quantile-
Regression, beide sind in der Energieprognose-Literatur etabliert).

Auf diesem Setup (Apple Silicon, macOS, Python 3.12) kommt es jedoch bei
Darts' LightGBMModel in Kombination mit multi_models=True zu einem
reproduzierbaren SIGSEGV in einem OpenMP-Worker-Thread — betroffen sind
alle Thread-Zahlen > 0 und auch multi_models=False brachte nur Stabilität
bei einem schwerwiegenden Qualitätsproblem (rekursive Drift über
24 Prognoseschritte → MAPE > 100%).

XGBoost (ab Version 2.0, native Quantile-Regression via
`objective="reg:quantileerror"`) nutzt eine andere Thread-Infrastruktur
(`tree_method="hist"`), die auf Apple Silicon stabil läuft — auch mit
multi_models=True. Damit können wir:
  - Die wissenschaftlich stärkere **direkte Mehrschritt-Strategie** einsetzen
    (24 spezialisierte Modelle pro Quantil, keine Drift-Akkumulation)
  - Multi-Threading nutzen → deutlich schnellere Laufzeiten
  - Ergebnisqualität, die realistisch mit TFT konkurrieren kann

Architektur
───────────
Gradient Boosted Decision Trees (Chen & Guestrin 2016, KDD). Darts' XGBModel
trainiert intern einen XGBRegressor pro (Quantil × Horizont-Schritt).
Bei multi_models=True und output_chunk_length=24, 5 Quantilen ergibt das
120 Modelle insgesamt — parallelisierbar, in Minuten trainiert.

Wissenschaftliche Vergleichbarkeit
──────────────────────────────────
  - lags=168                  → identischer 1-Wochen-Lookback wie TFT & Regression
  - lags_past_covariates=72   → identisch zur Regression (3 Tage Wetter/Last-Lag,
                                 deckt aktuelle Wetterlage + jüngste Trajektorie ab)
  - lags_future_covariates=(0, H) → identisch zur Regression (Kalender im Horizont)
  - multi_models=True         → identisch zur Regression (direkte Strategie)
  - quantiles=[0.1,…,0.9]     → identisch zu TFT (fairer Pinball-Loss-Vergleich)
Einziger paradigmatischer Unterschied zu Regression: **nichtlineare Baum-Splits
plus Ensemble-Aggregation** statt linearer Gleichung. Das ist exakt die Achse,
die der Vergleich messen soll.
"""

import logging

import numpy as np
from darts import TimeSeries
from darts.models import XGBModel

logger = logging.getLogger(__name__)


def build_xgboost_model(
    output_chunk_length: int = 24,
    lags: int = 168,
    lags_past_covariates: int | None = None,
    lags_future_covariates: tuple | None = None,
    quantiles: list[float] | None = None,
    n_estimators: int = 600,
    max_depth: int = 8,
    learning_rate: float = 0.04,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    tree_method: str = "hist",
    n_jobs: int = -1,
    multi_models: bool = True,
    random_state: int = 42,
    **kwargs,  # zusätzliche/unbenutzte Parameter ignorieren (Signatur-Kompatibilität)
) -> XGBModel:
    """
    Erstellt ein XGBModel mit nativer Quantile-Regression.

    Die Hyperparameter sind bewusst auf gute Ergebnisse bei moderatem
    Rechenbudget ausgelegt — nicht auf Minimal-Baseline. Für einen sauberen
    Paradigmenvergleich ist es wichtig, dass jedes Modell in einem
    realistischen Konfigurationsbereich seiner Klasse läuft.

    Parameters
    ----------
    output_chunk_length : int
        Prognosehorizont in Stunden (24 = day-ahead, 168 = week-ahead).
    lags : int
        Target-Lags. Default 168 (= 1 Woche), identisch zu TFT & Regression.
    lags_past_covariates : int
        Past-Covariate-Lags. Default: min(72, lags). 72h (3 Tage) decken
        den aktuellen Wetterzustand plus seine jüngste Trajektorie ab.
        Das Wochenmuster liegt in den Target-Lags, nicht in den Kovariaten.
        Identisch zur Regression-Baseline für einen fairen Paradigmenvergleich.
    lags_future_covariates : tuple
        (past, future) für Future Covariates. Default (0, output_chunk_length)
        — im Forecast-Horizont werden Kalender-Features genutzt.
    quantiles : list[float]
        Probabilistische Quantile. Default [0.1, 0.25, 0.5, 0.75, 0.9]
        → identisch zu TFT (fairer Pinball-Loss-Vergleich).
    n_estimators : int
        Anzahl Boosting-Runden pro (Quantil × Horizont-Schritt) bei
        multi_models=True. Default 600 — höher als typisch, weil die
        moderate learning_rate davon profitiert.
    max_depth : int
        Maximale Baumtiefe. Default 8 — tiefer als klassisches XGBoost (6),
        weil unsere Daten viele Feature-Interaktionen haben (Wetter × Zeit
        × Last-Lags). Gepaart mit starker Regularisierung (min_child_weight,
        subsample) ohne Overfitting-Gefahr.
    learning_rate : float
        Boosting-Schrittweite. Default 0.04 — leicht unter Standard (0.05),
        kombiniert mit mehr Bäumen (600) für bessere Konvergenz.
    subsample : float
        Zufälliger Anteil der Trainings-Samples pro Baum (Stochastic GB).
        Default 0.8 — klassischer Wert, reduziert Overfitting.
    colsample_bytree : float
        Zufälliger Anteil der Features pro Baum. Default 0.8 — dekorreliert
        die Bäume im Ensemble.
    min_child_weight : int
        Mindest-Summe der Hessian-Werte pro Blatt. Default 5 — verhindert
        zu kleine Blätter, die nur auf Rauschen fitten.
    reg_alpha : float
        L1-Regularisierung. Default 0.0 — für Residuallast-Daten nicht
        nötig (kein erhofftes Feature-Sparse-Lernen).
    reg_lambda : float
        L2-Regularisierung. Default 1.0 — klassischer XGBoost-Wert,
        glättet die Score-Funktion.
    tree_method : str
        Histogramm-basierter Tree-Algorithmus. "hist" ist der schnellste
        und stabilste Modus auf CPU — auch auf Apple Silicon.
    n_jobs : int
        Parallelität. -1 = alle verfügbaren Kerne.
    multi_models : bool
        True (Default): pro Horizont-Schritt ein eigenes Modell pro Quantil
        → **direkte Mehrschritt-Strategie**, keine Fehler-Akkumulation.
        Identisch zur Regression-Baseline, damit die beiden tabellarischen
        Modelle sich nur im Paradigma unterscheiden (linear vs. tree).
    random_state : int
        Seed für Reproduzierbarkeit.

    Returns
    -------
    XGBModel
        Ungetrainiertes Darts-XGBoost-Modell mit QuantileRegression.
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    if lags_past_covariates is None:
        lags_past_covariates = min(72, lags)
    if lags_future_covariates is None:
        lags_future_covariates = (0, output_chunk_length)

    model = XGBModel(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=lags_future_covariates,
        output_chunk_length=output_chunk_length,
        likelihood="quantile",
        quantiles=quantiles,
        random_state=random_state,
        multi_models=multi_models,
        # XGBoost-Hyperparameter werden als kwargs an XGBRegressor durchgereicht
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method=tree_method,
        n_jobs=n_jobs,
        verbosity=0,  # XGBoost-interne Logs unterdrücken
    )

    logger.info(
        "XGBoost erstellt: output=%d, lags_target=%d, lags_past_cov=%d, "
        "quantiles=%s, n_estimators=%d, max_depth=%d, lr=%.3f, "
        "subsample=%.2f, colsample_bytree=%.2f, multi_models=%s, n_jobs=%d",
        output_chunk_length,
        lags,
        lags_past_covariates,
        quantiles,
        n_estimators,
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
        multi_models,
        n_jobs,
    )
    return model


def train_xgboost(
    model: XGBModel,
    target_train: TimeSeries,
    target_val: TimeSeries | None = None,
    past_cov_train: TimeSeries | None = None,
    past_cov_val: TimeSeries | None = None,
    future_cov_train: TimeSeries | None = None,
    future_cov_val: TimeSeries | None = None,
) -> XGBModel:
    """
    Trainiert das XGBoost-Modell.

    val_series wird akzeptiert, aber nicht an XGBoost weitergegeben. Darts'
    Scikit-learn-Wrapper unterstützt keine native Early-Stopping-Schleife
    für Ensemble-Quantile-Modelle — n_estimators wirkt als festes Budget.
    Die Regularisierung (subsample, colsample_bytree, min_child_weight,
    reg_lambda) ersetzt Early Stopping als Overfitting-Schutz.
    """
    logger.info("Starte XGBoost-Training ...")
    model.fit(
        series=target_train,
        past_covariates=past_cov_train,
        future_covariates=future_cov_train,
    )
    logger.info("XGBoost-Training abgeschlossen.")
    return model


def predict_xgboost(
    model: XGBModel,
    n: int,
    series: TimeSeries,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
    num_samples: int = 500,
) -> TimeSeries:
    """Erstellt probabilistische Vorhersage (Quantile-Samples)."""
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
