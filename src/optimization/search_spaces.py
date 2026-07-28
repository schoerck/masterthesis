"""
Hyperparameter-Suchräume pro Modell für die Optuna-Optimierung.

Designprinzip (nach Felix' Empfehlung): KLEIN halten. Pro Modell nur die
2–3 wichtigsten Parameter, je 2–3 sinnvolle Kandidatenwerte. Das hält die
HPO überschaubar und vermeidet kombinatorische Explosion.

Wichtig — alle Werte sind als **kategoriale, beschränkte Mengen** definiert.
Das verhindert pathologische Ausreißer: z.B. machte beim Regression-alpha ein
zu kleiner Wert (1e-4) den LP-Solver extrem langsam (ein Trial lief 36 h).
Mit einer kategorialen Untergrenze von 0.001 kann das nicht mehr passieren.
"""

from __future__ import annotations

import optuna


def suggest_regression(trial: optuna.Trial) -> dict:
    """Regression: nur alpha (L1-Stärke). Kategoriale Werte ≥ 0.001 — verhindert
    den pathologisch langsamen LP-Solver bei sehr kleinen alpha."""
    return {
        "alpha": trial.suggest_categorical("alpha", [0.001, 0.01, 0.1]),
    }


def suggest_xgboost(trial: optuna.Trial) -> dict:
    """XGBoost: die drei einflussreichsten Boosting-Parameter."""
    return {
        "max_depth":     trial.suggest_categorical("max_depth", [6, 8, 10]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.03, 0.05, 0.1]),
        "n_estimators":  trial.suggest_categorical("n_estimators", [400, 600]),
    }


def suggest_tft(trial: optuna.Trial) -> dict:
    """TFT: bewusst klein, weil Training teuer (~5 h/Trial). Nur Kapazität
    und Regularisierung."""
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128]),
        "dropout":     trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]),
    }


# Registry — model_name → suggest-Funktion.
SEARCH_SPACES = {
    "regression": suggest_regression,
    "xgboost":    suggest_xgboost,
    "tft":        suggest_tft,
}


# Empfohlene Trial-Anzahl pro Modell (Default für die CLI, überschreibbar).
DEFAULT_N_TRIALS = {
    "regression": 3,    # nur 3 alpha-Kandidaten
    "xgboost":    15,
    "tft":        6,    # teuer → wenige Trials
}
