"""
XAI (Explainable AI) Modul
TFT Explainability (Attention, Variable Selection) und SHAP für Regression.
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries

logger = logging.getLogger(__name__)


# ============================================================
# TFT Explainability (Darts built-in)
# ============================================================

def explain_tft(
    model,
    target: TimeSeries,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
    forecast_horizon: int = 24,
) -> dict:
    """
    Erstellt TFT-Erklärungen mit dem Darts TFTExplainer.

    Parameters
    ----------
    model : TFTModel
        Trainiertes TFT-Modell.
    target : TimeSeries
        Zielzeitreihe (inkl. Vorhersagezeitraum für Kontext).
    past_covariates, future_covariates : TimeSeries, optional
        Kovariaten.
    forecast_horizon : int
        Prognosehorizont.

    Returns
    -------
    dict
        Enthält: explainability_result, encoder_importance, decoder_importance
    """
    from darts.explainability.tft_explainer import TFTExplainer

    explainer = TFTExplainer(
        model=model,
        background_series=target,
        background_past_covariates=past_covariates,
        background_future_covariates=future_covariates,
    )

    result = explainer.explain()

    encoder_imp = explainer.get_encoder_importance()
    decoder_imp = explainer.get_decoder_importance()

    logger.info("TFT Explainability berechnet.")
    logger.info(f"Encoder Importance:\n{encoder_imp}")
    logger.info(f"Decoder Importance:\n{decoder_imp}")

    return {
        "explainer": explainer,
        "result": result,
        "encoder_importance": encoder_imp,
        "decoder_importance": decoder_imp,
    }


def plot_tft_variable_selection(
    explainer,
    save_path: str | None = None,
):
    """Plottet die Variable Selection Weights des TFT."""
    fig = explainer.plot_variable_selection()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_tft_attention(
    explainer,
    save_path: str | None = None,
):
    """Plottet die Attention Weights des TFT."""
    fig = explainer.plot_attention()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_feature_importance_bar(
    importance_dict: dict,
    title: str = "Feature Importance",
    save_path: str | None = None,
):
    """
    Erstellt ein Balkendiagramm für Feature Importance (Encoder oder Decoder).

    Parameters
    ----------
    importance_dict : dict
        Dictionary {feature_name: importance_score}.
    title : str
        Titel.
    save_path : str, optional
        Speicherpfad.
    """
    if isinstance(importance_dict, pd.DataFrame):
        # TFT Explainer gibt DataFrame zurück
        importance_dict = importance_dict.iloc[0].to_dict()

    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_features)

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
    ax.barh(range(len(names)), values, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ============================================================
# SHAP Explainability (für Regression und andere sklearn-basierte Modelle)
# ============================================================

def compute_shap_values(
    model,
    X_train: np.ndarray | pd.DataFrame,
    X_explain: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Berechnet SHAP Values für ein sklearn-kompatibles Modell.

    Parameters
    ----------
    model : sklearn-like
        Trainiertes Modell mit predict() Methode.
    X_train : array-like
        Trainingsdaten (für Background-Berechnung).
    X_explain : array-like
        Daten, die erklärt werden sollen.
    feature_names : list[str], optional
        Feature-Namen.

    Returns
    -------
    dict
        shap_values, expected_value, feature_names
    """
    import shap

    # Subsample für Effizienz
    if len(X_train) > 1000:
        idx = np.random.choice(len(X_train), 1000, replace=False)
        X_bg = X_train[idx] if isinstance(X_train, np.ndarray) else X_train.iloc[idx]
    else:
        X_bg = X_train

    explainer = shap.Explainer(model, X_bg, feature_names=feature_names)
    shap_values = explainer(X_explain)

    logger.info(f"SHAP Values berechnet: {shap_values.shape}")
    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_names": feature_names,
    }


def plot_shap_summary(
    shap_result: dict,
    title: str = "SHAP Feature Importance",
    save_path: str | None = None,
):
    """Erstellt einen SHAP Summary Plot."""
    import shap

    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_result["shap_values"], show=False)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_shap_bar(
    shap_result: dict,
    title: str = "SHAP Mean Absolute Values",
    save_path: str | None = None,
):
    """Erstellt einen SHAP Bar Plot (globale Feature Importance)."""
    import shap

    fig = plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_result["shap_values"], show=False)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ============================================================
# Vergleich: Feature Importance über Modelle
# ============================================================

def compare_feature_importance(
    tft_importance: dict,
    shap_importance: dict | None = None,
    save_path: str | None = None,
):
    """
    Erstellt einen Vergleichsplot der Feature Importance über Modelle.

    Parameters
    ----------
    tft_importance : dict
        Feature Importance vom TFT (Encoder Importance).
    shap_importance : dict, optional
        Feature Importance aus SHAP (mean |SHAP|).
    save_path : str, optional
        Speicherpfad.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # TFT
    if isinstance(tft_importance, pd.DataFrame):
        tft_imp = tft_importance.iloc[0].to_dict()
    else:
        tft_imp = tft_importance

    sorted_tft = sorted(tft_imp.items(), key=lambda x: x[1], reverse=True)[:15]
    names_tft, vals_tft = zip(*sorted_tft)
    axes[0].barh(range(len(names_tft)), vals_tft, edgecolor="black", linewidth=0.5)
    axes[0].set_yticks(range(len(names_tft)))
    axes[0].set_yticklabels(names_tft)
    axes[0].invert_yaxis()
    axes[0].set_title("TFT — Variable Selection Weights")
    axes[0].set_xlabel("Importance")

    # SHAP
    if shap_importance is not None:
        import shap
        sv = shap_importance["shap_values"]
        mean_abs = np.abs(sv.values).mean(axis=0)
        feature_names = shap_importance["feature_names"] or [f"F{i}" for i in range(len(mean_abs))]

        sorted_idx = np.argsort(mean_abs)[::-1][:15]
        axes[1].barh(
            range(len(sorted_idx)),
            mean_abs[sorted_idx],
            edgecolor="black", linewidth=0.5,
        )
        axes[1].set_yticks(range(len(sorted_idx)))
        axes[1].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[1].invert_yaxis()
        axes[1].set_title("Regression — SHAP Feature Importance")
        axes[1].set_xlabel("Mean |SHAP Value|")

    plt.suptitle("Feature Importance Vergleich", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
