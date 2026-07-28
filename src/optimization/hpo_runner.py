"""
Orchestrierung einer Optuna-Study: anlegen/fortsetzen, optimieren, Ergebnisse
persistieren.

Persistenz:
  - Study in SQLite (output/optimization/studies/<name>.db) → unterbrechungs-
    sicher und resume-bar. Bricht ein Lauf ab (Absturz, Neustart), setzt der
    nächste Aufruf desselben Commands genau dort fort.
  - Beste Parameter als JSON (output/optimization/<name>_best_params.json)
    → von der finalen Trainings-Stufe (final-train) eingelesen.
  - Alle Trials als CSV (output/optimization/<name>_trials.csv) → Material
    für Methodik-Kapitel und Konvergenz-Plots der Thesis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import optuna

logger = logging.getLogger(__name__)

OPTIMIZATION_DIR = Path("output/optimization")
STUDIES_DIR = OPTIMIZATION_DIR / "studies"


def _study_name(model_name: str, target: str, horizon: str) -> str:
    return f"{model_name}_{target}_{horizon}"


def run_study(
    model_name: str,
    target: str,
    horizon: str,
    objective: Callable,
    n_trials: int,
    seed: int = 42,
    timeout: float | None = None,
) -> dict:
    """
    Legt eine Optuna-Study an (oder setzt sie fort) und optimiert.

    Parameters
    ----------
    model_name, target, horizon : str
        Identifizieren die Study eindeutig.
    objective : callable
        Die `objective(trial) -> float`-Funktion (aus build_objective()).
    n_trials : int
        Anzahl zusätzlicher Trials in diesem Aufruf.
    seed : int
        Seed für den TPE-Sampler (Reproduzierbarkeit).
    timeout : float, optional
        Hartes Zeitlimit in Sekunden für DIESEN Aufruf. Optuna startet nach
        Ablauf keinen neuen Trial mehr (Prüfung zwischen Trials). Zusammen mit
        den beschränkten Suchräumen (kein einzelner Trial kann pathologisch
        lange laufen) verhindert das die früher beobachteten Endlos-Läufe.

    Returns
    -------
    dict
        {study, best_params, best_value, n_trials_total, paths}.
    """
    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    name = _study_name(model_name, target, horizon)
    storage = f"sqlite:///{STUDIES_DIR / (name + '.db')}"

    # TPE-Sampler = Bayesian-artige Suche. Seed → reproduzierbare Trial-Folge.
    sampler = optuna.samplers.TPESampler(seed=seed)

    # MedianPruner: bricht aussichtslose Trials früh ab, SOFERN das Objective
    # Zwischenwerte meldet (trial.report). Für die tabellarischen Modelle ohne
    # Epochen greift er nicht — dort schützen Suchraum-Grenzen + Zeitlimit.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=name,
        storage=storage,
        direction="minimize",       # Pinball Loss minimieren
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,        # resume statt überschreiben
    )

    n_done_before = len(study.trials)
    if n_done_before > 0:
        logger.info(
            "Study '%s' fortgesetzt — %d Trials bereits vorhanden.",
            name, n_done_before,
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    # ── Ergebnisse persistieren ───────────────────────────────────────
    OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)

    best_params_path = OPTIMIZATION_DIR / f"{name}_best_params.json"
    best_payload = {
        "model": model_name,
        "target": target,
        "horizon": horizon,
        "objective": "pinball_mean_validation",
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials_total": len(study.trials),
    }
    best_params_path.write_text(
        json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    trials_csv_path = OPTIMIZATION_DIR / f"{name}_trials.csv"
    study.trials_dataframe().to_csv(trials_csv_path, index=False)

    logger.info(
        "Study '%s' fertig: %d Trials gesamt, bester Pinball(val)=%.4f",
        name, len(study.trials), study.best_value,
    )
    logger.info("Beste Parameter: %s", study.best_params)
    logger.info("Gespeichert: %s | %s", best_params_path, trials_csv_path)

    return {
        "study": study,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials_total": len(study.trials),
        "best_params_path": best_params_path,
        "trials_csv_path": trials_csv_path,
    }


def load_best_params(model_name: str, target: str, horizon: str) -> dict | None:
    """Lädt die besten Hyperparameter einer abgeschlossenen Study (oder None)."""
    name = _study_name(model_name, target, horizon)
    path = OPTIMIZATION_DIR / f"{name}_best_params.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
