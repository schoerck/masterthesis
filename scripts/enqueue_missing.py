"""
Stellt die noch fehlenden HPO-Kombinationen als feste Warteschlange in die
Optuna-Studien ein (Runde 2 nach dem CPU-Pod, Stand 02.08.2026).

Hintergrund: Der TPE-Sampler zieht kategoriale Kombinationen mit Zurücklegen
und hat auf dem CPU-Pod viermal dieselbe Kombination bewertet. Enqueued
Trials werden vom laufenden optimize-Prozess der Reihe nach abgearbeitet,
dadurch ist die Restabdeckung des Suchraums deterministisch.

Reihenfolge: aussichtsreiche Nachbarn des bisherigen Optimums (Tiefe 8,
lr 0,05, 600 Bäume) zuerst, damit ein vorzeitiger Timeout die
informativsten Kandidaten nicht abschneidet. Bei der Regression zuerst das
schnelle alpha=0,1, dann das langsame alpha=0,001.

Aufruf (auf dem Pod, im Repo-Verzeichnis):
    python scripts/enqueue_missing.py
"""

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

XGB_QUEUE = [
    {"max_depth": 8,  "learning_rate": 0.03, "n_estimators": 400},
    {"max_depth": 8,  "learning_rate": 0.1,  "n_estimators": 600},
    {"max_depth": 8,  "learning_rate": 0.1,  "n_estimators": 400},
    {"max_depth": 10, "learning_rate": 0.05, "n_estimators": 600},
    {"max_depth": 10, "learning_rate": 0.05, "n_estimators": 400},
    {"max_depth": 10, "learning_rate": 0.03, "n_estimators": 600},
    {"max_depth": 10, "learning_rate": 0.1,  "n_estimators": 600},
    {"max_depth": 10, "learning_rate": 0.1,  "n_estimators": 400},
    {"max_depth": 6,  "learning_rate": 0.05, "n_estimators": 600},
    {"max_depth": 6,  "learning_rate": 0.05, "n_estimators": 400},
    {"max_depth": 6,  "learning_rate": 0.03, "n_estimators": 600},
    {"max_depth": 6,  "learning_rate": 0.03, "n_estimators": 400},
    {"max_depth": 6,  "learning_rate": 0.1,  "n_estimators": 600},
    {"max_depth": 6,  "learning_rate": 0.1,  "n_estimators": 400},
]

REG_QUEUE = [
    {"alpha": 0.1},
    {"alpha": 0.001},
]


def enqueue(db_path: str, queue: list) -> None:
    url = f"sqlite:///{db_path}"
    name = optuna.study.get_all_study_summaries(storage=url)[0].study_name
    study = optuna.load_study(study_name=name, storage=url)
    fertig = {tuple(sorted(t.params.items()))
              for t in study.trials if t.state.name == "COMPLETE"}
    n = 0
    for params in queue:
        if tuple(sorted(params.items())) in fertig:
            print(f"  übersprungen (schon fertig): {params}")
            continue
        study.enqueue_trial(params, skip_if_exists=False)
        n += 1
    print(f"{db_path}: {n} Trials eingereiht.")


if __name__ == "__main__":
    enqueue("output/optimization/studies/xgboost_residual_load_day_ahead.db", XGB_QUEUE)
    enqueue("output/optimization/studies/regression_residual_load_day_ahead.db", REG_QUEUE)
