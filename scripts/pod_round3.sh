#!/usr/bin/env bash
# HPO Runde 3 — Restprogramm (Stand 13.08.2026, Budget 9 USD bei 0,74 USD/h ≈ 12 h).
#
# Inhalt:
#   - Regression: lambda=0.001 (in Runde 1 und 2b je >9 h ohne Abschluss),
#     jetzt mit 11-h-Deckel und weitgehend freier Maschine.
#   - XGBoost: die 7 restlichen Kombinationen aus der Warteschlange
#     ((10,0.1,400) + alle sechs depth=6), startet nach Ende des
#     letzten Runde-2b-Trials (Session hpoxgb).
#
# Budgetrechnung: spaetester Selbststopp ~09:25 UTC = 11,3 h ab Start
# = 8,40 USD von 9 USD. Abholfenster nur 20 min — die DBs werden ohnehin
# laufend per scp gesichert.
#
# Aufruf auf dem Pod:  bash scripts/pod_round3.sh

set -euo pipefail
cd /workspace/thesis

PYBIN="/root/venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="python"
POD_ID="14l06hmhelw9cv"
mkdir -p output/optimization/logs /root/.runpod
touch /root/.runpod/.runpod.yaml

# lambda=0.001 wieder einreihen (der Zombie-RUNNING aus 2b zaehlt nicht als COMPLETE)
"$PYBIN" - <<'PY'
import sqlite3, optuna
db = "/workspace/thesis/output/optimization/studies/regression_residual_load_day_ahead.db"
name = sqlite3.connect(db).execute("SELECT study_name FROM studies").fetchone()[0]
study = optuna.load_study(study_name=name, storage=f"sqlite:///{db}")
waiting = [t for t in study.get_trials(deepcopy=False) if t.state.name == "WAITING"]
if not waiting:
    study.enqueue_trial({"alpha": 0.001})
    print("eingereiht: alpha=0.001")
else:
    print("Warteschlange bereits belegt:", [t.params for t in waiting])
PY

# Regression sofort und solo-prioritaer: harter Deckel 11 h
tmux new-session -d -s hporeg3 \
  "timeout 39600 $PYBIN -u main.py optimize --model regression \
     --target residual_load --horizon day_ahead \
     --n-trials 1 --timeout 37800 2>&1 \
   | tee -a output/optimization/logs/hpo_regression_round3.log"

# XGBoost: erst nach Ende des letzten 2b-Trials, dann 7 Trials, Deckel 8 h
# ACHTUNG (Lehre 13.08.): tmux -t matcht PRAEFIXE — ohne "=" wartete die
# Schleife auf ihre eigene Session hpoxgb3 und hing endlos. Immer "=Name".
tmux new-session -d -s hpoxgb3 \
  "while tmux has-session -t =hpoxgb 2>/dev/null; do sleep 120; done; \
   timeout 28800 $PYBIN -u main.py optimize --model xgboost \
     --target residual_load --horizon day_ahead \
     --n-trials 7 --timeout 27000 2>&1 \
   | tee -a output/optimization/logs/hpo_xgboost_round3.log"

# Alten Stopper (6-h-Fenster aus Runde 2b) ersetzen: 20-min-Fenster, feste Pod-ID
tmux kill-session -t stopper 2>/dev/null || true
tmux new-session -d -s stopper3 \
  "while tmux has-session -t =hporeg3 2>/dev/null || tmux has-session -t =hpoxgb3 2>/dev/null; do sleep 300; done; \
   echo \"Runde 3 beendet \$(date -u) — Selbststopp in 20 min\" >> /workspace/thesis/output/optimization/logs/stopper.log; \
   sleep 1200; \
   runpodctl stop pod $POD_ID"

echo "Gestartet: hporeg3 (max 11 h), hpoxgb3 (wartet auf hpoxgb, dann max 8 h), stopper3 (20-min-Fenster)."
tmux ls
