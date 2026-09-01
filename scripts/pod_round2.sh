#!/usr/bin/env bash
# HPO Runde 2 auf frischem GPU-Pod (Stand 02.08.2026).
#
# Voraussetzungen auf dem Pod:
#   - Repo unter /workspace/thesis, Abhängigkeiten installiert
#   - lokale Studien-DBs nach output/optimization/studies/ hochgeladen
#   - Warteschlange eingereiht:  python scripts/enqueue_missing.py
#   - runpodctl konfiguriert (für den Selbststopp): im Web-Terminal
#     einmalig `runpodctl config --apiKey=...` ausführen
#
# Lehren aus Runde 1, hier eingebaut:
#   - GNU `timeout` um beide Läufe: beendet auch einen hängenden
#     LP-Solver hart, damit der Selbststopp IMMER feuert
#     (der Optuna-Timeout verhindert nur neue Trials).
#   - Regression zuerst alpha=0.1 (schnell), dann alpha=0.001 (langsam),
#     Reihenfolge steckt in der Warteschlange.
#
# Aufruf:  bash scripts/pod_round2.sh

set -euo pipefail
cd /workspace/thesis

PYBIN="${PYBIN:-python}"
mkdir -p output/optimization/logs /root/.runpod
touch /root/.runpod/.runpod.yaml   # bekannter runpodctl-Henne-Ei-Fix

# XGBoost: 14 eingereihte Kombinationen, harter Deckel 16 h
tmux new-session -d -s hpoxgb \
  "timeout 57600 $PYBIN -u main.py optimize --model xgboost \
     --target residual_load --horizon day_ahead \
     --n-trials 14 --timeout 54000 2>&1 \
   | tee -a output/optimization/logs/hpo_xgboost_round2.log"

# Regression: 2 eingereihte Alphas, harter Deckel 12 h
tmux new-session -d -s hporeg \
  "timeout 43200 $PYBIN -u main.py optimize --model regression \
     --target residual_load --horizon day_ahead \
     --n-trials 2 --timeout 39600 2>&1 \
   | tee -a output/optimization/logs/hpo_regression_round2.log"

# Selbststopp, sobald beide Sessions beendet sind — mit 6 h Abholfenster:
# Lehre aus Runde 2a: Ein gestoppter Pod ließ sich wegen belegter
# Host-Ressourcen nicht mehr starten, die Ergebnisse waren gefangen.
# Das Fenster gibt Zeit, die DBs per scp zu sichern, bevor der Pod stoppt.
tmux new-session -d -s stopper \
  'while tmux has-session -t hpoxgb 2>/dev/null || tmux has-session -t hporeg 2>/dev/null; do sleep 300; done; \
   echo "Läufe beendet $(date -u) — Selbststopp in 6 h (Abholfenster)" >> /workspace/thesis/output/optimization/logs/stopper.log; \
   sleep 21600; \
   ID=$(runpodctl get pod 2>/dev/null | awk "NR==2{print \$1}"); \
   [ -n "$ID" ] && runpodctl stop pod $ID'

echo "Gestartet: hpoxgb (max 16 h), hporeg (max 12 h), stopper."
tmux ls
