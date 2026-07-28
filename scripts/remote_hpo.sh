#!/usr/bin/env bash
# =====================================================================
# Serieller HPO-Lauf auf einem GPU-Pod (RunPod o. ä.) — in tmux starten:
#
#   tmux new -s hpo
#   bash scripts/remote_hpo.sh
#
# Arbeitet alle vier Läufe nacheinander ab, loggt je Lauf in eine Datei
# und stoppt den RunPod-Pod am Ende selbst (SELF_STOP=0 deaktiviert das).
# Ein einzelner fehlgeschlagener Lauf bricht die übrigen nicht ab.
#
# Übersteuerbar per Umgebungsvariablen:
#   TIMEOUT   Sekunden je Lauf (Default 28800 = 8 h)
#   RUNS      Liste "modell:horizont", Default alle vier
#   SELF_STOP 1 = Pod nach Abschluss stoppen (Default), 0 = weiterlaufen
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

TIMEOUT="${TIMEOUT:-28800}"
RUNS="${RUNS:-xgboost:day_ahead tft:day_ahead xgboost:week_ahead tft:week_ahead}"
LOG_DIR="output/optimization/logs"
mkdir -p "$LOG_DIR"

echo "=== Remote-HPO gestartet: $(date '+%F %T') | Timeout je Lauf: ${TIMEOUT}s ==="
nvidia-smi -L 2>/dev/null || echo "(keine GPU gefunden — Läufe nutzen CPU)"

for spec in $RUNS; do
    model="${spec%%:*}"
    horizon="${spec##*:}"
    log="$LOG_DIR/hpo_${model}_${horizon}.log"
    echo ""
    echo "=== $(date '+%F %T')  optimize --model $model --horizon $horizon ==="
    python -u main.py optimize \
        --model "$model" \
        --target residual_load \
        --horizon "$horizon" \
        --timeout "$TIMEOUT" 2>&1 | tee "$log"
    echo "=== Lauf $model/$horizon beendet (Exit ${PIPESTATUS[0]}) ==="
done

echo ""
echo "=== Alle HPO-Läufe abgeschlossen: $(date '+%F %T') ==="
echo "Ergebnisse:"
ls -la output/optimization/ output/optimization/studies/ 2>/dev/null

# Pod selbst stoppen (nur auf RunPod wirksam): Abrechnung endet,
# das Volume mit den Ergebnissen bleibt erhalten.
if [ "${SELF_STOP:-1}" = "1" ] && command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "Stoppe Pod ${RUNPOD_POD_ID} in 60 s ... (Abbruch: Ctrl+C)"
    sleep 60
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
