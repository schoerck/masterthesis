#!/usr/bin/env bash
# final-train Restkette (Stand 14.08.2026): TFT-Wiederaufnahme + Regression.
#
# Vorgeschichte: Der Lauf vom 13./14.08. brach im finalen TFT-Fit ab
# (ReduceLROnPlateau verlangte val_loss, das es auf Train+Val nicht gibt —
# gefixt in src/models/tft_model.py). XGBoost ist komplett. Der TFT-Val-Lauf
# liegt als Cache vor, best_epoch=8 wurde aus final_train.log rekonstruiert
# (Checkpoint best-epoch=7, 0-indexiert; Early Stop nach Epoche 32).
#
# Reihenfolge bewusst TFT zuerst: kurz (~1 h GPU), ein erneuter Fehler
# wuerde frueh sichtbar. Danach Regression komplett (~20 h, CPU-lastig).

set -euo pipefail
cd /workspace/thesis
PYBIN="/root/venv/bin/python"

echo "=== TFT-Rest (Val-Cache + best_epoch=8) — $(date -u) ==="
"$PYBIN" -u main.py final-train --model tft --target residual_load --horizon day_ahead

echo "=== Regression komplett — $(date -u) ==="
"$PYBIN" -u main.py final-train --model regression --target residual_load --horizon day_ahead

echo "=== Kette fertig — $(date -u) ==="
