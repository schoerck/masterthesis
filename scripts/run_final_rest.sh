#!/usr/bin/env bash
# Starter fuer die final-train-Restkette (TFT-Rest + Regression).
# Aufruf auf dem Pod:  bash scripts/run_final_rest.sh
#
# Baut bei Bedarf das venv neu auf (Container-Disk ueberlebt den Pod-Stop
# nicht), verweigert den Start ohne GPU und startet die Kette in tmux.
# KEIN Selbststopp (Entscheidung 14.08.2026): Johannes stoppt den Pod
# manuell nach Statusabfrage.

set -euo pipefail
cd /workspace/thesis
mkdir -p output/final

if [ ! -x /root/venv/bin/python ]; then
  echo "venv fehlt — Neuaufbau aus requirements.txt ..."
  python -m venv /root/venv
  /root/venv/bin/pip install --upgrade pip
  /root/venv/bin/pip install -r requirements.txt
  # torch-Drift-Schutz (Lehre 14.08.2026): requirements pinnt nur >=2.0,
  # pip zieht sonst das neueste cu130-Wheel, dessen CUDA 13 der Pod-Treiber
  # (570.x) nicht unterstuetzt -> torch.cuda.is_available() == False.
  # Deshalb auf die treiberkompatible Version des Pod-Images festnageln.
  /root/venv/bin/pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
fi
PYBIN="/root/venv/bin/python"

# Der TFT-Fit braucht die GPU — auf einem 0-GPU-Start nicht loslaufen.
"$PYBIN" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("ABBRUCH: keine GPU verfuegbar — Pod mit GPU starten.")
print("GPU:", torch.cuda.get_device_name(0))
PY

tmux new-session -d -s finalrest \
  "cd /workspace/thesis && timeout 108000 bash scripts/run_final_rest_chain.sh 2>&1 | tee -a output/final/final_rest.log; echo \"final-rest-Session beendet: \$(date -u)\" | tee -a output/final/final_rest.log"

echo "Gestartet: finalrest (TFT→Regression, Deckel 30 h). Kein Selbststopp — Pod manuell stoppen."
tmux ls
