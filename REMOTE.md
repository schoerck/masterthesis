# HPO-Läufe remote (Laptop darf zu)

Die Hyperparameter-Optimierung läuft auf gemieteter GPU statt auf dem
Laptop. Alle Wege sind abbruchsicher: Optuna speichert jeden Trial in
einer SQLite-Datenbank (`output/optimization/studies/`) und setzt beim
nächsten Start automatisch dort fort (`load_if_exists=True`).

- **Weg A (empfohlen): GPU-Pod bei RunPod** — kein Sessionlimit, stoppt
  sich nach Abschluss selbst, ca. 10–15 $ für alle vier Läufe.
- **Weg B: Google Colab** — Etappen-Modus; echtes Background nur mit
  Pro+.

## Weg A: GPU-Pod bei RunPod (empfohlen)

### Einmalig

1. Konto auf runpod.io anlegen und Guthaben laden (~20 $ reichen).
2. SSH-Key hinterlegen: RunPod → Settings → SSH Public Keys → Inhalt
   von `~/.ssh/runpod_thesis.pub` einfügen (liegt lokal bereit).

### Pod starten

1. Deploy → GPU wählen (RTX 4090, alternativ RTX 3090) → Template
   **RunPod PyTorch** → Volume ≥ 40 GB → *Deploy On-Demand*.
2. Pod-Detailseite → *Connect* → SSH-Befehl kopieren und an Claude
   geben. Ab hier übernimmt Claude: Repo klonen, Abhängigkeiten
   installieren, Lauf in `tmux` starten (`scripts/remote_hpo.sh`).
3. Laptop zu. Das Skript arbeitet alle vier Läufe seriell ab
   (Timeout je Lauf 8 h, per `TIMEOUT` übersteuerbar), loggt nach
   `output/optimization/logs/` und **stoppt den Pod danach selbst** —
   die Abrechnung endet, das Volume mit den Ergebnissen bleibt
   erhalten (~4 $/Monat, bis der Pod terminiert wird).

### Ergebnisse zurückholen

Pod in der Web-UI kurz starten → Claude zieht `output/optimization/`
per `scp` auf den Laptop → danach Pod **terminieren** (löscht Volume,
beendet alle Restkosten).

## Weg B: Google Colab

### Einmalige Einrichtung (ca. 10 Minuten)

1. **Colab Pro abonnieren** (colab.google) — nötig für *Background
   Execution*: Nur damit läuft das Notebook nach dem Schließen des
   Browsers/Laptops weiter.
2. **Privates GitHub-Repo anlegen**: github.com → New repository →
   Name z. B. `masterthesis-code`, **Private** ✓, ohne README anlegen.
   Danach lokal pushen (Befehle unten) oder Claude machen lassen.
3. **Personal Access Token (PAT) erstellen** — damit Colab das private
   Repo klonen darf: github.com → Settings → Developer settings →
   Fine-grained tokens → Generate new token → Repository access: *Only
   select repositories* → das Thesis-Repo → Permissions: *Contents:
   Read-only* → Token kopieren.
4. **Token in Colab hinterlegen**: Notebook in Colab öffnen → linke
   Seitenleiste, Schlüssel-Symbol („Secrets") → Name `GITHUB_TOKEN`,
   Wert = das PAT → „Notebook access" aktivieren.

## Pro Lauf

1. `notebooks/colab_hpo.ipynb` in Colab öffnen
   (https://colab.research.google.com/github/schoerck/masterthesis/blob/main/notebooks/colab_hpo.ipynb).
2. Oben in der Lauf-Zelle `MODEL` und `HORIZON` setzen.
3. Laufzeit → Laufzeittyp ändern → **GPU (T4)**.
4. Laufzeit → **Alle ausführen**. Beim ersten Mal Drive-Zugriff bestätigen.
5. **Achtung Tarif-Detail:** Echte *Background Execution* (Browser/Laptop
   komplett zu, Lauf läuft bis 24 h weiter) bietet nur **Colab Pro+**.
   Mit **Colab Pro** endet die Session einige Zeit nach dem Trennen der
   Verbindung. Das ist dank Resume verkraftbar: Es gehen höchstens
   ~10 Minuten seit der letzten Drive-Sicherung verloren, und ein
   erneutes „Alle ausführen" setzt die Optimierung automatisch beim
   letzten Trial fort (Etappen-Modus).

### Empfohlene Reihenfolge der Läufe

| Nr. | MODEL   | HORIZON    | erwartete Dauer (T4)  |
|-----|---------|------------|-----------------------|
| 1   | xgboost | day_ahead  | einige Stunden        |
| 2   | tft     | day_ahead  | einige Stunden (GPU!) |
| 3   | xgboost | week_ahead | einige Stunden        |
| 4   | tft     | week_ahead | einige Stunden        |

Die Regression-HPO (8 Trials, wenige Parameter) läuft problemlos lokal:
`python main.py optimize --model regression --target residual_load --horizon day_ahead`

## Ergebnisse zurückholen

Nach dem Lauf liegt alles in Google Drive unter `MyDrive/thesis_hpo/`:

- `studies/*.db` — vollständige Optuna-Studies (Resume + Analyse)
- `*_best_params.json` — beste gefundene Parameter
- `*_trials.csv` — alle Trials (Material für Kapitel 3.2.8)

Diese Dateien herunterladen und lokal nach `output/optimization/`
(Studies nach `output/optimization/studies/`) legen. Danach nutzt das
finale Training die optimierten Parameter.

## Lokales Repo zu GitHub pushen (einmalig)

```bash
git remote add origin https://github.com/schoerck/masterthesis.git
git push -u origin main
```

Bei jeder Code-Änderung vor einem Colab-Lauf: committen und pushen —
Colab klont immer den aktuellen `main`-Stand.
