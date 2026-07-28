# HPO-Läufe auf Google Colab (Laptop darf zu)

Die Hyperparameter-Optimierung läuft auf Colab-GPUs statt auf dem Laptop.
Der Lauf ist abbruchsicher: Optuna speichert jeden Trial in einer
SQLite-Datenbank (`output/optimization/studies/`), das Colab-Notebook
sichert diese alle 10 Minuten nach Google Drive und setzt beim nächsten
Start automatisch dort fort (`load_if_exists=True`).

## Einmalige Einrichtung (ca. 10 Minuten)

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
5. Fenster/Laptop schließen — mit Colab Pro läuft der Lauf im Hintergrund
   weiter und sichert sich selbst nach Drive (`MyDrive/thesis_hpo/`).

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
