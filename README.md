# Probabilistische Day-Ahead-Prognose der deutschen Residuallast

Code zur Masterarbeit **„Multivariate Zeitreihenprognose der Residuallast im
deutschen Stromnetz unter Einbeziehung exogener Kovariaten"**
(Berliner Hochschule für Technik, 2026, Johannes Schörck).

Die Arbeit vergleicht drei Modellklassen — lineare Quantilregression, XGBoost
und Temporal Fusion Transformer (Darts) — für die probabilistische
Day-Ahead-Prognose der deutschen Residuallast (SMARD, 2021–2025), kalibriert
die 80-%-Prognosebänder mit einer adaptiven Conformal-Kalibrierung
(CQR-Startwert + ACI-Nachführung) und analysiert die Treiber der
verbleibenden Unsicherheit.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Für GPU-Training des TFT wird `torch==2.8.0` mit passendem CUDA-Build
benötigt (siehe `scripts/run_final_rest.sh`); auf CPU läuft die Pipeline
ohne Anpassung, nur langsamer.

## Daten

Alle Quellen sind frei zugänglich. Der in der Arbeit verwendete Datenstand
liegt unter `data/` bei (SMARD-Energiereihen, ERA5-Wetteraggregat über
Open-Meteo, historische Wettervorhersagen). Ein Neuabruf ist über die
Pipeline möglich:

```bash
python main.py download     # SMARD- und Open-Meteo-APIs
python main.py preprocess   # Merkmalsaufbereitung -> data/processed/
```

## Reproduktion

```bash
python main.py pipeline     # kompletter Workflow für alle drei Modelle
python main.py optimize ... # Hyperparameter-Suche (Optuna, unterbrechungssicher)
python main.py --help       # alle Stufen und Optionen
```

Seeds sind global fixiert (`config/config.yaml`, Seed 42). Regression und
XGBoost sind damit exakt reproduzierbar; beim TFT verbleibt die in der
Arbeit dokumentierte GPU-bedingte Streuung wiederholter Trainingsläufe.

## Skripte zu den Abbildungen und Analysen der Arbeit

| Skript | Inhalt |
|---|---|
| `scripts/make_eda_figures.py` | Datenanalyse-Abbildungen (Kapitel 3.1) |
| `scripts/make_ablauf_figure.py` | Ablaufschema der Modellierung (Kapitel 3.2) |
| `scripts/make_xgb_schema.py` | Gradient-Boosting-Schema (Kapitel 3.2.3) |
| `scripts/make_kap2_figures.py` | Schemata zur Conformal-Kalibrierung (Kapitel 2.3) |
| `scripts/make_kap4_figures.py` | Ergebnis-Abbildungen und Kennzahlen (Kapitel 4) |
| `scripts/make_smard_vergleich.py` | Vergleich mit der SMARD-Residuallastprognose (Kapitel 4.1) |
| `scripts/make_gutachten_analysen.py` | γ-Sensitivität, Signifikanztests (Diebold-Mariano/Holm, Block-Bootstrap), multivariate Treiberanalyse, Monatsabdeckung, SMARD-Referenz (Anhang) |

## Struktur

```
config/          zentrale Konfiguration (Datenaufteilung, Seeds, Modellparameter)
src/data/        SMARD-/Open-Meteo-Loader, Wetteraggregation, Preprocessing
src/models/      Modelldefinitionen (Regression, XGBoost, TFT via Darts)
src/optimization/  Optuna-Zielfunktionen
data/            archivierter Rohdaten- und Merkmalsstand der Arbeit
scripts/         Abbildungs- und Analyseskripte, Remote-Trainingsskripte
notebooks/       Colab-Notebook für die Hyperparameter-Suche
```

## Lizenz

MIT, siehe `LICENSE`.
