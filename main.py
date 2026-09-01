"""
Main Pipeline — Residuallast-Prognose
Steuert Datenbeschaffung, Preprocessing, Modelltraining, Evaluation und Forecasting.
"""

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from darts import TimeSeries
from darts.models import LinearRegressionModel, TFTModel, XGBModel

from src.data.smard_loader import load_all_smard_data
from src.data.weather_loader import load_all_weather_data
from src.data.weather_forecast_loader import load_all_weather_forecast_data
from src.data.preprocessing import (
    add_lag_features,
    add_time_features,
    combine_data,
    create_darts_datasets,
    handle_missing_values,
    split_data,
)
from src.evaluation.metrics import create_comparison_table, pinball_loss
from src.reporting.notebook_generator import generate_run_notebook
from src.evaluation.visualization import plot_forecast_vs_actual, setup_plot_style
from src.models.xgboost_model import (
    build_xgboost_model,
    get_quantile_predictions as xgb_quantiles,
    predict_xgboost,
    train_xgboost,
)
from src.models.regression_model import (
    build_regression_model,
    get_quantile_predictions as reg_quantiles,
    predict_regression,
    train_regression,
)
from src.models.tft_model import (
    build_tft_model,
    get_quantile_predictions as tft_quantiles,
    predict_tft,
    train_tft,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("models/trained")
EVALUATION_DIR = Path("output/results")
FORECAST_DIR = Path("output/forecasts")
ARTIFACT_VERSION = 1

ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_MAGENTA = "\033[95m"


def load_config(path: str = "config/config.yaml") -> dict:
    """Lädt die Konfiguration."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_global_seeds(seed: int) -> None:
    """Setzt alle relevanten Zufalls-Seeds für Reproduzierbarkeit.

    Umfasst: Python `random`, NumPy, `PYTHONHASHSEED`, PyTorch (CPU + CUDA).
    XGBoost- und Darts-Modelle erhalten ihren Seed zusätzlich über den
    jeweiligen `random_state`-Parameter in `get_model_defs()` — diese Funktion
    deckt die globalen Pfade ab (Lightning Trainer, Dropout-Masken, NumPy-
    Sampling in `historical_forecasts`).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Globale Seeds gesetzt (seed=%d).", seed)


def ensure_output_dirs() -> None:
    """Stellt sicher, dass Artefakt- und Ausgabeordner existieren."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)


def supports_color() -> bool:
    """Prüft, ob ANSI-Farben im aktuellen Terminal sinnvoll sind."""
    return sys.stdout.isatty() and not bool(getattr(sys.stdout, "closed", False))


def color_text(text: str, color: str) -> str:
    """Färbt Text im Terminal, falls ANSI unterstützt wird."""
    if not supports_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def format_duration(seconds: float) -> str:
    """Formatiert Sekunden kompakt als h/m/s."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_clock_duration(seconds: float) -> str:
    """Formatiert Sekunden als laufende Stoppuhr HH:MM:SS."""
    total_seconds = int(max(0, seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_timed_status(command_name: str, status: str, color: str, elapsed: float | None = None) -> None:
    """Gibt gut sichtbare Statuszeilen fuer lange Laeufe aus."""
    clear_live_status_line()
    if elapsed is None:
        message = f"[{command_name}] {status}"
    else:
        message = f"[{command_name}] {status} | Dauer: {format_duration(elapsed)}"
    print(color_text(message, color), flush=True)


def format_bytes_gb(num_bytes: int) -> str:
    """Formatiert Bytes als GB mit einer Nachkommastelle."""
    return f"{num_bytes / (1024 ** 3):.1f} GB"


_COMMAND_START_TIME = 0.0
_LAST_STATUS_WIDTH = 0


def get_job_process_stats() -> tuple[float, int]:
    """Liefert CPU- und RAM-Nutzung fuer Hauptprozess plus Kindprozesse."""
    root_process = psutil.Process(os.getpid())
    processes = [root_process]

    try:
        processes.extend(root_process.children(recursive=True))
    except Exception:
        pass

    cpu_percent = 0.0
    memory_bytes = 0

    for process in processes:
        try:
            cpu_percent += process.cpu_percent(interval=None)
            memory_bytes += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return cpu_percent, memory_bytes


def build_live_status_line(command_name: str) -> str:
    """Erzeugt die kombinierte Live-Statuszeile fuer Stoppuhr und Ressourcen."""
    elapsed = time.perf_counter() - _COMMAND_START_TIME
    cpu_percent_total = psutil.cpu_percent(interval=None)
    cpu_percent_job, memory_bytes_job = get_job_process_stats()
    memory = psutil.virtual_memory()
    ram_percent = (memory.used / memory.total) * 100 if memory.total else 0.0

    message = (
        f"[{command_name}] "
        f"{format_clock_duration(elapsed)} | "
        f"CPU gesamt {cpu_percent_total:.0f}% | "
        f"CPU job {cpu_percent_job:.0f}% | "
        f"RAM gesamt {ram_percent:.0f}% ({format_bytes_gb(memory.used)}/{format_bytes_gb(memory.total)}) | "
        f"RAM job {format_bytes_gb(memory_bytes_job)}"
    )
    return color_text(message, ANSI_MAGENTA)


def clear_live_status_line() -> None:
    """Entfernt die aktuelle Live-Statuszeile aus dem Terminal."""
    global _LAST_STATUS_WIDTH
    try:
        sys.stdout.write("\r" + " " * _LAST_STATUS_WIDTH + "\r")
        sys.stdout.flush()
    except Exception:
        pass
    _LAST_STATUS_WIDTH = 0


def run_live_status(command_name: str, stop_event: threading.Event, interval_seconds: float = 1.0) -> None:
    """Aktualisiert eine einzige Live-Zeile waehrend des Laufs."""
    global _LAST_STATUS_WIDTH
    if not supports_color():
        return

    psutil.cpu_percent(interval=None)
    get_job_process_stats()
    while not stop_event.wait(interval_seconds):
        try:
            message = build_live_status_line(command_name)
            visible_width = len(
                f"[{command_name}] "
                f"{format_clock_duration(time.perf_counter() - _COMMAND_START_TIME)} | "
                f"CPU gesamt 100% | CPU job 100% | RAM gesamt 100% (000.0 GB/000.0 GB) | RAM job 000.0 GB"
            )
            _LAST_STATUS_WIDTH = max(_LAST_STATUS_WIDTH, visible_width)
            sys.stdout.write("\r" + message + " " * max(0, _LAST_STATUS_WIDTH - len(message)))
            sys.stdout.flush()
        except Exception:
            return

    clear_live_status_line()


def run_periodic_status_lines(
    command_name: str,
    stop_event: threading.Event,
    interval_seconds: float = 30.0,
) -> None:
    """Gibt periodisch Ressourceninfos als normale Zeilen aus."""
    psutil.cpu_percent(interval=None)
    get_job_process_stats()

    while not stop_event.wait(interval_seconds):
        try:
            print(color_text(build_live_status_line(command_name), ANSI_MAGENTA), flush=True)
        except Exception:
            return


def get_model_defs(config: dict) -> dict:
    """Erstellt eine zentrale Definition aller Modelltypen."""
    model_cfg = config["models"]
    return {
        "regression": {
            "builder": build_regression_model,
            "trainer": train_regression,
            "predictor": predict_regression,
            "quantile_fn": reg_quantiles,
            "model_version": 5,  # v5: + Wetter-Forecasts als Future Covariate
            "config": {
                "lags": model_cfg["regression"]["lags"],
                "quantiles": model_cfg["regression"]["quantiles"],
                "random_state": model_cfg["seed"],
            },
            "loader_class": LinearRegressionModel,
        },
        "tft": {
            "builder": build_tft_model,
            "trainer": train_tft,
            "predictor": predict_tft,
            "quantile_fn": tft_quantiles,
            "model_version": 2,  # v2: + Wetter-Forecasts als Future Covariate
            "config": {
                "input_chunk_length": model_cfg["tft"]["input_chunk_length"],
                "hidden_size": model_cfg["tft"]["hidden_size"],
                "lstm_layers": model_cfg["tft"]["lstm_layers"],
                "num_attention_heads": model_cfg["tft"]["num_attention_heads"],
                "dropout": model_cfg["tft"]["dropout"],
                "batch_size": model_cfg["tft"]["batch_size"],
                "n_epochs": model_cfg["tft"]["n_epochs"],
                "early_stopping_patience": model_cfg["tft"].get("early_stopping_patience", 10),
                "learning_rate": model_cfg["tft"]["learning_rate"],
                "quantiles": model_cfg["tft"]["quantiles"],
                "random_state": model_cfg["seed"],
            },
            "loader_class": TFTModel,
        },
        "xgboost": {
            "builder": build_xgboost_model,
            "trainer": train_xgboost,
            "predictor": predict_xgboost,
            "quantile_fn": xgb_quantiles,
            "model_version": 2,  # v2: + Wetter-Forecasts als Future Covariate
            "config": {
                "lags": model_cfg["xgboost"]["lags"],
                "n_estimators": model_cfg["xgboost"]["n_estimators"],
                "max_depth": model_cfg["xgboost"]["max_depth"],
                "learning_rate": model_cfg["xgboost"]["learning_rate"],
                "subsample": model_cfg["xgboost"]["subsample"],
                "colsample_bytree": model_cfg["xgboost"]["colsample_bytree"],
                "min_child_weight": model_cfg["xgboost"]["min_child_weight"],
                "reg_alpha": model_cfg["xgboost"]["reg_alpha"],
                "reg_lambda": model_cfg["xgboost"]["reg_lambda"],
                "tree_method": model_cfg["xgboost"]["tree_method"],
                "n_jobs": model_cfg["xgboost"].get("n_jobs", -1),
                "multi_models": model_cfg["xgboost"].get("multi_models", True),
                "quantiles": model_cfg["xgboost"]["quantiles"],
                "random_state": model_cfg["seed"],
            },
            "loader_class": XGBModel,
        },
    }


def get_horizon_hours(config: dict, horizon_name: str) -> int:
    """Liefert die Stundenzahl für einen Prognosehorizont."""
    horizons = config["models"]["forecast_horizons"]
    if horizon_name not in horizons:
        raise ValueError(
            f"Unbekannter Horizont '{horizon_name}'. "
            f"Verfügbar: {', '.join(horizons.keys())}"
        )
    return horizons[horizon_name]


def get_target_datasets(datasets_direct: dict, datasets_indirect: dict, target_name: str) -> dict:
    """Wählt den passenden Darts-Datensatz für ein Ziel aus."""
    if target_name == "residual_load":
        return datasets_direct
    if target_name not in datasets_indirect:
        raise ValueError(
            f"Unbekanntes Target '{target_name}'. "
            "Verfügbar: residual_load, total_load, solar, wind_total"
        )
    return datasets_indirect[target_name]


def build_artifact_stem(model_name: str, target_name: str, horizon_name: str) -> str:
    """Erzeugt einen konsistenten Dateinamen für Artefakte."""
    return f"{model_name}_{target_name}_{horizon_name}"


def get_artifact_paths(model_name: str, target_name: str, horizon_name: str) -> tuple[Path, Path]:
    """Liefert Pfade für Modellartefakt und Metadaten."""
    stem = build_artifact_stem(model_name, target_name, horizon_name)
    return ARTIFACTS_DIR / f"{stem}.pkl", ARTIFACTS_DIR / f"{stem}.json"


def save_model_artifact(
    model,
    model_name: str,
    target_name: str,
    horizon_name: str,
    model_version: int,
) -> tuple[Path, Path]:
    """Speichert Modell und Metadaten."""
    model_path, meta_path = get_artifact_paths(model_name, target_name, horizon_name)
    model.save(str(model_path))

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": model_name,
        "target_name": target_name,
        "horizon_name": horizon_name,
        "model_version": model_version,
        "python_class": type(model).__name__,
        "model_path": str(model_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, meta_path


def load_model_artifact(config: dict, model_name: str, target_name: str, horizon_name: str):
    """Lädt ein gespeichertes Modell basierend auf Metadaten."""
    model_path, meta_path = get_artifact_paths(model_name, target_name, horizon_name)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Kein gespeichertes Modell gefunden: {model_path}. "
            "Bitte zuerst trainieren."
        )

    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    model_defs = get_model_defs(config)
    model_def = model_defs[model_name]
    expected_model_version = model_def["model_version"]

    if not metadata:
        raise ValueError(
            f"Fuer {model_name}/{target_name}/{horizon_name} fehlen Metadaten. "
            f"Bitte Modell neu trainieren: {model_path}"
        )

    artifact_version = metadata.get("artifact_version")
    if artifact_version != ARTIFACT_VERSION:
        raise ValueError(
            f"Artefaktformat veraltet fuer {model_name}/{target_name}/{horizon_name}. "
            f"Erwartet Version {ARTIFACT_VERSION}, gefunden {artifact_version}. "
            "Bitte Modell neu trainieren."
        )

    saved_model_version = metadata.get("model_version")
    if saved_model_version != expected_model_version:
        raise ValueError(
            f"Gespeichertes Modell ist veraltet fuer {model_name}/{target_name}/{horizon_name}. "
            f"Erwartet Modellversion {expected_model_version}, gefunden {saved_model_version}. "
            "Bitte Modell neu trainieren."
        )

    class_name = metadata.get("python_class", "")

    if class_name == "TFTModel":
        return TFTModel.load(str(model_path), weights_only=False)
    if class_name == "LinearRegressionModel":
        return LinearRegressionModel.load(str(model_path))
    if class_name == "XGBModel":
        return XGBModel.load(str(model_path))

    loader_class = model_def["loader_class"]
    if loader_class is None:
        raise ValueError(
            f"Das Modell '{model_name}' benötigt Metadaten zum Laden. "
            f"Bitte das Modell neu trainieren: {model_path}"
        )
    return loader_class.load(str(model_path))


def step_1_download_data(config: dict):
    """Lädt SMARD-, Wetter-Ist- und Wetter-Forecast-Daten herunter oder
    verwendet vorhandene CSVs."""
    logger.info("=" * 60)
    logger.info("SCHRITT 1: DATENBESCHAFFUNG")
    logger.info("=" * 60)

    raw_path = config["data"]["paths"]["raw"]
    smard_cfg = config["data"]["smard"]
    weather_cfg = config["data"]["weather"]

    smard_file = Path(raw_path) / "smard_data.csv"
    if smard_file.exists():
        logger.info(f"SMARD-Daten bereits vorhanden: {smard_file}")
        smard = pd.read_csv(smard_file, index_col=0, parse_dates=True)
    else:
        logger.info("Lade SMARD-Daten frisch aus der API ...")
        smard = load_all_smard_data(
            region=smard_cfg["region"],
            resolution=smard_cfg["resolution"],
            start_date=smard_cfg["start_date"],
            end_date=smard_cfg["end_date"],
            save_path=str(smard_file),
        )

    weather_file = Path(raw_path) / "weather_data.csv"
    if weather_file.exists():
        logger.info(f"Wetterdaten (ERA5-Ist) bereits vorhanden: {weather_file}")
        weather = pd.read_csv(weather_file, index_col=0, parse_dates=True)
    else:
        logger.info("Lade ERA5-Wetterdaten frisch aus der API ...")
        weather = load_all_weather_data(
            variables=weather_cfg["variables"],
            start_date=weather_cfg["start_date"],
            end_date=weather_cfg["end_date"],
            save_path=str(weather_file),
        )

    # Historische Wetterforecasts — als Future Covariate für Day-Ahead
    # Prognosen. Methodisch wichtig: historische Forecasts (mit echten
    # Vorhersagefehlern) statt ERA5-Ist, sonst hätte das Modell perfekten
    # Wetterblick in die Zukunft.
    weather_forecast_file = Path(raw_path) / "weather_forecast_data.csv"
    if weather_forecast_file.exists():
        logger.info(
            f"Wetter-Forecasts bereits vorhanden: {weather_forecast_file}"
        )
        weather_forecast = pd.read_csv(
            weather_forecast_file, index_col=0, parse_dates=True
        )
    else:
        logger.info(
            "Lade historische Wetter-Forecasts (Open-Meteo) frisch aus der API ..."
        )
        weather_forecast = load_all_weather_forecast_data(
            variables=weather_cfg["variables"],
            start_date=weather_cfg["start_date"],
            end_date=weather_cfg["end_date"],
            save_path=str(weather_forecast_file),
        )

    return smard, weather, weather_forecast


def step_2_preprocessing(
    config: dict,
    smard: pd.DataFrame,
    weather: pd.DataFrame,
    weather_forecast: pd.DataFrame,
):
    """Preprocessing: Kombinieren, Features, Split und Darts-Objekte.

    Drei Datenströme:
      smard            — SMARD-Energiedaten (Residuallast, Last, Wind, Solar)
      weather          — ERA5-Ist-Werte (gewichtete Aggregate)        → Past Cov
      weather_forecast — Open-Meteo historische Forecasts (Aggregate) → Future Cov
    """
    logger.info("=" * 60)
    logger.info("SCHRITT 2: PREPROCESSING")
    logger.info("=" * 60)

    prep_cfg = config["preprocessing"]

    combined = combine_data(
        smard, weather, prep_cfg["target_frequency"], weather_forecast
    )
    combined = add_time_features(combined)
    combined = add_lag_features(combined)
    combined = handle_missing_values(
        combined,
        prep_cfg["interpolation_method"],
        prep_cfg["max_gap_hours"],
    )

    processed_path = Path(config["data"]["paths"]["processed"]) / "combined_data.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(processed_path)
    logger.info(f"Aufbereitete Daten gespeichert: {processed_path}")

    train, val, test = split_data(
        combined,
        prep_cfg["train_end_date"],
        prep_cfg["val_end_date"],
    )

    datasets_direct = create_darts_datasets(
        train,
        val,
        test,
        target_col="residual_load",
    )

    datasets_indirect = {}
    for target_name in ["total_load", "solar", "wind_total"]:
        if target_name in combined.columns:
            datasets_indirect[target_name] = create_darts_datasets(
                train,
                val,
                test,
                target_col=target_name,
            )

    return combined, train, val, test, datasets_direct, datasets_indirect


def prepare_datasets(config: dict):
    """Führt Datenbeschaffung und Preprocessing aus."""
    smard, weather, weather_forecast = step_1_download_data(config)
    return step_2_preprocessing(config, smard, weather, weather_forecast)


def train_model(
    model,
    trainer_fn,
    datasets: dict,
    model_name: str,
):
    """Trainiert ein Modell abhängig vom Modelltyp.

    Zwei Trainingsmuster:
      regression       — fit auf Train (past + future cov), kein Val
      tft / xgboost    — fit auf Train mit Val-Set (Early Stopping), past + future
    """
    logger.info(f"Starte Training für '{model_name}' ...")

    if model_name == "regression":
        model = trainer_fn(
            model,
            datasets["target_train_scaled"],
            past_cov_train=datasets["past_cov_train"],
            future_cov_train=datasets["future_cov_train"],
        )
    else:  # tft, xgboost
        model = trainer_fn(
            model,
            datasets["target_train_scaled"],
            datasets["target_val_scaled"],
            past_cov_train=datasets["past_cov_train"],
            past_cov_val=datasets["past_cov_val"],
            future_cov_train=datasets["future_cov_train"],
            future_cov_val=datasets["future_cov_val"],
        )

    logger.info(f"Training abgeschlossen für '{model_name}'.")
    return model


def build_rolling_forecast_dataframe(
    model,
    predictor_fn,
    quantile_extractor_fn,
    datasets: dict,
    model_name: str,
    horizon_hours: int,
    quantiles: list[float],
    stride: int,
) -> pd.DataFrame:
    """Erzeugt einen Forecast über den gesamten Testzeitraum aus Rolling-Fenstern."""
    target_trainval_scaled = datasets["target_train_scaled"].append(
        datasets["target_val_scaled"]
    )
    target_test_unscaled = datasets["target_test"]
    past_cov_full = datasets["past_cov_full"]
    future_cov_full = datasets["future_cov_full"]
    scaler = datasets["target_scaler"]
    test_start = datasets["target_test_scaled"].start_time()

    # Alle drei Modelle sind probabilistisch (TFT & XGBoost nativ,
    # Regression via likelihood="quantile") → identisches num_samples für
    # einen fairen Monte-Carlo-Vergleich der Quantilverteilungen.
    num_samples = 500

    try:
        backtest_preds = model.historical_forecasts(
            series=target_trainval_scaled.append(datasets["target_test_scaled"]),
            past_covariates=past_cov_full,
            future_covariates=future_cov_full,
            start=test_start,
            forecast_horizon=horizon_hours,
            stride=stride,
            retrain=False,
            last_points_only=False,
            num_samples=num_samples,
            verbose=True,
        )
    except Exception as exc:
        logger.warning("historical_forecasts fehlgeschlagen: %s", exc)
        logger.info("Fallback auf Einzelvorhersage vom Ende des Validierungssets.")
        backtest_preds = [
            predictor_fn(
                model,
                n=horizon_hours,
                series=datasets["target_val_scaled"],
                past_covariates=past_cov_full,
                future_covariates=future_cov_full,
                num_samples=num_samples,
            )
        ]

    rows = []
    for pred_scaled in backtest_preds:
        pred_median = pred_scaled.quantile(0.5) if pred_scaled.n_samples > 1 else pred_scaled
        pred_unscaled = scaler.inverse_transform(pred_median)
        pred_idx = pred_unscaled.time_index
        actual_slice = target_test_unscaled.slice(pred_idx[0], pred_idx[-1])
        if len(actual_slice) == 0:
            continue

        min_len = min(len(actual_slice), len(pred_unscaled))
        actual_vals = actual_slice[:min_len].values().flatten()
        pred_vals = pred_unscaled[:min_len].values().flatten()

        # Alle Quantilspalten (q10, q25, q50, q75, q90) — Kapitel 4 braucht
        # neben dem 80-%-Band auch das innere Paar für die Asymmetrie-Analyse.
        q_cols: dict[str, np.ndarray] = {}
        if pred_scaled.n_samples > 1:
            q_preds = quantile_extractor_fn(pred_scaled, quantiles)
            for qi, q in enumerate(quantiles):
                q_ts = TimeSeries.from_times_and_values(
                    pred_scaled.time_index[:min_len],
                    q_preds[:min_len, qi],
                )
                q_cols[f"q{int(round(q * 100))}"] = (
                    scaler.inverse_transform(q_ts).values().flatten()
                )

        for i, ts in enumerate(pred_unscaled.time_index[:min_len]):
            row = {
                "timestamp": ts,
                "actual": actual_vals[i],
                "prediction": pred_vals[i],
            }
            for col, vals in q_cols.items():
                row[col] = vals[i]
            rows.append(row)

    if not rows:
        raise RuntimeError("Keine Forecast-Daten erzeugt.")

    forecast_df = pd.DataFrame(rows)
    agg_map = {"actual": "first", "prediction": "mean"}
    for col in forecast_df.columns:
        if col.startswith("q") and col[1:].isdigit():
            agg_map[col] = "mean"

    forecast_df = (
        forecast_df.groupby("timestamp", as_index=False)
        .agg(agg_map)
        .sort_values("timestamp")
    )
    return forecast_df


def build_validation_forecast_dataframe(
    model,
    predictor_fn,
    quantile_extractor_fn,
    datasets: dict,
    model_name: str,
    horizon_hours: int,
    quantiles: list[float],
    stride: int,
) -> pd.DataFrame:
    """Erzeugt Rolling-Forecast über die VALIDATION-Periode.

    Spiegelbild von :func:`build_rolling_forecast_dataframe` für Test —
    nur dass hier die Validation-Periode prognostiziert wird (anhand des
    Trainings-Sets als Historie). Wird für die Conformal-Kalibrierung
    benötigt: wir brauchen Modell-Vorhersagen auf Daten, die das Modell
    weder gesehen hat (Train) noch beurteilen darf (Test).
    """
    target_train_scaled = datasets["target_train_scaled"]
    target_val_scaled = datasets["target_val_scaled"]
    target_val_unscaled = datasets["target_val"]
    past_cov_full = datasets["past_cov_full"]
    future_cov_full = datasets["future_cov_full"]
    scaler = datasets["target_scaler"]
    val_start = target_val_scaled.start_time()

    num_samples = 500

    try:
        backtest_preds = model.historical_forecasts(
            series=target_train_scaled.append(target_val_scaled),
            past_covariates=past_cov_full,
            future_covariates=future_cov_full,
            start=val_start,
            forecast_horizon=horizon_hours,
            stride=stride,
            retrain=False,
            last_points_only=False,
            num_samples=num_samples,
            verbose=True,
        )
    except Exception as exc:
        logger.warning(
            "Validation-historical_forecasts fehlgeschlagen: %s. "
            "Fallback auf Einzelvorhersage vom Ende des Trainingssets.", exc
        )
        backtest_preds = [
            predictor_fn(
                model,
                n=horizon_hours,
                series=target_train_scaled,
                past_covariates=past_cov_full,
                future_covariates=future_cov_full,
                num_samples=num_samples,
            )
        ]

    rows = []
    for pred_scaled in backtest_preds:
        pred_median = pred_scaled.quantile(0.5) if pred_scaled.n_samples > 1 else pred_scaled
        pred_unscaled = scaler.inverse_transform(pred_median)
        pred_idx = pred_unscaled.time_index
        actual_slice = target_val_unscaled.slice(pred_idx[0], pred_idx[-1])
        if len(actual_slice) == 0:
            continue

        min_len = min(len(actual_slice), len(pred_unscaled))
        actual_vals = actual_slice[:min_len].values().flatten()
        pred_vals = pred_unscaled[:min_len].values().flatten()

        # Alle Quantilspalten (q10, q25, q50, q75, q90) — Kapitel 4 braucht
        # neben dem 80-%-Band auch das innere Paar für die Asymmetrie-Analyse.
        q_cols: dict[str, np.ndarray] = {}
        if pred_scaled.n_samples > 1:
            q_preds = quantile_extractor_fn(pred_scaled, quantiles)
            for qi, q in enumerate(quantiles):
                q_ts = TimeSeries.from_times_and_values(
                    pred_scaled.time_index[:min_len],
                    q_preds[:min_len, qi],
                )
                q_cols[f"q{int(round(q * 100))}"] = (
                    scaler.inverse_transform(q_ts).values().flatten()
                )

        for i, ts in enumerate(pred_unscaled.time_index[:min_len]):
            row = {
                "timestamp": ts,
                "actual": actual_vals[i],
                "prediction": pred_vals[i],
            }
            for col, vals in q_cols.items():
                row[col] = vals[i]
            rows.append(row)

    if not rows:
        raise RuntimeError("Keine Validation-Forecast-Daten erzeugt.")

    forecast_df = pd.DataFrame(rows)
    agg_map = {"actual": "first", "prediction": "mean"}
    for col in forecast_df.columns:
        if col.startswith("q") and col[1:].isdigit():
            agg_map[col] = "mean"

    forecast_df = (
        forecast_df.groupby("timestamp", as_index=False)
        .agg(agg_map)
        .sort_values("timestamp")
    )
    return forecast_df


def evaluate_trained_model(
    model,
    predictor_fn,
    quantile_extractor_fn,
    datasets: dict,
    model_name: str,
    target_name: str,
    horizon_name: str,
    horizon_hours: int,
    quantiles: list[float],
    backtesting_stride: int,
) -> dict:
    """Evaluiert ein bereits trainiertes Modell per Rolling Backtest."""
    logger.info(
        "Starte Evaluation: Modell=%s, Target=%s, Horizont=%s (%sh)",
        model_name,
        target_name,
        horizon_name,
        horizon_hours,
    )

    target_trainval_scaled = datasets["target_train_scaled"].append(
        datasets["target_val_scaled"]
    )
    past_cov_full = datasets["past_cov_full"]
    future_cov_full = datasets["future_cov_full"]
    test_start = datasets["target_test_scaled"].start_time()

    # Alle drei Modelle sind probabilistisch (TFT & XGBoost nativ,
    # Regression via likelihood="quantile") → identisches num_samples für
    # einen fairen Monte-Carlo-Vergleich der Quantilverteilungen.
    num_samples = 500

    try:
        backtest_preds = model.historical_forecasts(
            series=target_trainval_scaled.append(datasets["target_test_scaled"]),
            past_covariates=past_cov_full,
            future_covariates=future_cov_full,
            start=test_start,
            forecast_horizon=horizon_hours,
            stride=backtesting_stride,
            retrain=False,
            last_points_only=False,
            num_samples=num_samples,
            verbose=True,
        )
        logger.info("Backtesting abgeschlossen: %s Forecast-Fenster.", len(backtest_preds))
    except Exception as exc:
        logger.warning("historical_forecasts fehlgeschlagen: %s", exc)
        logger.info("Fallback auf Einzelvorhersage vom Ende des Validierungssets.")
        prediction_scaled = predictor_fn(
            model,
            n=horizon_hours,
            series=datasets["target_val_scaled"],
            past_covariates=past_cov_full,
            future_covariates=future_cov_full,
            num_samples=num_samples,
        )
        backtest_preds = [prediction_scaled]

    all_actual_vals = []
    all_pred_vals = []
    all_quantile_vals = []
    scaler = datasets["target_scaler"]
    target_test_unscaled = datasets["target_test"]

    for pred_scaled in backtest_preds:
        pred_median = pred_scaled.quantile(0.5) if pred_scaled.n_samples > 1 else pred_scaled
        pred_unscaled = scaler.inverse_transform(pred_median)

        pred_idx = pred_unscaled.time_index
        actual_slice = target_test_unscaled.slice(pred_idx[0], pred_idx[-1])
        if len(actual_slice) == 0:
            continue

        min_len = min(len(actual_slice), len(pred_unscaled))
        all_actual_vals.append(actual_slice[:min_len].values().flatten())
        all_pred_vals.append(pred_unscaled[:min_len].values().flatten())

        if pred_scaled.n_samples > 1:
            q_preds = quantile_extractor_fn(pred_scaled, quantiles)
            for i in range(q_preds.shape[1]):
                q_ts = TimeSeries.from_times_and_values(
                    pred_scaled.time_index[:min_len],
                    q_preds[:min_len, i],
                )
                q_unscaled = scaler.inverse_transform(q_ts)
                q_preds[:min_len, i] = q_unscaled.values().flatten()
            all_quantile_vals.append(q_preds[:min_len])

    if not all_actual_vals:
        raise RuntimeError("Keine Vorhersagen konnten evaluiert werden.")

    actual_concat = np.concatenate(all_actual_vals)
    pred_concat = np.concatenate(all_pred_vals)

    first_pred_scaled = backtest_preds[0]
    first_median = first_pred_scaled.quantile(0.5) if first_pred_scaled.n_samples > 1 else first_pred_scaled
    first_pred_unscaled = scaler.inverse_transform(first_median)
    first_actual = target_test_unscaled.slice(
        first_pred_unscaled.time_index[0],
        first_pred_unscaled.time_index[-1],
    )
    min_len_first = min(len(first_actual), len(first_pred_unscaled))

    mae_val = np.mean(np.abs(actual_concat - pred_concat))
    rmse_val = np.sqrt(np.mean((actual_concat - pred_concat) ** 2))
    # Bewusst kein MAPE: Die Residuallast geht an PV-starken Mittagen gegen
    # null (teils negativ) — die Division durch Fast-Null-Istwerte bläht den
    # Mittelwert strukturell auf. Relative Einordnung übernimmt R².

    # R² (Bestimmtheitsmaß): 1 − SS_res / SS_tot.
    # Misst den Anteil der Varianz der Residuallast, den das Modell erklärt.
    # R²=1 → perfekt; R²=0 → nicht besser als der Mittelwert; R²<0 → schlechter.
    # Bezugsbasis ist der Mittelwert der TATSÄCHLICHEN Testwerte (klassische
    # Out-of-Sample-Definition).
    ss_res = np.sum((actual_concat - pred_concat) ** 2)
    ss_tot = np.sum((actual_concat - np.mean(actual_concat)) ** 2)
    r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    result = {
        "model": model_name,
        "target": target_name,
        "horizon": horizon_name,
        "MAE": mae_val,
        "RMSE": rmse_val,
        "R2": r2_val,
        "n_forecasts": len(backtest_preds),
        "n_datapoints": len(actual_concat),
    }

    quantile_concat = None
    if all_quantile_vals:
        quantile_concat = np.concatenate(all_quantile_vals, axis=0)
        result.update(pinball_loss(actual_concat, quantile_concat, quantiles))

        if 0.1 in quantiles and 0.9 in quantiles:
            idx_10 = quantiles.index(0.1)
            idx_90 = quantiles.index(0.9)
            covered = np.mean(
                (actual_concat >= quantile_concat[:, idx_10])
                & (actual_concat <= quantile_concat[:, idx_90])
            )
            width = np.mean(quantile_concat[:, idx_90] - quantile_concat[:, idx_10])
            result["coverage_80"] = covered
            result["interval_width_80"] = width

    logger.info(
        "[%s | %s | %s] MAE=%.1f MW, RMSE=%.1f MW, R²=%.3f",
        model_name,
        target_name,
        horizon_name,
        mae_val,
        rmse_val,
        r2_val,
    )

    return {
        "result": result,
        "prediction": first_pred_unscaled[:min_len_first],
        "actual": first_actual[:min_len_first],
        "quantile_preds": quantile_concat,
    }


def forecast_with_model(
    model,
    predictor_fn,
    quantile_extractor_fn,
    datasets: dict,
    model_name: str,
    target_name: str,
    horizon_name: str,
    horizon_hours: int,
    quantiles: list[float],
) -> dict:
    """Erzeugt einen Forecast-Plot über den gesamten Testzeitraum."""
    logger.info(
        "Erzeuge Prognose: Modell=%s, Target=%s, Horizont=%s (%sh)",
        model_name,
        target_name,
        horizon_name,
        horizon_hours,
    )

    stride = 24
    forecast_df = build_rolling_forecast_dataframe(
        model=model,
        predictor_fn=predictor_fn,
        quantile_extractor_fn=quantile_extractor_fn,
        datasets=datasets,
        model_name=model_name,
        horizon_hours=horizon_hours,
        quantiles=quantiles,
        stride=stride,
    )

    stem = build_artifact_stem(model_name, target_name, horizon_name)
    csv_path = FORECAST_DIR / f"{stem}_forecast.csv"
    plot_path = FORECAST_DIR / f"{stem}_forecast.png"

    forecast_df.to_csv(csv_path, index=False)

    actual_ts = TimeSeries.from_dataframe(
        forecast_df,
        time_col="timestamp",
        value_cols="actual",
        freq="h",
    )
    prediction_ts = TimeSeries.from_dataframe(
        forecast_df,
        time_col="timestamp",
        value_cols="prediction",
        freq="h",
    )

    lower = forecast_df["q10"].to_numpy() if "q10" in forecast_df.columns else None
    upper = forecast_df["q90"].to_numpy() if "q90" in forecast_df.columns else None

    plot_forecast_vs_actual(
        actual=actual_ts,
        predicted=prediction_ts,
        title=f"Forecast ueber Testzeitraum: {model_name} | {target_name} | {horizon_name}",
        quantile_lower=lower,
        quantile_upper=upper,
        save_path=str(plot_path),
    )

    logger.info("Forecast gespeichert: %s", csv_path)
    logger.info("Forecast-Plot gespeichert: %s", plot_path)
    return {
        "csv_path": csv_path,
        "plot_path": plot_path,
        "forecast_df": forecast_df,
    }


def command_download(config: dict, _args):
    """Lädt Rohdaten und speichert sie lokal."""
    step_1_download_data(config)


def command_preprocess(config: dict, _args):
    """Führt Datenbeschaffung und Preprocessing aus."""
    prepare_datasets(config)


def command_train(config: dict, args):
    """Trainiert ein einzelnes Modell und speichert es."""
    ensure_output_dirs()
    _, _, _, _, datasets_direct, datasets_indirect = prepare_datasets(config)
    model_defs = get_model_defs(config)
    model_def = model_defs[args.model]
    datasets = get_target_datasets(datasets_direct, datasets_indirect, args.target)
    horizon_hours = get_horizon_hours(config, args.horizon)

    model_cfg = {**model_def["config"], "output_chunk_length": horizon_hours}
    model = model_def["builder"](**model_cfg)
    model = train_model(model, model_def["trainer"], datasets, args.model)
    model_path, meta_path = save_model_artifact(
        model,
        args.model,
        args.target,
        args.horizon,
        model_def["model_version"],
    )

    logger.info("Modell gespeichert: %s", model_path)
    logger.info("Metadaten gespeichert: %s", meta_path)


def run_evaluate_command(config: dict, args) -> dict:
    """Fuehrt die Evaluation aus und liefert das Ergebnisobjekt zurueck."""
    ensure_output_dirs()
    _, _, _, _, datasets_direct, datasets_indirect = prepare_datasets(config)
    model_defs = get_model_defs(config)
    model_def = model_defs[args.model]
    datasets = get_target_datasets(datasets_direct, datasets_indirect, args.target)
    horizon_hours = get_horizon_hours(config, args.horizon)
    quantiles = config["models"]["tft"]["quantiles"]
    backtesting_stride = config["evaluation"]["backtesting"]["stride"]

    model = load_model_artifact(config, args.model, args.target, args.horizon)
    output = evaluate_trained_model(
        model=model,
        predictor_fn=model_def["predictor"],
        quantile_extractor_fn=model_def["quantile_fn"],
        datasets=datasets,
        model_name=args.model,
        target_name=args.target,
        horizon_name=args.horizon,
        horizon_hours=horizon_hours,
        quantiles=quantiles,
        backtesting_stride=backtesting_stride,
    )

    result_df = create_comparison_table([output["result"]])
    result_path = EVALUATION_DIR / f"{build_artifact_stem(args.model, args.target, args.horizon)}_evaluation.csv"
    result_df.to_csv(result_path)
    logger.info("Evaluation gespeichert: %s", result_path)
    return output


def print_evaluation_summary(result: dict) -> None:
    """Gibt die wichtigsten Fehlermetriken direkt im Terminal aus."""
    metrics = [
        f"MAE={result['MAE']:.2f}",
        f"RMSE={result['RMSE']:.2f}",
    ]
    if "R2" in result:
        metrics.append(f"R2={result['R2']:.3f}")

    if "pinball_mean" in result:
        metrics.append(f"Pinball={result['pinball_mean']:.4f}")
    if "coverage_80" in result:
        metrics.append(f"Coverage80={result['coverage_80']:.4f}")
    if "interval_width_80" in result:
        metrics.append(f"IntervalWidth80={result['interval_width_80']:.2f}")

    message = (
        f"[evaluate] Ergebnisse fuer {result['model']} | {result['target']} | {result['horizon']}: "
        + ", ".join(metrics)
    )
    print(color_text(message, ANSI_GREEN), flush=True)


def command_evaluate(config: dict, args):
    """Evaluiert ein gespeichertes Modell."""
    output = run_evaluate_command(config, args)
    print_evaluation_summary(output["result"])


def command_forecast(config: dict, args):
    """Erzeugt eine Prognose mit einem gespeicherten Modell."""
    ensure_output_dirs()
    _, _, _, _, datasets_direct, datasets_indirect = prepare_datasets(config)
    model_defs = get_model_defs(config)
    model_def = model_defs[args.model]
    datasets = get_target_datasets(datasets_direct, datasets_indirect, args.target)
    horizon_hours = get_horizon_hours(config, args.horizon)
    quantiles = config["models"]["tft"]["quantiles"]

    model = load_model_artifact(config, args.model, args.target, args.horizon)
    forecast_with_model(
        model=model,
        predictor_fn=model_def["predictor"],
        quantile_extractor_fn=model_def["quantile_fn"],
        datasets=datasets,
        model_name=args.model,
        target_name=args.target,
        horizon_name=args.horizon,
        horizon_hours=horizon_hours,
        quantiles=quantiles,
    )


# ════════════════════════════════════════════════════════════════════
#  Conformal Calibration (CQR — Romano et al. 2019)
# ════════════════════════════════════════════════════════════════════

def calibrate_one_model(
    config: dict,
    args,
    model_name: str,
    datasets: dict,
    horizon_hours: int,
    quantiles: list[float],
) -> dict:
    """Konformitätskalibrierung für ein einzelnes Modell.

    Workflow:
      1. Trainiertes Modell laden (kein Re-Training!)
      2. Rolling-Forecasts auf der VALIDATION-Periode generieren
      3. Bestehenden Test-Forecast aus output/forecasts/ laden
      4. CQR-Kalibrierung anwenden (q̂ aus Val, Anwendung auf Test)
      5. Kalibrierte Test-Forecasts speichern; Report zurückgeben.

    Returns
    -------
    dict mit keys 'model' und 'report' (CQR-Diagnose pro Quantil-Paar).
    """
    from src.calibration import calibrate_forecast_adaptive

    model_defs = get_model_defs(config)
    model_def = model_defs[model_name]
    stride = config["evaluation"]["backtesting"]["stride"]

    logger.info(
        "[calibrate/%s] Lade trainiertes Modell aus models/trained/ ...",
        model_name,
    )
    model = load_model_artifact(config, model_name, args.target, args.horizon)

    # Validation-Forecasts cachen — Backtesting ist teuer (~10–15 min pro
    # Modell). Beim wiederholten Aufruf von calibrate sparen wir uns das.
    stem = build_artifact_stem(model_name, args.target, args.horizon)
    val_path = Path("output/forecasts") / f"{stem}_validation_forecast.csv"
    if val_path.exists():
        logger.info(
            "[calibrate/%s] Verwende gecachte Validation-Forecasts: %s",
            model_name, val_path,
        )
        val_forecast_df = pd.read_csv(val_path, parse_dates=["timestamp"])
    else:
        logger.info(
            "[calibrate/%s] Erzeuge Validation-Forecasts (Backtesting %d h "
            "Horizont, Stride %d) ...",
            model_name, horizon_hours, stride,
        )
        val_forecast_df = build_validation_forecast_dataframe(
            model=model,
            predictor_fn=model_def["predictor"],
            quantile_extractor_fn=model_def["quantile_fn"],
            datasets=datasets,
            model_name=model_name,
            horizon_hours=horizon_hours,
            quantiles=quantiles,
            stride=stride,
        )
        val_path.parent.mkdir(parents=True, exist_ok=True)
        val_forecast_df.to_csv(val_path, index=False)
        logger.info("[calibrate/%s] Validation-Forecasts gespeichert: %s",
                    model_name, val_path)

    # Bestehenden Test-Forecast laden — wurde im run-Command schon erzeugt.
    test_path = Path("output/forecasts") / f"{stem}_forecast.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test-Forecast fehlt: {test_path}. Bitte zuerst "
            f"'python main.py run --model {model_name} ...' ausführen."
        )
    test_forecast_df = pd.read_csv(test_path, parse_dates=["timestamp"])

    logger.info(
        "[calibrate/%s] Wende Adaptive Conformal Inference an "
        "(Val n=%d, Test n=%d) ...",
        model_name, len(val_forecast_df), len(test_forecast_df),
    )
    test_calibrated, report = calibrate_forecast_adaptive(
        val_df=val_forecast_df,
        test_df=test_forecast_df,
        # Nur 80%-Band — q25/q75 sind in den CSVs nicht enthalten.
        quantile_pairs={"80": ("q10", "q90", 0.8)},
    )

    # Kalibrierten Test-Forecast speichern (Original bleibt erhalten).
    calibrated_path = Path("output/forecasts") / f"{stem}_forecast_calibrated.csv"
    test_calibrated.to_csv(calibrated_path, index=False)
    logger.info("[calibrate/%s] Kalibrierten Test-Forecast gespeichert: %s",
                model_name, calibrated_path)

    return {"model": model_name, "report": report, "calibrated_df": test_calibrated}


def command_calibrate(config: dict, args):
    """CLI-Command: Conformal-Prediction-Kalibrierung der gespeicherten Modelle.

    Kein Re-Training nötig — wir nutzen die in models/trained/ liegenden
    Modelle, generieren Validation-Forecasts (Inference), berechnen daraus
    Konformitätsscores, und kalibrieren die bestehenden Test-Forecasts.
    """
    ensure_output_dirs()
    _, _, _, _, datasets_direct, datasets_indirect = prepare_datasets(config)
    datasets = get_target_datasets(datasets_direct, datasets_indirect, args.target)
    horizon_hours = get_horizon_hours(config, args.horizon)
    quantiles = config["models"]["tft"]["quantiles"]

    model_names = ["regression", "tft", "xgboost"] if args.model == "all" else [args.model]

    reports: list[dict] = []
    for name in model_names:
        result = calibrate_one_model(
            config=config,
            args=args,
            model_name=name,
            datasets=datasets,
            horizon_hours=horizon_hours,
            quantiles=quantiles,
        )
        reports.append(result)

    # Übersichtstabelle als CSV
    summary_rows = []
    for r in reports:
        for level, info in r["report"].items():
            summary_rows.append({
                "model": r["model"],
                "coverage_level": float(level) / 100,
                # ACI-spezifische Diagnose
                "gamma":        info["gamma"],
                "q_hat_init":   info["q_hat_init"],
                "q_hat_mean":   info["q_hat_mean"],
                "q_hat_median": info["q_hat_median"],
                "q_hat_min":    info["q_hat_min"],
                "q_hat_max":    info["q_hat_max"],
                # Val-Diagnose (rohe Modell-Performance auf Val)
                "val_coverage": info["val_coverage_uncal"],
                "val_width":    info["val_width_uncal"],
                # Test-Vergleich
                "test_coverage_uncalibrated":   info["test_coverage_uncal"],
                "test_width_uncalibrated":      info["test_width_uncal"],
                "test_coverage_calibrated":     info["test_coverage_calibrated"],
                "test_width_calibrated_mean":   info["test_width_calibrated_mean"],
                "test_width_calibrated_median": info["test_width_calibrated_median"],
                "n_val":  info["n_val"],
                "n_test": info["n_test"],
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = (
        Path("output/results")
        / f"all_models_{args.target}_{args.horizon}_conformal_calibration.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    logger.info("Conformal-Übersicht gespeichert: %s", summary_path)

    # Konsolen-Ausgabe
    print()
    print(color_text("=" * 96, ANSI_CYAN))
    print(color_text(
        "ADAPTIVE CONFORMAL INFERENCE — Kalibrierungs-Ergebnisse "
        "(Gibbs & Candès 2021)", ANSI_CYAN,
    ))
    print(color_text("=" * 96, ANSI_CYAN))
    print(
        f"{'Modell':<12} {'Cov vor':>9}  {'Cov nach':>9}  {'Δ Cov':>8}  "
        f"{'Breite vor':>11}  {'Ø Breite nach':>14}  {'q̂ Range':>16}"
    )
    print("-" * 96)
    for r in reports:
        for level, info in r["report"].items():
            cov_before = info["test_coverage_uncal"]
            cov_after = info["test_coverage_calibrated"]
            width_before = info["test_width_uncal"]
            width_after = info["test_width_calibrated_mean"]
            q_hat_range = f"[{info['q_hat_min']:.0f}…{info['q_hat_max']:.0f}]"
            print(
                f"{r['model']:<12} {cov_before:>9.3f}  {cov_after:>9.3f}  "
                f"{cov_after - cov_before:>+8.3f}  "
                f"{width_before:>11.0f}  {width_after:>14.0f}  "
                f"{q_hat_range:>16}"
            )
    print(color_text("=" * 96, ANSI_CYAN))
    print(f"Detail-CSV: {summary_path}")
    print(
        "Hinweis: ACI passt q̂ pro Stunde adaptiv an — die Spannweite des "
        "q̂-Verlaufs (rechte Spalte)\n"
        "        ist diagnostisch wertvoll und liegt für die Plot-Generierung "
        "in den _forecast_calibrated.csv vor."
    )

    # ── Notebook-Generierung — finaler Pipeline-Schritt ─────────────
    # Hier (statt am Ende von `run`) erzeugt, damit das Notebook bereits die
    # ACI-Sektion mit q̂-Plot, kalibrierten Bändern und korrigierter
    # Probabilistik-Tabelle enthält.
    notebook_models = (
        ["regression", "tft", "xgboost"] if args.model == "all" else [args.model]
    )
    try:
        nb_path = generate_run_notebook(
            config=config,
            target=args.target,
            horizon=args.horizon,
            models_run=notebook_models,
            results_dir=EVALUATION_DIR,
            notebooks_dir=Path("output/notebooks"),
        )
        print(color_text(
            f"\n[calibrate] Notebook (inkl. ACI-Sektion) gespeichert: {nb_path}",
            ANSI_GREEN,
        ), flush=True)
    except Exception as exc:
        logger.warning("Notebook-Generierung fehlgeschlagen: %s", exc)


# ════════════════════════════════════════════════════════════════════
#  Finales Training (Train+Val) + Testbewertung + ACI — Kapitel-4-Daten
# ════════════════════════════════════════════════════════════════════

FINAL_DIR = Path("output/final")


def load_best_hpo_params(model_name: str, target_name: str, horizon_name: str) -> dict:
    """Liest die beste Konfiguration aus der Optuna-Studien-Datenbank."""
    import optuna

    db_path = (
        Path("output/optimization/studies")
        / f"{build_artifact_stem(model_name, target_name, horizon_name)}.db"
    )
    if not db_path.exists():
        raise FileNotFoundError(
            f"HPO-Studie fehlt: {db_path}. Bitte zuerst 'optimize' ausführen."
        )
    study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_path}")
    best = study.best_trial
    logger.info(
        "[final/%s] Beste HPO-Konfiguration (Trial %d, Pinball=%.2f): %s",
        model_name, best.number, best.value, best.params,
    )
    return dict(best.params)


def _extract_tft_best_epoch(model) -> dict:
    """Bestimmt die Epoche des besten Validierungsverlusts nach Early Stopping.

    Early Stopping beendet das Training `patience` Epochen NACH dem Optimum.
    Primärquelle ist der Callback: Bei ausgelöstem Stop liegt das Optimum bei
    stopped_epoch − wait_count (0-indexiert), als Epochenzahl für das finale
    Training also +1. Der Fallback epochs_trained − wait_count greift nur,
    wenn das Training das Epochenlimit erreichte — model.epochs_trained ist
    nach dem Reload des besten Checkpoints nicht verlässlich (liefert 0),
    deshalb zählt der Callback zuerst.
    """
    epochs_trained = int(getattr(model, "epochs_trained", 0) or 0)
    wait_count = 0
    stopped_epoch = 0
    for accessor in (
        lambda m: m.model.trainer.early_stopping_callback,
        lambda m: m.trainer.early_stopping_callback,
    ):
        try:
            cb = accessor(model)
            if cb is not None:
                wait_count = int(getattr(cb, "wait_count", 0) or 0)
                stopped_epoch = int(getattr(cb, "stopped_epoch", 0) or 0)
                break
        except Exception:
            continue
    if stopped_epoch > 0:
        best_epoch = max(1, stopped_epoch - wait_count + 1)
    else:
        best_epoch = max(1, epochs_trained - wait_count)
    return {
        "epochs_trained": epochs_trained,
        "wait_count": wait_count,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
    }


def _band_pinball(df: pd.DataFrame, lower_col: str, upper_col: str) -> float:
    """Mittlerer Pinball-Loss des 80-%-Band-Paars (q10/q90)."""
    actual = df["actual"].to_numpy(dtype=float)
    losses = []
    for q, col in ((0.1, lower_col), (0.9, upper_col)):
        err = actual - df[col].to_numpy(dtype=float)
        losses.append(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))
    return float(np.mean(losses))


def compute_final_metrics(df: pd.DataFrame, quantiles: list[float]) -> dict:
    """Punkt- und Verteilungsmetriken aus einem Forecast-DataFrame."""
    actual = df["actual"].to_numpy(dtype=float)
    pred = df["prediction"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    out = {"MAE": mae, "RMSE": rmse, "R2": r2, "n": int(len(df))}

    q_losses = []
    for q in quantiles:
        col = f"q{int(round(q * 100))}"
        if col not in df.columns:
            continue
        err = actual - df[col].to_numpy(dtype=float)
        q_losses.append(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))
    if q_losses:
        out["pinball_mean"] = float(np.mean(q_losses))
    return out


def command_final_train(config: dict, args):
    """Finales Training mit den besten HPO-Konfigurationen (Design: 3.2.7).

    Ablauf je Modell:
      1. Val-Lauf: Modell allein auf Train trainieren, rollierend auf Val
         prognostizieren → Konformitätsscores für c0/gamma; beim TFT
         zusätzlich die beste Epochenzahl (gecacht in output/final/).
      2. Finales Training auf Train+Val. TFT: Epochenzahl fixiert, kein
         Early Stopping. Skalierung unverändert aus der Trainingsmenge.
      3. Rollierende Testbewertung (Stride aus config), unkalibriert.
      4. ACI auf dem 80-%-Band (c0/gamma aus Schritt 1).
      5. Kennzahlen roh vs. kalibriert → CSV + Meta-JSON.
    """
    from src.calibration import calibrate_forecast_adaptive
    from src.calibration.conformal import evaluate_coverage

    ensure_output_dirs()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    _, _, _, _, datasets_direct, datasets_indirect = prepare_datasets(config)
    datasets = get_target_datasets(datasets_direct, datasets_indirect, args.target)
    horizon_hours = get_horizon_hours(config, args.horizon)
    quantiles = config["models"]["tft"]["quantiles"]
    stride = config["evaluation"]["backtesting"]["stride"]
    model_defs = get_model_defs(config)

    model_names = (
        ["regression", "xgboost", "tft"] if args.model == "all" else [args.model]
    )

    result_rows = []
    for name in model_names:
        model_def = model_defs[name]
        stem = build_artifact_stem(name, args.target, args.horizon)
        meta_path = FINAL_DIR / f"{stem}_final_meta.json"
        val_path = FINAL_DIR / f"{stem}_final_validation_forecast.csv"

        best_params = load_best_hpo_params(name, args.target, args.horizon)
        cfg = dict(model_def["config"])
        cfg.update(best_params)

        # ── Schritt 1: Val-Lauf (Train-only) — gecacht ───────────────
        meta: dict = {}
        cached_ok = False
        if val_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            cached_ok = meta.get("best_params") == best_params and (
                name != "tft" or meta.get("tft_best_epoch")
            )
        if cached_ok:
            logger.info("[final/%s] Verwende gecachten Val-Lauf: %s", name, val_path)
            val_df = pd.read_csv(val_path, parse_dates=["timestamp"])
        else:
            logger.info(
                "[final/%s] Val-Lauf: Training auf Train, rollierende "
                "Bewertung auf Val (Stride %d h) ...", name, stride,
            )
            model_val = model_def["builder"](**cfg)
            model_val = train_model(model_val, model_def["trainer"], datasets, name)
            meta = {"model": name, "best_params": best_params}
            if name == "tft":
                epoch_info = _extract_tft_best_epoch(model_val)
                meta["tft_epoch_info"] = epoch_info
                meta["tft_best_epoch"] = epoch_info["best_epoch"]
                logger.info("[final/tft] Beste Epoche aus Val-Lauf: %s", epoch_info)
            val_df = build_validation_forecast_dataframe(
                model=model_val,
                predictor_fn=model_def["predictor"],
                quantile_extractor_fn=model_def["quantile_fn"],
                datasets=datasets,
                model_name=name,
                horizon_hours=horizon_hours,
                quantiles=quantiles,
                stride=stride,
            )
            val_df.to_csv(val_path, index=False)
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
            logger.info("[final/%s] Val-Forecast gespeichert: %s", name, val_path)

        # ── Schritt 2: Finales Training auf Train+Val ────────────────
        final_cfg = dict(cfg)
        if name == "tft":
            final_cfg["n_epochs"] = int(meta["tft_best_epoch"])
            final_cfg["early_stopping_enabled"] = False
            final_cfg["model_name"] = "tft_final"
            logger.info(
                "[final/tft] Fixierte Epochenzahl für Train+Val: %d",
                final_cfg["n_epochs"],
            )
        logger.info("[final/%s] Finales Training auf Train+Val ...", name)
        model_final = model_def["builder"](**final_cfg)
        target_trainval = datasets["target_train_scaled"].append(
            datasets["target_val_scaled"]
        )
        model_final.fit(
            series=target_trainval,
            past_covariates=datasets["past_cov_full"],
            future_covariates=datasets["future_cov_full"],
        )
        final_model_path = ARTIFACTS_DIR / f"{stem}_final.pkl"
        try:
            model_final.save(str(final_model_path))
            logger.info("[final/%s] Modell gespeichert: %s", name, final_model_path)
        except Exception as exc:
            logger.warning("[final/%s] Modell-Speichern fehlgeschlagen: %s", name, exc)

        # ── Schritt 3: Testbewertung (roh) ───────────────────────────
        logger.info(
            "[final/%s] Rollierende Testbewertung (Stride %d h) ...", name, stride,
        )
        test_df = build_rolling_forecast_dataframe(
            model=model_final,
            predictor_fn=model_def["predictor"],
            quantile_extractor_fn=model_def["quantile_fn"],
            datasets=datasets,
            model_name=name,
            horizon_hours=horizon_hours,
            quantiles=quantiles,
            stride=stride,
        )
        test_path = FINAL_DIR / f"{stem}_final_forecast.csv"
        test_df.to_csv(test_path, index=False)
        logger.info("[final/%s] Test-Forecast gespeichert: %s", name, test_path)

        # ── Schritt 4: ACI auf dem 80-%-Band ─────────────────────────
        test_cal, report = calibrate_forecast_adaptive(
            val_df=val_df,
            test_df=test_df,
            quantile_pairs={"80": ("q10", "q90", 0.8)},
        )
        cal_path = FINAL_DIR / f"{stem}_final_forecast_calibrated.csv"
        test_cal.to_csv(cal_path, index=False)
        logger.info("[final/%s] Kalibrierter Forecast gespeichert: %s", name, cal_path)

        # ── Schritt 5: Kennzahlen roh vs. kalibriert ─────────────────
        info = report["80"]
        metrics = compute_final_metrics(test_df, quantiles)
        cov_raw = evaluate_coverage(
            test_df["actual"].to_numpy(float),
            test_df["q10"].to_numpy(float),
            test_df["q90"].to_numpy(float),
        )
        row = {
            "model": name,
            **metrics,
            "coverage_80_raw": cov_raw["coverage"],
            "width_80_raw": cov_raw["mean_width"],
            "coverage_80_aci": info["test_coverage_calibrated"],
            "width_80_aci": info["test_width_calibrated_mean"],
            "pinball_band_raw": _band_pinball(test_df, "q10", "q90"),
            "pinball_band_aci": _band_pinball(test_cal, "q10_cal80", "q90_cal80"),
            "gamma": info["gamma"],
            "c0": info["q_hat_init"],
            "c_mean": info["q_hat_mean"],
            "c_min": info["q_hat_min"],
            "c_max": info["q_hat_max"],
            "val_coverage_raw": info["val_coverage_uncal"],
            "n_val": info["n_val"],
            "n_test": info["n_test"],
        }
        result_rows.append(row)

        meta["aci"] = {
            k: info[k]
            for k in (
                "gamma", "q_hat_init", "q_hat_mean", "q_hat_min", "q_hat_max",
                "test_coverage_uncal", "test_coverage_calibrated",
                "test_width_uncal", "test_width_calibrated_mean",
            )
        }
        meta["test_metrics_raw"] = metrics
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    results_df = pd.DataFrame(result_rows)
    out_csv = FINAL_DIR / f"all_models_{args.target}_{args.horizon}_final_results.csv"
    # Einzelmodell-Läufe ergänzen die Tabelle, statt sie zu überschreiben.
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        old = old[~old["model"].isin(results_df["model"])]
        results_df = pd.concat([old, results_df], ignore_index=True)
    results_df.to_csv(out_csv, index=False)
    logger.info("Finale Ergebnistabelle gespeichert: %s", out_csv)

    print()
    print(color_text("=" * 100, ANSI_CYAN))
    print(color_text(
        "FINALE TESTBEWERTUNG (Train+Val-Modelle, taegliches Raster) — "
        "roh vs. ACI-kalibriert", ANSI_CYAN,
    ))
    print(color_text("=" * 100, ANSI_CYAN))
    print(
        f"{'Modell':<12} {'MAE':>8} {'RMSE':>8} {'R2':>6} {'Pinball':>9}  "
        f"{'Cov roh':>8} {'Cov ACI':>8}  {'Breite roh':>11} {'Breite ACI':>11}"
    )
    print("-" * 100)
    for row in result_rows:
        print(
            f"{row['model']:<12} {row['MAE']:>8.0f} {row['RMSE']:>8.0f} "
            f"{row['R2']:>6.3f} {row.get('pinball_mean', float('nan')):>9.1f}  "
            f"{row['coverage_80_raw']:>8.3f} {row['coverage_80_aci']:>8.3f}  "
            f"{row['width_80_raw']:>11.0f} {row['width_80_aci']:>11.0f}"
        )
    print(color_text("=" * 100, ANSI_CYAN))
    print(f"Artefakte: {FINAL_DIR}/ (Forecasts, kalibrierte Baender, c_t-Verlauf, Meta-JSON)")


# ════════════════════════════════════════════════════════════════════
#  Hyperparameter-Optimierung (Optuna)
# ════════════════════════════════════════════════════════════════════

def optimize_one_model(
    config: dict,
    args,
    model_name: str,
    datasets: dict,
    horizon_hours: int,
    quantiles: list[float],
) -> dict:
    """Führt die Optuna-Optimierung für ein einzelnes Modell durch.

    Bewertet jeden Trial per Pinball Loss auf der VALIDATION-Periode —
    das Test-Set bleibt unangetastet (Felix' Workflow).
    """
    from src.optimization import build_objective, run_study
    from src.optimization.search_spaces import DEFAULT_N_TRIALS

    model_defs = get_model_defs(config)
    model_def = model_defs[model_name]

    n_trials = args.n_trials if args.n_trials is not None else DEFAULT_N_TRIALS[model_name]
    eval_stride = args.eval_stride
    timeout = args.timeout  # Sekunden, None = kein Limit

    logger.info(
        "[optimize/%s] Starte Optuna — %d Trials, eval_stride=%d, "
        "timeout=%s, Objektiv=Pinball(val)",
        model_name, n_trials, eval_stride,
        f"{timeout}s" if timeout else "kein",
    )

    # Training-Callback kapselt die modellspezifische Trainingslogik.
    # Reingereicht, damit das Objective-Modul unabhängig von main.py bleibt.
    def train_callback(model):
        return train_model(model, model_def["trainer"], datasets, model_name)

    objective = build_objective(
        model_name=model_name,
        model_def=model_def,
        datasets=datasets,
        horizon_hours=horizon_hours,
        quantiles=quantiles,
        train_callback=train_callback,
        eval_stride=eval_stride,
    )

    result = run_study(
        model_name=model_name,
        target=args.target,
        horizon=args.horizon,
        objective=objective,
        n_trials=n_trials,
        seed=config["models"].get("seed", 42),
        timeout=timeout,
    )
    return result


def command_optimize(config: dict, args):
    """CLI-Command: Hyperparameter-Optimierung mit Optuna.

    Phase 1 von Felix' Workflow: HPO auf Train→Val. Die besten Parameter
    werden als JSON gespeichert; die spätere `final-train`-Stufe trainiert
    damit auf Train+Val und evaluiert einmalig auf Test.
    """
    ensure_output_dirs()
    _, _, _, _, datasets_direct, datasets_indirect = prepare_datasets(config)
    datasets = get_target_datasets(datasets_direct, datasets_indirect, args.target)
    horizon_hours = get_horizon_hours(config, args.horizon)
    quantiles = config["models"]["tft"]["quantiles"]

    model_names = ["regression", "tft", "xgboost"] if args.model == "all" else [args.model]

    results = []
    for name in model_names:
        result = optimize_one_model(
            config=config,
            args=args,
            model_name=name,
            datasets=datasets,
            horizon_hours=horizon_hours,
            quantiles=quantiles,
        )
        results.append((name, result))

    # Konsolen-Zusammenfassung
    print()
    print(color_text("=" * 84, ANSI_CYAN))
    print(color_text("OPTUNA — Beste Hyperparameter (Objektiv: Pinball Loss auf Validation)",
                     ANSI_CYAN))
    print(color_text("=" * 84, ANSI_CYAN))
    for name, res in results:
        print(f"\n{color_text(name.upper(), ANSI_GREEN)} "
              f"| {res['n_trials_total']} Trials | "
              f"bester Pinball(val) = {res['best_value']:.4f}")
        for k, v in res["best_params"].items():
            val_str = f"{v:.5f}" if isinstance(v, float) else str(v)
            print(f"    {k:<22} = {val_str}")
        print(f"    {color_text('→ ' + str(res['best_params_path']), ANSI_CYAN)}")
    print(color_text("=" * 84, ANSI_CYAN))
    print("Nächster Schritt (sobald HPO für alle Modelle steht):")
    print("    python main.py final-train --model all "
          f"--target {args.target} --horizon {args.horizon}")


def generate_week_detail_plot(
    model,
    datasets: dict,
    model_name: str,
    target_name: str,
    horizon_name: str,
    horizon_hours: int,
):
    """Erzeugt einen detaillierten Wochenplot mit Konfidenzintervallen."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timedelta

    scaler = datasets["target_scaler"]
    target_full = datasets["target_full"]

    target_all_scaled = scaler.transform(target_full)
    past_cov_full = datasets["past_cov_full"]
    future_cov_full = datasets["future_cov_full"]

    # Woche im Testset auswählen (2 Wochen nach Teststart)
    test_start = datasets["target_test"].start_time()
    week_start = test_start + timedelta(hours=336)  # 2 Wochen rein

    logger.info("Erzeuge Wochenplot ab %s ...", week_start)

    # Alle Modelle probabilistisch → einheitliches num_samples.
    num_samples = 500
    forecasts = model.historical_forecasts(
        series=target_all_scaled,
        past_covariates=past_cov_full,
        future_covariates=future_cov_full,
        start=week_start,
        start_format="value",
        forecast_horizon=horizon_hours,
        stride=horizon_hours,
        retrain=False,
        num_samples=num_samples,
        last_points_only=False,
        verbose=True,
    )

    # Nur 7 Tage
    n_days = min(7, len(forecasts))
    forecasts = forecasts[:n_days]
    forecasts_inv = [scaler.inverse_transform(f) for f in forecasts]

    first_time = forecasts_inv[0].time_index[0]
    last_time = forecasts_inv[-1].time_index[-1]
    actual_week = target_full.slice(
        first_time - timedelta(hours=12),
        last_time + timedelta(hours=12),
    )

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={"height_ratios": [3, 1]})
    day_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    ax1 = axes[0]
    actual_df = actual_week.to_dataframe()
    ax1.plot(actual_df.index, actual_df.values / 1000, color="black", linewidth=2.5, label="Ist-Werte", zorder=5)

    all_errors = []
    coverage_in = 0
    coverage_total = 0

    for i, fc in enumerate(forecasts_inv):
        times = fc.time_index
        color = day_colors[i % len(day_colors)]

        if fc.n_samples > 1:
            q10 = fc.quantile(0.1).values().flatten() / 1000
            q25 = fc.quantile(0.25).values().flatten() / 1000
            q50 = fc.quantile(0.5).values().flatten() / 1000
            q75 = fc.quantile(0.75).values().flatten() / 1000
            q90 = fc.quantile(0.9).values().flatten() / 1000

            ax1.fill_between(times, q10, q90, alpha=0.15, color=color,
                             label="80%-Konfidenzintervall" if i == 0 else None)
            ax1.fill_between(times, q25, q75, alpha=0.3, color=color,
                             label="50%-Konfidenzintervall" if i == 0 else None)
            pred_vals = q50
        else:
            pred_vals = fc.values().flatten() / 1000

        day_label = times[0].strftime("%a %d.%m")
        ax1.plot(times, pred_vals, color=color, linewidth=2, linestyle="--", label=f"Median {day_label}")

        # Fehler berechnen
        actual_slice = target_full.slice(times[0], times[-1])
        actual_vals = actual_slice.values().flatten()[:len(pred_vals)]
        errors_mw = actual_vals - pred_vals * 1000
        all_errors.extend(errors_mw)

        # Coverage
        if fc.n_samples > 1:
            q10_mw = fc.quantile(0.1).values().flatten()
            q90_mw = fc.quantile(0.9).values().flatten()
            coverage_in += np.sum((actual_vals >= q10_mw[:len(actual_vals)]) & (actual_vals <= q90_mw[:len(actual_vals)]))
            coverage_total += len(actual_vals)

    ax1.set_title(
        f"{model_name.upper()} Day-Ahead Prognose \u2014 Detailansicht einer Woche im Testset",
        fontsize=16, fontweight="bold",
    )
    ax1.set_ylabel("Residuallast [GW]", fontsize=14)
    ax1.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d.%m"))
    ax1.xaxis.set_major_locator(mdates.DayLocator())

    # Fehlerbalken
    ax2 = axes[1]
    for i, fc in enumerate(forecasts_inv):
        fc_times = fc.time_index
        q50 = fc.quantile(0.5).values().flatten() if fc.n_samples > 1 else fc.values().flatten()
        actual_slice = target_full.slice(fc_times[0], fc_times[-1])
        actual_vals = actual_slice.values().flatten()[:len(q50)]
        errors = (actual_vals - q50) / 1000
        color = day_colors[i % len(day_colors)]
        ax2.bar(fc_times[:len(errors)], errors, width=0.035, alpha=0.7, color=color)

    ax2.axhline(y=0, color="black", linewidth=1)
    ax2.set_title("Prognosefehler (Ist \u2212 Median) pro Stunde", fontsize=13)
    ax2.set_ylabel("Fehler [GW]", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d.%m"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())

    plt.tight_layout()

    stem = build_artifact_stem(model_name, target_name, horizon_name)
    plot_dir = Path("output/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    png_path = plot_dir / f"{stem}_week_detail.png"
    pdf_path = plot_dir / f"{stem}_week_detail.pdf"
    plt.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.savefig(str(pdf_path), bbox_inches="tight")
    plt.close()

    all_errors_arr = np.array(all_errors)
    logger.info("Wochenplot gespeichert: %s", png_path)
    logger.info(
        "Wochenstatistik: MAE=%.0f MW, RMSE=%.0f MW, Bias=%+.0f MW",
        np.mean(np.abs(all_errors_arr)),
        np.sqrt(np.mean(all_errors_arr**2)),
        np.mean(all_errors_arr),
    )
    if coverage_total > 0:
        logger.info("80%%-Coverage (Woche): %.1f%%", coverage_in / coverage_total * 100)

    return str(png_path)


def _run_single_model(config: dict, args, model_name: str, datasets_direct, datasets_indirect):
    """Trainiert, evaluiert und forecastet ein einzelnes Modell."""
    model_defs = get_model_defs(config)
    model_def = model_defs[model_name]
    datasets = get_target_datasets(datasets_direct, datasets_indirect, args.target)
    horizon_hours = get_horizon_hours(config, args.horizon)
    quantiles = config["models"]["tft"]["quantiles"]
    backtesting_stride = config["evaluation"]["backtesting"]["stride"]

    logger.info("=" * 60)
    logger.info("MODELL: %s | Target: %s | Horizont: %s", model_name.upper(), args.target, args.horizon)
    logger.info("=" * 60)

    # 1. Training
    model_cfg = {**model_def["config"], "output_chunk_length": horizon_hours}
    model = model_def["builder"](**model_cfg)
    model = train_model(model, model_def["trainer"], datasets, model_name)
    model_path, meta_path = save_model_artifact(
        model, model_name, args.target, args.horizon, model_def["model_version"],
    )
    logger.info("Modell gespeichert: %s", model_path)

    # 2. Evaluation
    output = evaluate_trained_model(
        model=model,
        predictor_fn=model_def["predictor"],
        quantile_extractor_fn=model_def["quantile_fn"],
        datasets=datasets,
        model_name=model_name,
        target_name=args.target,
        horizon_name=args.horizon,
        horizon_hours=horizon_hours,
        quantiles=quantiles,
        backtesting_stride=backtesting_stride,
    )
    result_df = create_comparison_table([output["result"]])
    result_path = EVALUATION_DIR / f"{build_artifact_stem(model_name, args.target, args.horizon)}_evaluation.csv"
    result_df.to_csv(result_path)
    logger.info("Evaluation gespeichert: %s", result_path)
    print_evaluation_summary(output["result"])

    # 3. Forecast
    forecast_with_model(
        model=model,
        predictor_fn=model_def["predictor"],
        quantile_extractor_fn=model_def["quantile_fn"],
        datasets=datasets,
        model_name=model_name,
        target_name=args.target,
        horizon_name=args.horizon,
        horizon_hours=horizon_hours,
        quantiles=quantiles,
    )

    # 4. Wochenplot
    try:
        generate_week_detail_plot(
            model=model,
            datasets=datasets,
            model_name=model_name,
            target_name=args.target,
            horizon_name=args.horizon,
            horizon_hours=horizon_hours,
        )
    except Exception as e:
        logger.warning("Wochenplot fehlgeschlagen: %s", e)

    return output["result"]


def command_run(config: dict, args):
    """Trainiert, evaluiert, erstellt Forecast und Wochenplot.
    Mit --model all: regression → tft → xgboost nacheinander."""
    ensure_output_dirs()

    # Preprocessing nur EINMAL ausführen (auch bei --model all)
    _, _, _, _, datasets_direct, datasets_indirect = prepare_datasets(config)

    # ── all: alle drei Modelle nacheinander ──────────────────────────────────
    if args.model == "all":
        all_models = ["regression", "tft", "xgboost"]
        all_results = []
        for i, model_name in enumerate(all_models, 1):
            print_timed_status(
                "run",
                f"Modell {i}/{len(all_models)}: {model_name.upper()} startet ...",
                ANSI_CYAN,
            )
            result = _run_single_model(
                config, args, model_name, datasets_direct, datasets_indirect
            )
            all_results.append(result)

        # Kombinierte Vergleichstabelle aller drei Modelle
        comparison = create_comparison_table(all_results)
        comparison_path = EVALUATION_DIR / f"all_models_{args.target}_{args.horizon}_comparison.csv"
        comparison.to_csv(comparison_path)
        logger.info("Vergleichstabelle (alle Modelle) gespeichert: %s", comparison_path)
        print_timed_status("run", "Alle Modelle abgeschlossen!", ANSI_GREEN)
        print(comparison.to_string())

        # Hinweis auf den nächsten Pipeline-Schritt — das Notebook wird
        # bewusst NICHT hier erzeugt, damit es nur einmal (mit ACI-
        # Kalibrierung) am Ende von `calibrate` rauskommt.
        print(color_text(
            "\n[run] Nächster Schritt: probabilistische Kalibrierung + "
            "Notebook-Generierung mit\n"
            f"      python main.py calibrate --model {args.model} "
            f"--target {args.target} --horizon {args.horizon}",
            ANSI_CYAN,
        ), flush=True)
        return

    # ── einzelnes Modell ─────────────────────────────────────────────────────
    _run_single_model(config, args, args.model, datasets_direct, datasets_indirect)


def command_pipeline(config: dict, _args):
    """Führt die gesamte Pipeline für alle Modelltypen und Horizonte aus."""
    ensure_output_dirs()
    _, _, _, _, datasets_direct, datasets_indirect = prepare_datasets(config)
    model_defs = get_model_defs(config)
    quantiles = config["models"]["tft"]["quantiles"]
    backtesting_stride = config["evaluation"]["backtesting"]["stride"]
    all_results = []

    for model_name, model_def in model_defs.items():
        for horizon_name in config["models"]["forecast_horizons"]:
            horizon_hours = get_horizon_hours(config, horizon_name)
            for target_name in ["residual_load", "total_load", "solar", "wind_total"]:
                if target_name != "residual_load" and target_name not in datasets_indirect:
                    continue

                logger.info("=" * 60)
                logger.info(
                    "PIPELINE: Modell=%s | Target=%s | Horizont=%s",
                    model_name,
                    target_name,
                    horizon_name,
                )
                logger.info("=" * 60)

                datasets = get_target_datasets(datasets_direct, datasets_indirect, target_name)
                model_cfg = {**model_def["config"], "output_chunk_length": horizon_hours}
                model = model_def["builder"](**model_cfg)
                model = train_model(model, model_def["trainer"], datasets, model_name)
                save_model_artifact(
                    model,
                    model_name,
                    target_name,
                    horizon_name,
                    model_def["model_version"],
                )

                output = evaluate_trained_model(
                    model=model,
                    predictor_fn=model_def["predictor"],
                    quantile_extractor_fn=model_def["quantile_fn"],
                    datasets=datasets,
                    model_name=model_name,
                    target_name=target_name,
                    horizon_name=horizon_name,
                    horizon_hours=horizon_hours,
                    quantiles=quantiles,
                    backtesting_stride=backtesting_stride,
                )
                all_results.append(output["result"])

    comparison = create_comparison_table(all_results)
    comparison_path = EVALUATION_DIR / "comparison_table.csv"
    comparison.to_csv(comparison_path)
    logger.info("Vergleichstabelle gespeichert: %s", comparison_path)


def build_parser() -> argparse.ArgumentParser:
    """Erstellt den CLI-Parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Residuallast-Prognose: Daten laden, vorverarbeiten, Modelle trainieren, "
            "evaluieren und Forecasts erzeugen."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Lädt Rohdaten aus den APIs.")
    subparsers.add_parser("preprocess", help="Bereitet die Daten für das Modelltraining auf.")
    subparsers.add_parser("pipeline", help="Führt den kompletten Workflow für alle Modelle aus.")

    train_parser = subparsers.add_parser("train", help="Trainiert ein einzelnes Modell.")
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluiert ein gespeichertes Modell.")
    forecast_parser = subparsers.add_parser("forecast", help="Erzeugt eine Prognose mit einem gespeicherten Modell.")
    run_parser = subparsers.add_parser(
        "run",
        help="Trainiert, evaluiert und erzeugt danach einen Forecast.",
    )
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help=(
            "Conformal-Prediction-Kalibrierung (CQR) für gespeicherte Modelle. "
            "Kein Re-Training nötig — nutzt vorhandene Modelle und Test-Forecasts."
        ),
    )
    optimize_parser = subparsers.add_parser(
        "optimize",
        help=(
            "Hyperparameter-Optimierung mit Optuna. Bewertet Trials per "
            "Pinball Loss auf Validation; Test-Set bleibt unangetastet."
        ),
    )
    optimize_parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=(
            "Anzahl Optuna-Trials. Default modellabhängig "
            "(Regression 8, XGBoost 40, TFT 15)."
        ),
    )
    optimize_parser.add_argument(
        "--eval-stride",
        type=int,
        default=72,
        help=(
            "Backtest-Stride für die Validation-Bewertung während der HPO "
            "(Default 72 = alle 3 Tage, schneller). Finale Bewertung nutzt 24."
        ),
    )
    optimize_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Hartes Zeitlimit in Sekunden für die HPO (z.B. 28800 = 8 h). "
            "Optuna startet nach Ablauf keinen neuen Trial mehr. Schützt "
            "zusammen mit den beschränkten Suchräumen vor Endlos-Läufen."
        ),
    )

    final_train_parser = subparsers.add_parser(
        "final-train",
        help=(
            "Finales Training auf Train+Val mit den besten HPO-Konfigurationen, "
            "einmalige Testbewertung und ACI-Kalibrierung (Kapitel-4-Daten)."
        ),
    )

    for subparser in [train_parser, evaluate_parser, forecast_parser]:
        subparser.add_argument(
            "--model",
            required=True,
            choices=["regression", "tft", "xgboost"],
            help="Welches Modell verwendet werden soll.",
        )

    for subparser in [run_parser, calibrate_parser, optimize_parser, final_train_parser]:
        subparser.add_argument(
            "--model",
            required=True,
            choices=["regression", "tft", "xgboost", "all"],
            help=(
                "Welches Modell verarbeitet werden soll. "
                "'all' führt regression → tft → xgboost nacheinander aus."
            ),
        )

    # --target und --horizon gelten für alle Subparser
    for subparser in [train_parser, evaluate_parser, forecast_parser, run_parser, calibrate_parser, optimize_parser, final_train_parser]:
        subparser.add_argument(
            "--target",
            required=True,
            choices=["residual_load", "total_load", "solar", "wind_total"],
            help="Welche Zielgröße trainiert, evaluiert oder prognostiziert werden soll.",
        )
        subparser.add_argument(
            "--horizon",
            required=True,
            choices=["day_ahead", "week_ahead"],
            help="Welcher Prognosehorizont genutzt werden soll.",
        )

    return parser


def main():
    """CLI-Einstiegspunkt."""
    setup_plot_style()
    config = load_config()
    set_global_seeds(config["models"]["seed"])
    parser = build_parser()
    args = parser.parse_args()

    command_map = {
        "download": command_download,
        "preprocess": command_preprocess,
        "train": command_train,
        "evaluate": command_evaluate,
        "forecast": command_forecast,
        "run": command_run,
        "pipeline": command_pipeline,
        "calibrate": command_calibrate,
        "optimize": command_optimize,
        "final-train": command_final_train,
    }

    global _COMMAND_START_TIME
    start_time = time.perf_counter()
    _COMMAND_START_TIME = start_time
    print_timed_status(args.command, "Start", ANSI_CYAN)
    stop_event = threading.Event()

    # Lange, log-intensive Commands (Training, HPO) bekommen periodische
    # Statuszeilen statt einer überschreibenden Einzelzeile — sonst kollidiert
    # der Live-Status mit Optuna-/Trainings-Logs.
    use_single_line_status = not (
        (
            args.command in {"train", "run"}
            and getattr(args, "model", None) in {"tft", "all"}
        )
        or args.command in {"optimize", "final-train"}
    )

    status_thread = threading.Thread(
        target=run_live_status if use_single_line_status else run_periodic_status_lines,
        args=(args.command, stop_event),
        daemon=True,
    )
    status_thread.start()

    try:
        command_map[args.command](config, args)
    except Exception:
        elapsed = time.perf_counter() - start_time
        stop_event.set()
        status_thread.join(timeout=0.2)
        print_timed_status(args.command, "Fehlgeschlagen", ANSI_YELLOW, elapsed)
        raise

    elapsed = time.perf_counter() - start_time
    stop_event.set()
    status_thread.join(timeout=0.2)
    print_timed_status(args.command, "Abgeschlossen", ANSI_GREEN, elapsed)


if __name__ == "__main__":
    main()
