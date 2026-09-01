"""
Temporal Fusion Transformer (TFT)
Hauptmodell für probabilistische Residuallast-Prognose.
"""

import logging

import numpy as np
import torch
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch 2.6+ setzt weights_only=True als Default in torch.load.
# Darts-Checkpoints enthalten QuantileRegression-Objekte die damit nicht
# geladen werden können. Wir erlauben diese explizit als sicher.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

logger = logging.getLogger(__name__)


def build_tft_model(
    input_chunk_length: int = 168,
    output_chunk_length: int = 24,
    hidden_size: int = 128,
    lstm_layers: int = 1,
    num_attention_heads: int = 4,
    dropout: float = 0.3,
    batch_size: int = 64,
    n_epochs: int = 200,
    early_stopping_patience: int = 10,
    learning_rate: float = 0.001,
    quantiles: list[float] | None = None,
    random_state: int = 42,
    model_name: str = "tft",
    work_dir: str = "models",
    use_gpu: bool = True,
    early_stopping_enabled: bool = True,
) -> TFTModel:
    """
    Erstellt ein TFT-Modell.

    Parameters
    ----------
    input_chunk_length : int
        Lookback-Fenster (z.B. 168 = 1 Woche stündlich).
    output_chunk_length : int
        Prognosehorizont (z.B. 24 = Day-Ahead).
    hidden_size : int
        Größe der Hidden Layers (128 bewährt für dieses Feature-Set).
    lstm_layers : int
        Anzahl LSTM-Schichten.
    num_attention_heads : int
        Anzahl Attention Heads.
    dropout : float
        Dropout-Rate (0.3 für stärkere Regularisierung).
    batch_size : int
        Batch-Größe.
    n_epochs : int
        Maximale Trainingsepochen (Early Stopping stoppt vorher).
    early_stopping_patience : int
        Anzahl Epochen ohne Verbesserung, bevor gestoppt wird.
    learning_rate : float
        Lernrate.
    quantiles : list[float], optional
        Quantile für probabilistische Vorhersagen.
    random_state : int
        Seed für Reproduzierbarkeit.
    model_name : str
        Name des Modells (für Speicherung).
    work_dir : str
        Verzeichnis für Modellspeicherung.
    use_gpu : bool
        GPU verwenden falls verfügbar.

    Returns
    -------
    TFTModel
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # GPU-Erkennung
    # Hinweis: Apple MPS unterstützt kein float64, das Darts standardmäßig nutzt.
    # Daher verwenden wir CPU, was auf Apple Silicon trotzdem performant ist.
    pl_trainer_kwargs = {}
    if use_gpu and torch.cuda.is_available():
        pl_trainer_kwargs["accelerator"] = "gpu"
        pl_trainer_kwargs["devices"] = 1
        logger.info("GPU erkannt — Training auf GPU.")
    else:
        pl_trainer_kwargs["accelerator"] = "cpu"
        logger.info("Training auf CPU.")

    pl_trainer_kwargs["enable_progress_bar"] = True
    pl_trainer_kwargs["enable_model_summary"] = True
    pl_trainer_kwargs["log_every_n_steps"] = 10

    # Early Stopping: Stoppt automatisch wenn der Validierungsfehler
    # sich nach 'patience' Epochen nicht mehr verbessert.
    # Das beste Modell (niedrigster Val-Loss) wird automatisch gespeichert.
    # Beim finalen Training auf Train+Val gibt es keine Validierungsmenge
    # mehr — dann wird mit early_stopping_enabled=False eine feste
    # Epochenzahl trainiert (n_epochs = beste Epoche des Val-Laufs).
    if early_stopping_enabled:
        early_stopper = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            min_delta=0.0001,           # Minimale Verbesserung um als "besser" zu gelten
            mode="min",                 # Niedrigerer Loss = besser
            verbose=True,
        )
        pl_trainer_kwargs["callbacks"] = [early_stopper]

    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer_kwargs={"lr": learning_rate},
        # LR Scheduler: Halbiert die Lernrate automatisch wenn der überwachte
        # Verlust sich 8 Epochen lang nicht verbessert → feineres Lernen.
        # Beim finalen Training auf Train+Val existiert kein val_loss mehr —
        # dann überwacht der Scheduler den Trainingsverlust (sonst wirft
        # Lightning eine MisconfigurationException).
        lr_scheduler_cls=ReduceLROnPlateau,
        lr_scheduler_kwargs={
            "patience": 8,       # 8 Epochen warten bevor LR gesenkt wird
            "factor": 0.5,       # LR halbieren
            "min_lr": 1e-6,      # Nicht unter 0.000001 gehen
            "monitor": "val_loss" if early_stopping_enabled else "train_loss",
        },
        likelihood=QuantileRegression(quantiles=quantiles),
        random_state=random_state,
        model_name=model_name,
        work_dir=work_dir,
        save_checkpoints=True,
        force_reset=True,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

    logger.info(
        f"TFT erstellt: input={input_chunk_length}, output={output_chunk_length}, "
        f"hidden={hidden_size}, heads={num_attention_heads}, epochs={n_epochs}"
    )
    return model


def train_tft(
    model: TFTModel,
    target_train: TimeSeries,
    target_val: TimeSeries,
    past_cov_train: TimeSeries | None = None,
    past_cov_val: TimeSeries | None = None,
    future_cov_train: TimeSeries | None = None,
    future_cov_val: TimeSeries | None = None,
) -> TFTModel:
    """
    Trainiert das TFT-Modell mit Validierungsdaten.

    Parameters
    ----------
    model : TFTModel
        Untrainiertes Modell.
    target_train, target_val : TimeSeries
        Trainings- und Validierungs-Zielvariable.
    past_cov_train, past_cov_val : TimeSeries, optional
        Past Covariates.
    future_cov_train, future_cov_val : TimeSeries, optional
        Future Covariates.

    Returns
    -------
    TFTModel
        Trainiertes Modell.
    """
    logger.info("Starte TFT-Training...")

    model.fit(
        series=target_train,
        past_covariates=past_cov_train,
        future_covariates=future_cov_train,
        val_series=target_val,
        val_past_covariates=past_cov_val,
        val_future_covariates=future_cov_val,
        # WICHTIG: Nach Early Stopping die Gewichte der BESTEN Epoche laden,
        # nicht der letzten (übertrainierten) Epoche.
        load_best=True,
    )

    logger.info("TFT-Training abgeschlossen.")
    return model


def predict_tft(
    model: TFTModel,
    n: int,
    series: TimeSeries,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
    num_samples: int = 500,
) -> TimeSeries:
    """Erstellt probabilistische Vorhersage mit dem TFT-Modell."""
    prediction = model.predict(
        n=n,
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        num_samples=num_samples,
    )
    return prediction


def get_quantile_predictions(
    prediction: TimeSeries,
    quantiles: list[float] | None = None,
) -> np.ndarray:
    """Extrahiert Quantil-Vorhersagen. Shape: [T, Q]."""
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    quantile_values = []
    for q in quantiles:
        q_ts = prediction.quantile(q)
        quantile_values.append(q_ts.values().flatten())

    return np.column_stack(quantile_values)
