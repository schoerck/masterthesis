"""
Preprocessing Pipeline
Kombiniert SMARD- und Wetterdaten, erzeugt Features und erstellt Darts TimeSeries.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import holidays
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import yaml

logger = logging.getLogger(__name__)


def load_raw_data(
    raw_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Lädt gespeicherte Rohdaten aus CSV.

    Returns
    -------
    tuple
        (smard, weather, weather_forecast). weather_forecast ist None,
        falls die Datei (noch) nicht existiert — der Aufrufer muss das
        dann separat anstoßen.
    """
    smard_path = Path(raw_path) / "smard_data.csv"
    weather_path = Path(raw_path) / "weather_data.csv"
    weather_forecast_path = Path(raw_path) / "weather_forecast_data.csv"

    smard = pd.read_csv(smard_path, index_col=0, parse_dates=True)
    weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)

    if weather_forecast_path.exists():
        weather_forecast = pd.read_csv(
            weather_forecast_path, index_col=0, parse_dates=True
        )
        logger.info(
            f"SMARD: {smard.shape}, Wetter-Ist: {weather.shape}, "
            f"Wetter-Forecast: {weather_forecast.shape}"
        )
    else:
        weather_forecast = None
        logger.warning(
            f"SMARD: {smard.shape}, Wetter-Ist: {weather.shape}, "
            f"Wetter-Forecast: NICHT VORHANDEN ({weather_forecast_path}) — "
            "bitte zuerst weather_forecast_loader laufen lassen."
        )

    return smard, weather, weather_forecast


def combine_data(
    smard: pd.DataFrame,
    weather: pd.DataFrame,
    target_freq: str = "1h",
    weather_forecast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Kombiniert SMARD-, Wetter-Ist- und Wetter-Forecast-Daten auf gemeinsamen
    stündlichen Index.

    Parameters
    ----------
    smard : pd.DataFrame
        SMARD-Daten (Residuallast, Gesamtlast, Solar, Wind).
    weather : pd.DataFrame
        Gewichtete ERA5-Ist-Wetterdaten (→ später als Past Covariate verwendet).
    target_freq : str
        Zielfrequenz für Resampling.
    weather_forecast : pd.DataFrame, optional
        Gewichtete historische Wetter-Forecasts (→ später als Future Covariate).
        Spaltennamen tragen Suffix `_forecast`. Wenn None, läuft Pipeline ohne
        Forecast-Features weiter (rückwärtskompatibel).

    Returns
    -------
    pd.DataFrame
        Kombinierter DataFrame, Inner-Join über alle Quellen.
    """
    smard_h = smard.resample(target_freq).mean()
    weather_h = weather.resample(target_freq).mean()
    combined = smard_h.join(weather_h, how="inner")

    if weather_forecast is not None:
        weather_fc_h = weather_forecast.resample(target_freq).mean()
        # Inner-Join, damit ausschließlich Zeitpunkte erhalten bleiben,
        # für die SMARD + Ist + Forecast verfügbar sind.
        combined = combined.join(weather_fc_h, how="inner")
        logger.info(
            f"Kombinierte Daten: {combined.shape[0]} Zeilen, "
            f"{combined.shape[1]} Spalten (inkl. {weather_fc_h.shape[1]} "
            f"Wetter-Forecast-Features)"
        )
    else:
        logger.info(
            f"Kombinierte Daten: {combined.shape[0]} Zeilen, "
            f"{combined.shape[1]} Spalten (ohne Wetter-Forecasts)"
        )
    return combined


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt kalendarische Zeitfeatures hinzu."""
    df = df.copy()
    idx = df.index

    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["day_of_year"] = idx.dayofyear
    df["week_of_year"] = idx.isocalendar().week.astype(int).values
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Deutsche Feiertage
    de_holidays = holidays.Germany(years=list(range(idx.year.min(), idx.year.max() + 1)))
    df["is_holiday"] = idx.normalize().isin(de_holidays.keys()).astype(int)

    # Zyklische Kodierung für Stunde und Monat
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt gelaggte Energiefeatures hinzu.

    Statt die rohen Energievariablen (total_load, solar, wind_total) als
    Past Covariates zu verwenden (→ Data Leakage, da residual_load direkt
    daraus berechenbar ist), erzeugen wir zeitverzögerte Versionen.

    Lag-24:  "Wie war der Wert gestern zur gleichen Uhrzeit?"
    Lag-168: "Wie war der Wert letzte Woche zur gleichen Uhrzeit?"

    So bekommt das Modell die Lastinformation OHNE Data Leakage.
    """
    df = df.copy()

    energy_cols = ["total_load", "solar", "wind_total"]
    for col in energy_cols:
        if col in df.columns:
            df[f"{col}_lag24"] = df[col].shift(24)
            df[f"{col}_lag168"] = df[col].shift(168)

    # Erste 168 Zeilen haben NaN durch shift → entfernen (kein bfill,
    # da bfill zukünftige Werte einfüllen würde = Data Leakage)
    df = df.iloc[168:]

    logger.info(
        "Lag-Features erzeugt: %s",
        [f"{c}_lag24/lag168" for c in energy_cols if c in df.columns],
    )
    return df


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "linear",
    max_gap_hours: int = 6,
) -> pd.DataFrame:
    """Interpoliert fehlende Werte (nur für kurze Lücken)."""
    df = df.copy()
    n_missing_before = df.isna().sum().sum()

    df = df.interpolate(method=method, limit=max_gap_hours)

    # Verbleibende NaNs (lange Lücken) loggen
    n_missing_after = df.isna().sum().sum()
    if n_missing_after > 0:
        logger.warning(
            f"Noch {n_missing_after} fehlende Werte nach Interpolation "
            f"(vorher: {n_missing_before}). Werden mit forward-fill aufgefüllt."
        )
        df = df.ffill().bfill()

    return df


def split_data(
    df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologischer Train/Val/Test Split anhand expliziter Kalender-Daten.

    Vorteil gegenüber ratio-basiertem Splitting: Validation und Test umfassen
    volle Kalenderjahre und decken damit alle Saisons (Winter-Heizlast,
    Sommer-PV-Spitzen) gleichermaßen ab. Das ist methodisch zwingend, weil
    die Residuallast eine starke Jahressaisonalität hat — eine kürzere
    Test-Periode würde systematisch bestimmte Lastregime auslassen und die
    Bewertung verzerren.

    Parameters
    ----------
    df : pd.DataFrame
        Aufbereiteter DataFrame mit DatetimeIndex (stündlich).
    train_end_date : str
        Letzter Tag des Train-Sets im Format "YYYY-MM-DD" (einschließlich).
        Beispiel: "2023-12-31" → Train enthält 2023-12-31 23:00 als letzte Zeile.
    val_end_date : str
        Letzter Tag des Validation-Sets im Format "YYYY-MM-DD" (einschließlich).
        Test-Set umfasst alle Daten danach bis zum Ende des DataFrames.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, Validation, Test — strikt nicht-überlappend.
    """
    # Explizite Zeitstempel-Grenzen — robuster als reines Partial-String-Slicing,
    # weil wir hier garantieren, dass nichts auf der Datumsgrenze "verschluckt"
    # oder doppelt vergeben wird.
    # Wichtig: tz des DataFrame-Index übernehmen, damit kein TypeError beim
    # Vergleich tz-aware vs. tz-naive auftritt (SMARD-Daten kommen z.B. in UTC).
    tz = df.index.tz
    train_end_ts = pd.Timestamp(train_end_date, tz=tz) + pd.Timedelta(hours=23)
    val_start_ts = pd.Timestamp(train_end_date, tz=tz) + pd.Timedelta(days=1)
    val_end_ts = pd.Timestamp(val_end_date, tz=tz) + pd.Timedelta(hours=23)
    test_start_ts = pd.Timestamp(val_end_date, tz=tz) + pd.Timedelta(days=1)

    train = df.loc[:train_end_ts]
    val = df.loc[val_start_ts:val_end_ts]
    test = df.loc[test_start_ts:]

    # Sanity-Check: keine Überlappung, keine Lücken
    if not train.empty and not val.empty:
        assert train.index.max() < val.index.min(), \
            f"Train/Val-Überlappung: train_max={train.index.max()}, val_min={val.index.min()}"
    if not val.empty and not test.empty:
        assert val.index.max() < test.index.min(), \
            f"Val/Test-Überlappung: val_max={val.index.max()}, test_min={test.index.min()}"

    logger.info(
        f"Split: Train={len(train)} ({train.index.min()} bis {train.index.max()}), "
        f"Val={len(val)} ({val.index.min()} bis {val.index.max()}), "
        f"Test={len(test)} ({test.index.min()} bis {test.index.max()})"
    )
    return train, val, test


def create_darts_datasets(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "residual_load",
    past_cov_cols: list[str] | None = None,
    future_cov_cols: list[str] | None = None,
) -> dict:
    """
    Erstellt Darts TimeSeries Objekte und skaliert die Daten.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Aufgeteilte Daten.
    target_col : str
        Zielvariable.
    past_cov_cols : list[str], optional
        Spalten für Past Covariates.
    future_cov_cols : list[str], optional
        Spalten für Future Covariates.

    Returns
    -------
    dict
        Dictionary mit allen TimeSeries und Scalern.
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    for df in [train, val, test]:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

    # Default Covariates
    if past_cov_cols is None:
        # --- Gelaggte Energiefeatures (kein Data Leakage) ---
        # Statt der rohen Werte (residual_load = total_load - solar - wind)
        # verwenden wir 24h- und 168h-Lags: "Wie war es gestern / letzte Woche?"
        lag_cols = [c for c in train.columns if c.endswith(("_lag24", "_lag168"))]

        # --- Wetter-ISTWERTE (ERA5) als Past Covariate ---
        # 3 ERA5-basierte Aggregate (weather_temperature, weather_wind_100m,
        # weather_radiation). Explizit OHNE die _forecast-Spalten — die sind
        # zukunftsgerichtet und kommen weiter unten in future_cov_cols.
        weather_ist_cols = [
            c for c in train.columns
            if c.startswith("weather_") and not c.endswith("_forecast")
        ]

        past_cov_cols = lag_cols + weather_ist_cols
        logger.info(
            "Past Covariates: %d Lag-Features + %d Wetter-Ist = %d gesamt",
            len(lag_cols), len(weather_ist_cols), len(past_cov_cols),
        )

    if future_cov_cols is None:
        # --- Kalender-Features ---
        calendar_cols = [
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "dow_sin", "dow_cos", "is_weekend", "is_holiday",
        ]
        # --- Wetter-FORECASTS (Open-Meteo) als Future Covariate ---
        # Methodisch wichtig: historische Forecasts (mit echten Vorhersage-
        # fehlern) statt ERA5-Ist, sonst hätte das Modell perfekten Wetter-
        # blick in die Zukunft.
        weather_forecast_cols = [
            c for c in train.columns
            if c.startswith("weather_") and c.endswith("_forecast")
        ]
        future_cov_cols = calendar_cols + weather_forecast_cols
        logger.info(
            "Future Covariates: %d Kalender + %d Wetter-Forecast = %d gesamt",
            len(calendar_cols), len(weather_forecast_cols), len(future_cov_cols),
        )

    # Nur vorhandene Spalten verwenden
    past_cov_cols = [c for c in past_cov_cols if c in train.columns]
    future_cov_cols = [c for c in future_cov_cols if c in train.columns]

    # Target TimeSeries
    full_df = pd.concat([train, val, test])

    target_train = TimeSeries.from_dataframe(train, value_cols=target_col, freq="h")
    target_val = TimeSeries.from_dataframe(val, value_cols=target_col, freq="h")
    target_test = TimeSeries.from_dataframe(test, value_cols=target_col, freq="h")
    target_full = TimeSeries.from_dataframe(full_df, value_cols=target_col, freq="h")

    # Past Covariates
    past_cov_train = TimeSeries.from_dataframe(train, value_cols=past_cov_cols, freq="h")
    past_cov_val = TimeSeries.from_dataframe(val, value_cols=past_cov_cols, freq="h")
    past_cov_test = TimeSeries.from_dataframe(test, value_cols=past_cov_cols, freq="h")
    past_cov_full = TimeSeries.from_dataframe(full_df, value_cols=past_cov_cols, freq="h")

    # Future Covariates
    future_cov_train = TimeSeries.from_dataframe(train, value_cols=future_cov_cols, freq="h")
    future_cov_val = TimeSeries.from_dataframe(val, value_cols=future_cov_cols, freq="h")
    future_cov_test = TimeSeries.from_dataframe(test, value_cols=future_cov_cols, freq="h")
    future_cov_full = TimeSeries.from_dataframe(full_df, value_cols=future_cov_cols, freq="h")

    # Skalierung (fit nur auf Trainingsdaten!)
    target_scaler = Scaler()
    past_cov_scaler = Scaler()
    future_cov_scaler = Scaler()

    target_train_scaled = target_scaler.fit_transform(target_train)
    target_val_scaled = target_scaler.transform(target_val)
    target_test_scaled = target_scaler.transform(target_test)

    past_cov_train_scaled = past_cov_scaler.fit_transform(past_cov_train)
    past_cov_val_scaled = past_cov_scaler.transform(past_cov_val)
    past_cov_test_scaled = past_cov_scaler.transform(past_cov_test)
    past_cov_full_scaled = past_cov_scaler.transform(past_cov_full)

    future_cov_train_scaled = future_cov_scaler.fit_transform(future_cov_train)
    future_cov_val_scaled = future_cov_scaler.transform(future_cov_val)
    future_cov_test_scaled = future_cov_scaler.transform(future_cov_test)
    future_cov_full_scaled = future_cov_scaler.transform(future_cov_full)

    return {
        # Unskaliert
        "target_train": target_train,
        "target_val": target_val,
        "target_test": target_test,
        "target_full": target_full,
        # Skaliert
        "target_train_scaled": target_train_scaled,
        "target_val_scaled": target_val_scaled,
        "target_test_scaled": target_test_scaled,
        # Covariates skaliert
        "past_cov_train": past_cov_train_scaled,
        "past_cov_val": past_cov_val_scaled,
        "past_cov_test": past_cov_test_scaled,
        "past_cov_full": past_cov_full_scaled,
        "future_cov_train": future_cov_train_scaled,
        "future_cov_val": future_cov_val_scaled,
        "future_cov_test": future_cov_test_scaled,
        "future_cov_full": future_cov_full_scaled,
        # Scaler (für Inverse-Transform)
        "target_scaler": target_scaler,
        "past_cov_scaler": past_cov_scaler,
        "future_cov_scaler": future_cov_scaler,
        # Spalteninformationen
        "target_col": target_col,
        "past_cov_cols": past_cov_cols,
        "future_cov_cols": future_cov_cols,
    }


def run_preprocessing(config: dict) -> tuple[dict, pd.DataFrame]:
    """
    Führt die komplette Preprocessing Pipeline aus.

    Parameters
    ----------
    config : dict
        Konfiguration aus config.yaml.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        (Darts-Datasets Dictionary, aufbereiteter DataFrame)
    """
    raw_path = config["data"]["paths"]["raw"]
    prep_cfg = config["preprocessing"]

    # 1. Rohdaten laden (inkl. Wetter-Forecasts, falls vorhanden)
    smard, weather, weather_forecast = load_raw_data(raw_path)

    # 2. Kombinieren
    combined = combine_data(
        smard, weather, prep_cfg["target_frequency"], weather_forecast
    )

    # 3. Zeitfeatures
    combined = add_time_features(combined)

    # 3b. Gelaggte Energiefeatures
    combined = add_lag_features(combined)

    # 4. Fehlende Werte
    combined = handle_missing_values(
        combined,
        method=prep_cfg["interpolation_method"],
        max_gap_hours=prep_cfg["max_gap_hours"],
    )

    # 5. Speichern
    processed_path = Path(config["data"]["paths"]["processed"]) / "combined_data.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(processed_path)

    # 6. Split
    train, val, test = split_data(
        combined,
        train_end_date=prep_cfg["train_end_date"],
        val_end_date=prep_cfg["val_end_date"],
    )

    # 7. Darts TimeSeries erstellen
    datasets = create_darts_datasets(train, val, test, target_col="residual_load")

    logger.info("Preprocessing abgeschlossen!")
    return datasets, combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    datasets, combined = run_preprocessing(config)
    print(f"Target Train: {datasets['target_train_scaled']}")
    print(f"Past Covariates: {datasets['past_cov_cols']}")
    print(f"Future Covariates: {datasets['future_cov_cols']}")
