"""
SMARD API Datenloader
Lädt Stromlast- und Erzeugungsdaten von der SMARD API (Bundesnetzagentur).
"""

import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

# SMARD API Filter-IDs
SMARD_FILTERS = {
    "residual_load": 4359,
    "total_load": 410,
    "solar": 4068,
    "wind_onshore": 4067,
    "wind_offshore": 1225,
}

# SMARD Region-IDs
SMARD_REGIONS = {
    "DE": "DE",
    "50Hertz": "50Hertz",
    "Amprion": "Amprion",
    "TenneT": "TenneT",
    "TransnetBW": "TransnetBW",
}

BASE_URL = "https://www.smard.de/app/chart_data"


def _timestamp_to_ms(dt: datetime) -> int:
    """Konvertiert datetime zu Millisekunden-Timestamp (SMARD-Format)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _get_available_timestamps(
    filter_id: int, region: str, resolution: str = "hour"
) -> list[int]:
    """Holt verfügbare Timestamps für einen Filter von der SMARD API."""
    url = f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("timestamps", [])


def _get_chart_data(
    filter_id: int,
    region: str,
    timestamp: int,
    resolution: str = "hour",
) -> list[list]:
    """Holt Zeitreihendaten für einen bestimmten Timestamp-Block."""
    url = (
        f"{BASE_URL}/{filter_id}/{region}"
        f"/{filter_id}_{region}_{resolution}_{timestamp}.json"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("series", [])


def load_smard_series(
    filter_id: int,
    region: str = "DE",
    resolution: str = "hour",
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
    request_delay: float = 0.3,
) -> pd.DataFrame:
    """
    Lädt eine Zeitreihe von der SMARD API.

    Parameters
    ----------
    filter_id : int
        SMARD Filter-ID (z.B. 4359 für Residuallast).
    region : str
        Region (z.B. "DE").
    resolution : str
        Zeitliche Auflösung ("hour", "quarterhour", etc.).
    start_date, end_date : str
        Zeitraum im Format "YYYY-MM-DD".
    request_delay : float
        Pause zwischen API-Requests (Sekunden).

    Returns
    -------
    pd.DataFrame
        DataFrame mit DatetimeIndex (UTC) und Werte-Spalte.
    """
    start_ms = _timestamp_to_ms(datetime.strptime(start_date, "%Y-%m-%d"))
    end_ms = _timestamp_to_ms(datetime.strptime(end_date, "%Y-%m-%d"))

    logger.info(
        f"Lade SMARD-Daten: Filter={filter_id}, Region={region}, "
        f"{start_date} bis {end_date}"
    )

    timestamps = _get_available_timestamps(filter_id, region, resolution)
    relevant_ts = [t for t in timestamps if start_ms <= t <= end_ms]

    if not relevant_ts:
        logger.warning("Keine Timestamps im angegebenen Zeitraum gefunden.")
        return pd.DataFrame()

    all_data = []
    for i, ts in enumerate(relevant_ts):
        try:
            series = _get_chart_data(filter_id, region, ts, resolution)
            all_data.extend(series)
            if (i + 1) % 50 == 0:
                logger.info(f"  ... {i + 1}/{len(relevant_ts)} Blöcke geladen")
        except requests.RequestException as e:
            logger.warning(f"Fehler bei Timestamp {ts}: {e}")
        time.sleep(request_delay)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp_ms", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp").drop(columns=["timestamp_ms"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Zeitraum filtern
    df = df.loc[start_date:end_date]

    logger.info(f"  → {len(df)} Datenpunkte geladen ({df.index.min()} bis {df.index.max()})")
    return df


def load_all_smard_data(
    region: str = "DE",
    resolution: str = "hour",
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
    save_path: str | None = None,
    request_delay: float = 0.3,
) -> pd.DataFrame:
    """
    Lädt alle relevanten SMARD-Zeitreihen und kombiniert sie in einem DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame mit Spalten: residual_load, total_load, solar, wind_onshore, wind_offshore
    """
    dfs = {}
    for name, filter_id in SMARD_FILTERS.items():
        logger.info(f"\n--- Lade {name} (Filter {filter_id}) ---")
        df = load_smard_series(
            filter_id=filter_id,
            region=region,
            resolution=resolution,
            start_date=start_date,
            end_date=end_date,
            request_delay=request_delay,
        )
        if not df.empty:
            dfs[name] = df["value"]

    if not dfs:
        raise ValueError("Keine Daten von der SMARD API geladen!")

    combined = pd.DataFrame(dfs)

    # Wind gesamt berechnen
    if "wind_onshore" in combined.columns and "wind_offshore" in combined.columns:
        combined["wind_total"] = (
            combined["wind_onshore"].fillna(0) + combined["wind_offshore"].fillna(0)
        )

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path)
        logger.info(f"Daten gespeichert: {path}")

    logger.info(
        f"\nSMARD-Daten geladen: {combined.shape[0]} Zeilen, "
        f"{combined.shape[1]} Spalten"
    )
    logger.info(f"Zeitraum: {combined.index.min()} bis {combined.index.max()}")
    logger.info(f"Fehlende Werte:\n{combined.isna().sum()}")

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    smard_cfg = config["data"]["smard"]
    df = load_all_smard_data(
        region=smard_cfg["region"],
        resolution=smard_cfg["resolution"],
        start_date=smard_cfg["start_date"],
        end_date=smard_cfg["end_date"],
        save_path=f"{config['data']['paths']['raw']}/smard_data.csv",
    )
    print(df.head())
    print(df.describe())
