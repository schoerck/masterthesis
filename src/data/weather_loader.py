"""
Open-Meteo Wetterdaten Loader — Bundesland-basierte ERA5-Aggregation

Datenquelle: Open-Meteo Historical Weather API (https://archive-api.open-meteo.com)
             basiert intern auf ECMWF ERA5-Reanalyse (Hersbach et al. 2020).

Methodik:
  - 45 Sampling-Punkte verteilt auf 16 deutsche Bundesländer
  - Pro Bundesland: Durchschnitt über alle Sampling-Punkte → 1 BL-Zeitreihe
  - 3 deutschlandweite Aggregate mit physikalisch motivierten Gewichten:
      1. weather_temperature  : Temperatur, bevölkerungsgewichtet
                                (Heiz-/Kühlbedarf folgt der Bevölkerungsverteilung)
      2. weather_wind_100m    : Wind auf 100 m, windkapazitätsgewichtet
                                (relevante Nabenhöhe moderner Windturbinen)
      3. weather_radiation    : Globalstrahlung, PV-kapazitätsgewichtet
                                (PV-Einspeisung proportional zur installierten Kapazität)

Gewichtsquellen (Stand 2024):
  - Bevölkerung : Destatis, Tabelle 12411-0010 (Jahresende 2023)
  - Wind-MW     : Bundesnetzagentur MaStR-Veröffentlichungsstatistik 2024
                  (Onshore + anteiliger Offshore je Küstenstaat)
  - Solar-MWp   : Bundesnetzagentur MaStR-Veröffentlichungsstatistik 2024
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import yaml

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ============================================================
# 45 Sampling-Punkte verteilt auf 16 Bundesländer
# Anzahl Punkte proportional zur Fläche des Bundeslandes:
#   Bayern, NRW, BW, Niedersachsen: je 5-6 Punkte
#   Mittelgroße BL: je 3 Punkte
#   Kleine BL / Stadtstaaten: 1-2 Punkte
# Format: (Stadtname, Breitengrad, Längengrad)
# ============================================================
BUNDESLAND_SAMPLING_POINTS: dict[str, list[tuple[str, float, float]]] = {
    "Bayern": [
        ("München",     48.14, 11.58),
        ("Nürnberg",    49.45, 11.08),
        ("Augsburg",    48.37, 10.89),
        ("Würzburg",    49.79,  9.93),
        ("Regensburg",  49.01, 12.10),
        ("Ingolstadt",  48.76, 11.43),
    ],
    "Nordrhein-Westfalen": [
        ("Köln",       50.94,  6.96),
        ("Dortmund",   51.52,  7.47),
        ("Münster",    51.96,  7.63),
        ("Bielefeld",  52.02,  8.53),
        ("Aachen",     50.78,  6.08),
    ],
    "Baden-Württemberg": [
        ("Stuttgart",  48.78,  9.18),
        ("Karlsruhe",  49.01,  8.40),
        ("Freiburg",   47.99,  7.84),
        ("Ulm",        48.40,  9.99),
        ("Heidelberg", 49.41,  8.71),
    ],
    "Niedersachsen": [
        ("Hannover",      52.37,  9.74),
        ("Braunschweig",  52.27, 10.52),
        ("Oldenburg",     53.14,  8.21),
        ("Osnabrück",     52.28,  8.05),
        ("Lüneburg",      53.25, 10.41),
    ],
    "Hessen": [
        ("Frankfurt",  50.11,  8.68),
        ("Kassel",     51.31,  9.49),
        ("Gießen",     50.59,  8.68),
    ],
    "Rheinland-Pfalz": [
        ("Mainz",           49.99,  8.27),
        ("Trier",           49.75,  6.64),
        ("Kaiserslautern",  49.44,  7.77),
    ],
    "Sachsen": [
        ("Dresden",  51.05, 13.74),
        ("Leipzig",  51.34, 12.38),
        ("Chemnitz", 50.83, 12.92),
    ],
    "Brandenburg": [
        ("Potsdam",        52.39, 13.06),
        ("Cottbus",        51.76, 14.33),
        ("Frankfurt_Oder", 52.34, 14.55),
    ],
    "Sachsen-Anhalt": [
        ("Magdeburg",  52.13, 11.62),
        ("Halle",      51.48, 11.97),
    ],
    "Schleswig-Holstein": [
        ("Kiel",       54.32, 10.13),
        ("Flensburg",  54.79,  9.44),
    ],
    "Mecklenburg-Vorpommern": [
        ("Rostock",  54.09, 12.10),
        ("Schwerin", 53.63, 11.42),
    ],
    "Thüringen": [
        ("Erfurt",  50.98, 11.03),
        ("Jena",    50.93, 11.59),
    ],
    "Saarland": [
        ("Saarbrücken",  49.24,  7.00),
    ],
    "Hamburg": [
        ("Hamburg",  53.55,  9.99),
    ],
    "Bremen": [
        ("Bremen",  53.08,  8.80),
    ],
    "Berlin": [
        ("Berlin",  52.52, 13.41),
    ],
}

# ============================================================
# Bundesland-Gewichte aus offiziellen Primärquellen (Stand 2024)
# Format: (Bevölkerung Mio., Wind-Kapazität MW, PV-Kapazität MWp)
#
# Niedersachsen und Schleswig-Holstein enthalten anteiligen Offshore-Wind
# (Nordsee-Parks werden dem nächstgelegenen Küstenstaat zugerechnet)
# ============================================================
BUNDESLAND_WEIGHTS: dict[str, tuple[float, int, int]] = {
    # Bundesland              Bev. (Mio.)  Wind (MW)  Solar (MWp)
    "Bayern":                   (13.37,  2_610,  22_400),
    "Nordrhein-Westfalen":      (17.98,  7_400,  11_200),
    "Baden-Württemberg":        (11.28,  1_840,  11_000),
    "Niedersachsen":            ( 8.09, 15_550,   6_100),
    "Hessen":                   ( 6.39,  2_400,   3_900),
    "Rheinland-Pfalz":          ( 4.13,  4_000,   3_000),
    "Sachsen":                  ( 4.06,  1_450,   2_900),
    "Brandenburg":              ( 2.57,  8_200,   5_400),
    "Sachsen-Anhalt":           ( 2.19,  5_600,   3_400),
    "Schleswig-Holstein":       ( 2.95, 12_000,   2_650),
    "Mecklenburg-Vorpommern":   ( 1.63,  3_750,   2_200),
    "Thüringen":                ( 2.10,  1_650,   2_100),
    "Saarland":                 ( 0.99,    530,     580),
    "Hamburg":                  ( 1.90,    130,     220),
    "Bremen":                   ( 0.68,    190,     150),
    "Berlin":                   ( 3.76,     15,     240),
}

# Welche Gewichtsspalte (Index in BUNDESLAND_WEIGHTS-Tuple) für welche Variable
VARIABLE_WEIGHT_INDEX = {
    "temperature_2m":      0,   # → Bevölkerungsgewicht
    "wind_speed_100m":     1,   # → Wind-MW-Gewicht
    "shortwave_radiation": 2,   # → Solar-MWp-Gewicht
}

# Ausgabespaltenname je Variable
OUTPUT_COLUMN_NAMES = {
    "temperature_2m":      "weather_temperature",
    "wind_speed_100m":     "weather_wind_100m",
    "shortwave_radiation": "weather_radiation",
}


def _create_client() -> openmeteo_requests.Client:
    """Erstellt einen Open-Meteo Client mit Caching und Retry."""
    cache_session = requests_cache.CachedSession(
        ".openmeteo_cache", expire_after=-1
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _fetch_point(
    client: openmeteo_requests.Client,
    lat: float,
    lon: float,
    variables: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Lädt ERA5-Stundendaten für einen einzelnen Punkt via Open-Meteo.

    Returns
    -------
    pd.DataFrame
        Index: UTC-Zeitstempel, Spalten: Variablennamen.
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     variables,
        "timezone":   "UTC",
    }

    responses = client.weather_api(ARCHIVE_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()

    time_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data = {"timestamp": time_range}
    for i, var in enumerate(variables):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data).set_index("timestamp")
    return df


def load_all_weather_data(
    variables: list[str],
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Lädt ERA5-Wetterdaten für alle 45 Sampling-Punkte und aggregiert zu
    3 deutschlandweiten, kapazitätsgewichteten Indikatoren.

    Parameters
    ----------
    variables : list[str]
        ERA5-Variablen (müssen in VARIABLE_WEIGHT_INDEX definiert sein).
        Erwartet: ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]
    start_date, end_date : str
        Zeitraum im Format "YYYY-MM-DD".
    save_path : str, optional
        Speicherpfad für die fertige CSV.

    Returns
    -------
    pd.DataFrame
        Stündliche Zeitreihe mit 3 Spalten:
        weather_temperature, weather_wind_100m, weather_radiation
    """
    # Validierung
    unknown = [v for v in variables if v not in VARIABLE_WEIGHT_INDEX]
    if unknown:
        raise ValueError(
            f"Unbekannte Variablen: {unknown}. "
            f"Erlaubt: {list(VARIABLE_WEIGHT_INDEX.keys())}"
        )

    client = _create_client()
    total_points = sum(len(pts) for pts in BUNDESLAND_SAMPLING_POINTS.values())
    logger.info(
        f"Lade ERA5-Daten für {total_points} Sampling-Punkte "
        f"in {len(BUNDESLAND_SAMPLING_POINTS)} Bundesländern "
        f"({start_date} bis {end_date}) ..."
    )

    # ----------------------------------------------------------------
    # Schritt 1: Pro Bundesland Mittelwert über alle Sampling-Punkte
    # ----------------------------------------------------------------
    bundesland_means: dict[str, pd.DataFrame] = {}
    fetched = 0

    for bl_name, points in BUNDESLAND_SAMPLING_POINTS.items():
        point_dfs = []
        for city_name, lat, lon in points:
            try:
                df = _fetch_point(client, lat, lon, variables, start_date, end_date)
                point_dfs.append(df)
                fetched += 1
                logger.info(
                    f"  [{fetched:2d}/{total_points}] {bl_name} – {city_name} "
                    f"({lat:.2f}°N, {lon:.2f}°E): {len(df)} Stunden"
                )
            except Exception as e:
                logger.warning(f"  Fehler bei {bl_name}/{city_name}: {e} — wird übersprungen")

        if not point_dfs:
            logger.error(f"Keine Daten für Bundesland {bl_name}!")
            continue

        # Gleichgewichteter Durchschnitt über alle Punkte im BL
        bl_mean = pd.concat(point_dfs).groupby(level=0).mean()
        bundesland_means[bl_name] = bl_mean
        logger.info(f"  → {bl_name}: Mittelwert aus {len(point_dfs)} Punkten berechnet")

    if not bundesland_means:
        raise RuntimeError("Keine Bundesland-Daten konnten geladen werden!")

    # ----------------------------------------------------------------
    # Schritt 2: Kapazitätsgewichtete Aggregation über Bundesländer
    # ----------------------------------------------------------------
    logger.info("\nBerechne kapazitätsgewichtete Deutschlandaggregate ...")

    # Gemeinsamen Zeitindex sicherstellen
    common_index = next(iter(bundesland_means.values())).index
    for bl_df in bundesland_means.values():
        common_index = common_index.intersection(bl_df.index)

    result = pd.DataFrame(index=common_index)

    for var in variables:
        w_idx = VARIABLE_WEIGHT_INDEX[var]
        out_col = OUTPUT_COLUMN_NAMES[var]

        total_weight = 0.0
        weighted_sum = pd.Series(0.0, index=common_index)

        for bl_name, bl_df in bundesland_means.items():
            weight = BUNDESLAND_WEIGHTS[bl_name][w_idx]
            total_weight += weight
            weighted_sum += bl_df.loc[common_index, var] * weight

        result[out_col] = weighted_sum / total_weight

        # Log Gewichtsverteilung (Top-5)
        weights = {bl: BUNDESLAND_WEIGHTS[bl][w_idx] for bl in bundesland_means}
        top5 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        weight_info = ", ".join(f"{bl} {w/total_weight:.0%}" for bl, w in top5)
        logger.info(f"  {out_col}: Top-5 Gewichte → {weight_info}")

    logger.info(
        f"\nWetterdaten fertig: {result.shape[0]} Stunden, "
        f"{result.shape[1]} Features | "
        f"{result.index.min()} bis {result.index.max()}"
    )
    logger.info(f"Fehlende Werte:\n{result.isna().sum()}")

    # ----------------------------------------------------------------
    # Plausibilitätscheck
    # ----------------------------------------------------------------
    temp_col = OUTPUT_COLUMN_NAMES.get("temperature_2m")
    if temp_col and temp_col in result.columns:
        t_mean = result[temp_col].mean()
        if not (-5 < t_mean < 20):
            logger.warning(
                f"Ungewöhnliche Temperatur (Jahresmittel): {t_mean:.1f}°C — "
                "bitte Einheiten prüfen!"
            )
        else:
            logger.info(f"  Plausibilitätscheck Temperatur: {t_mean:.1f}°C Jahresmittel ✓")

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path)
        logger.info(f"Wetterdaten gespeichert: {path}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    weather_cfg = config["data"]["weather"]
    df = load_all_weather_data(
        variables=weather_cfg["variables"],
        start_date=weather_cfg["start_date"],
        end_date=weather_cfg["end_date"],
        save_path=f"{config['data']['paths']['raw']}/weather_data.csv",
    )
    print(df.head(10))
    print("\nStatistiken:")
    print(df.describe())
