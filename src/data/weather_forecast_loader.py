"""
Open-Meteo Historical Forecast Loader — historische Wettervorhersagen

Datenquelle: Open-Meteo Historical Forecast API
             https://historical-forecast-api.open-meteo.com/v1/forecast

Methodik:
  Für jeden vergangenen Zeitpunkt wird der damals tatsächlich abgegebene
  Wetterforecast zurückgegeben (basierend auf dem ECMWF IFS-Modell). Im
  Gegensatz zur ERA5-Reanalyse enthalten diese Daten die echten
  Prognosefehler, die ein realer Day-Ahead-Forecaster zur damaligen Zeit
  gehabt hätte.

Warum das wichtig ist:
  Wenn wir ERA5-Ist-Werte als Future Covariates verwenden würden, hätte das
  Modell einen "perfekten Wetterblick" in die Zukunft — das ist in der
  Praxis nicht verfügbar und würde die Bewertung systematisch zu optimistisch
  machen. Historische Forecasts hingegen sind die ehrlichste verfügbare
  Information zur jeweiligen Forecast-Zeit.

Konsistenz mit weather_loader.py:
  - Gleiche 45 Sampling-Punkte und Bundesländer
  - Gleiche kapazitätsgewichtete Aggregation
  - Ergebnis: 3 deutschlandweite Aggregate, parallel zu den Ist-Aggregaten
      weather_temperature_forecast
      weather_wind_100m_forecast
      weather_radiation_forecast
"""

import logging
from pathlib import Path

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import yaml

# Sampling-Punkte und Gewichte direkt aus dem bestehenden Loader übernehmen
# — so bleibt die Konfiguration an einer einzigen Stelle gepflegt.
from src.data.weather_loader import (
    BUNDESLAND_SAMPLING_POINTS,
    BUNDESLAND_WEIGHTS,
    VARIABLE_WEIGHT_INDEX,
)

logger = logging.getLogger(__name__)

FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Ausgabespaltennamen — Suffix _forecast unterscheidet von den ERA5-Ist-Spalten
OUTPUT_COLUMN_NAMES_FORECAST = {
    "temperature_2m":      "weather_temperature_forecast",
    "wind_speed_100m":     "weather_wind_100m_forecast",
    "shortwave_radiation": "weather_radiation_forecast",
}


def _create_client() -> openmeteo_requests.Client:
    """Open-Meteo Client mit Cache und Retry. Eigener Cache-Namespace,
    damit Archive- und Forecast-Calls sich nicht gegenseitig invalidieren."""
    cache_session = requests_cache.CachedSession(
        ".openmeteo_forecast_cache", expire_after=-1
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
    """Lädt historische Forecast-Daten für einen einzelnen Punkt."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     variables,
        "timezone":   "UTC",
    }

    responses = client.weather_api(FORECAST_URL, params=params)
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


def load_all_weather_forecast_data(
    variables: list[str],
    start_date: str = "2021-01-01",
    end_date: str = "2025-12-31",
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Lädt historische Wetterforecasts für alle 45 Sampling-Punkte und
    aggregiert zu 3 deutschlandweiten, kapazitätsgewichteten Indikatoren —
    parallel zu den ERA5-Ist-Aggregaten aus weather_loader.py.

    Returns
    -------
    pd.DataFrame
        Stündliche Zeitreihe mit 3 Spalten:
        weather_temperature_forecast, weather_wind_100m_forecast,
        weather_radiation_forecast
    """
    unknown = [v for v in variables if v not in VARIABLE_WEIGHT_INDEX]
    if unknown:
        raise ValueError(
            f"Unbekannte Variablen: {unknown}. "
            f"Erlaubt: {list(VARIABLE_WEIGHT_INDEX.keys())}"
        )

    client = _create_client()
    total_points = sum(len(pts) for pts in BUNDESLAND_SAMPLING_POINTS.values())
    logger.info(
        f"Lade Open-Meteo Historical Forecasts für {total_points} Punkte "
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
                    f"({lat:.2f}°N, {lon:.2f}°E): {len(df)} Stunden Forecast"
                )
            except Exception as e:
                logger.warning(
                    f"  Fehler bei {bl_name}/{city_name}: {e} — wird übersprungen"
                )

        if not point_dfs:
            logger.error(f"Keine Forecast-Daten für Bundesland {bl_name}!")
            continue

        bl_mean = pd.concat(point_dfs).groupby(level=0).mean()
        bundesland_means[bl_name] = bl_mean
        logger.info(
            f"  → {bl_name}: Mittelwert aus {len(point_dfs)} Forecast-Punkten"
        )

    if not bundesland_means:
        raise RuntimeError(
            "Keine Forecast-Bundesland-Daten konnten geladen werden!"
        )

    # ----------------------------------------------------------------
    # Schritt 2: Kapazitätsgewichtete Aggregation über Bundesländer
    # ----------------------------------------------------------------
    logger.info(
        "\nBerechne kapazitätsgewichtete Deutschland-Forecast-Aggregate ..."
    )

    common_index = next(iter(bundesland_means.values())).index
    for bl_df in bundesland_means.values():
        common_index = common_index.intersection(bl_df.index)

    result = pd.DataFrame(index=common_index)

    for var in variables:
        w_idx = VARIABLE_WEIGHT_INDEX[var]
        out_col = OUTPUT_COLUMN_NAMES_FORECAST[var]

        total_weight = 0.0
        weighted_sum = pd.Series(0.0, index=common_index)

        for bl_name, bl_df in bundesland_means.items():
            weight = BUNDESLAND_WEIGHTS[bl_name][w_idx]
            total_weight += weight
            weighted_sum += bl_df.loc[common_index, var] * weight

        result[out_col] = weighted_sum / total_weight

        weights = {bl: BUNDESLAND_WEIGHTS[bl][w_idx] for bl in bundesland_means}
        top5 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        weight_info = ", ".join(
            f"{bl} {w/total_weight:.0%}" for bl, w in top5
        )
        logger.info(f"  {out_col}: Top-5 Gewichte → {weight_info}")

    logger.info(
        f"\nForecast-Daten fertig: {result.shape[0]} Stunden, "
        f"{result.shape[1]} Features | "
        f"{result.index.min()} bis {result.index.max()}"
    )
    logger.info(f"Fehlende Werte:\n{result.isna().sum()}")

    # ----------------------------------------------------------------
    # Plausibilitätscheck: Forecast-Temperatur sollte ähnlich zum
    # ERA5-Ist liegen (Jahresmittel im plausiblen Bereich)
    # ----------------------------------------------------------------
    temp_col = OUTPUT_COLUMN_NAMES_FORECAST.get("temperature_2m")
    if temp_col and temp_col in result.columns:
        t_mean = result[temp_col].mean()
        if not (-5 < t_mean < 20):
            logger.warning(
                f"Ungewöhnliche Forecast-Temperatur (Jahresmittel): "
                f"{t_mean:.1f}°C — Einheiten prüfen!"
            )
        else:
            logger.info(
                f"  Plausibilitätscheck Forecast-Temperatur: "
                f"{t_mean:.1f}°C Jahresmittel ✓"
            )

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path)
        logger.info(f"Forecast-Daten gespeichert: {path}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    weather_cfg = config["data"]["weather"]
    df = load_all_weather_forecast_data(
        variables=weather_cfg["variables"],
        start_date=weather_cfg["start_date"],
        end_date=weather_cfg["end_date"],
        save_path=f"{config['data']['paths']['raw']}/weather_forecast_data.csv",
    )
    print(df.head(10))
    print("\nStatistiken:")
    print(df.describe())
