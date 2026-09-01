"""
Analysen zur Absicherung der Kernaussagen (Vorgutachten-Punkte).

A) Sensitivitaetsanalyse der Adaptionsschrittweite gamma (0,5x / 1x / 2x)
B) Signifikanz der Modellvergleiche: Diebold-Mariano (Newey-West) und
   Moving-Block-Bootstrap fuer MAE- und Band-Pinball-Differenzen
C) Multivariate Robustheitspruefung der Treiberanalyse (OLS der Bandbreite
   auf Wettertreiber, Tagesstunde, Jahreszeit, Werktag)
D) Laenderspezifische Feiertage: MAE-Vergleich
E) Monatsabdeckung roh/ACI je Modell (Anhangstabelle)
F) Operative Referenz: SMARD-Residuallastprognose (Filter 715) fuer 2025
   abrufen (Cache) und mit denselben Punktmetriken bewerten

Aufruf:  .venv/bin/python scripts/make_gutachten_analysen.py
"""

from pathlib import Path
import json
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "output" / "final"
OUT = ROOT / "output" / "analysen"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {"tft": "TFT", "xgboost": "XGBoost", "regression": "Regression"}
ALPHA = 0.2

# ----------------------------------------------------------------- Daten
def load_model(tag):
    df = pd.read_csv(FINAL / f"{tag}_residual_load_day_ahead_final_forecast_calibrated.csv",
                     parse_dates=["timestamp"]).set_index("timestamp")
    meta = json.load(open(FINAL / f"{tag}_residual_load_day_ahead_final_meta.json"))
    return df, meta["aci"]

DATA = {m: load_model(m) for m in MODELS}
IDX = DATA["tft"][0].index
ACT = DATA["tft"][0]["actual"]

print("=" * 78)
print("A) GAMMA-SENSITIVITAET (ACI mit 0,5x / 1x / 2x der gewaehlten Schrittweite)")
print("=" * 78)

def run_aci(df, c0, gamma):
    q10 = df["q10"].to_numpy(); q90 = df["q90"].to_numpy(); y = df["actual"].to_numpy()
    c = c0; cov = 0; width = 0.0
    for i in range(len(y)):
        lo, hi = q10[i] - c, q90[i] + c
        hit = lo <= y[i] <= hi
        cov += hit
        width += hi - lo
        c = c + gamma * ((0 if hit else 1) - ALPHA)
    n = len(y)
    return cov / n, width / n

rows_a = []
for m, name in MODELS.items():
    df, aci = DATA[m]
    for f in (0.5, 1.0, 2.0):
        cov, w = run_aci(df, aci["q_hat_init"], f * aci["gamma"])
        rows_a.append((name, f, f * aci["gamma"], cov, w))
        print(f"{name:11s} {f:3.1f}x gamma={f*aci['gamma']:6.2f} MW  "
              f"Abdeckung {100*cov:5.2f} %  Breite {w:7.0f} MW")
    print(f"{'':11s} Pipeline-Referenz: Abdeckung {100*aci['test_coverage_calibrated']:5.2f} % "
          f"Breite {aci['test_width_calibrated_mean']:7.0f} MW")
pd.DataFrame(rows_a, columns=["Modell", "Faktor", "gamma_MW", "Abdeckung", "Breite"]).to_csv(
    OUT / "gamma_sensitivitaet.csv", index=False)

print()
print("=" * 78)
print("B) SIGNIFIKANZ: DIEBOLD-MARIANO (Newey-West, Lag 47) + BLOCK-BOOTSTRAP")
print("=" * 78)

def pinball(y, q, tau):
    e = y - q
    return np.where(e >= 0, tau * e, (tau - 1) * e)

LOSS = {}
for m in MODELS:
    df = DATA[m][0]
    y = df["actual"].to_numpy()
    LOSS[m] = {
        "MAE": np.abs(y - df["prediction"].to_numpy()),
        "BandPinball": 0.5 * (pinball(y, df["q10"].to_numpy(), 0.10)
                              + pinball(y, df["q90"].to_numpy(), 0.90)),
    }

def dm_test(d, lag=47):
    n = len(d); dbar = d.mean(); dc = d - dbar
    s = dc @ dc / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (dc[:-k] @ dc[k:]) / n
    stat = dbar / np.sqrt(s / n)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))
    return dbar, stat, p

def block_bootstrap_ci(d, block=168, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(d); nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    means = np.empty(B)
    for b in range(B):
        idx = (starts[b][:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = d[idx].mean()
    return np.percentile(means, [2.5, 97.5])

pairs = [("tft", "xgboost"), ("tft", "regression"), ("xgboost", "regression")]
rows_b = []
for metric in ("MAE", "BandPinball"):
    for a, b in pairs:
        d = LOSS[a][metric] - LOSS[b][metric]
        dbar, stat, p = dm_test(d)
        lo, hi = block_bootstrap_ci(d)
        rows_b.append([metric, MODELS[a], MODELS[b], dbar, stat, p, lo, hi])

# Holm-Korrektur ueber alle sechs Vergleiche (Familie = beide Metriken)
order = sorted(range(len(rows_b)), key=lambda i: rows_b[i][5])
m = len(rows_b); running = 0.0
holm = [0.0] * m
for rank, i in enumerate(order):
    running = max(running, (m - rank) * rows_b[i][5])
    holm[i] = min(1.0, running)
for r, ph in zip(rows_b, holm):
    r.append(ph)
    print(f"{r[0]:11s} {r[1]:10s} - {r[2]:10s}: "
          f"Diff {r[3]:8.1f} MW  DM {r[4]:6.2f}  p {r[5]:.2e}  "
          f"p_Holm {ph:.2e}  Bootstrap-KI [{r[6]:8.1f}, {r[7]:8.1f}]")
pd.DataFrame(rows_b, columns=["Metrik", "A", "B", "Differenz", "DM", "p", "KI_lo", "KI_hi", "p_holm"]).to_csv(
    OUT / "signifikanz.csv", index=False)

print()
print("=" * 78)
print("C) MULTIVARIATE ROBUSTHEIT DER TREIBERANALYSE (OLS, standardisierte Betas)")
print("=" * 78)

comb = pd.read_csv(ROOT / "data" / "processed" / "combined_data.csv",
                   parse_dates=["timestamp"]).set_index("timestamp")
if comb.index.tz is not None:
    comb.index = comb.index.tz_convert("UTC").tz_localize(None)
wet = comb.loc[IDX, ["weather_radiation_forecast", "weather_wind_100m_forecast",
                     "weather_temperature_forecast", "is_holiday"]]
loc = IDX.tz_localize("UTC").tz_convert(ZoneInfo("Europe/Berlin"))
hour = np.asarray([t.hour for t in loc]); month = np.asarray([t.month for t in loc])
dow = np.asarray([t.dayofweek for t in loc])
season = np.select([np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8]),
                    np.isin(month, [9, 10, 11])], [1, 2, 3], default=0)
werktag = ((dow < 5) & (wet["is_holiday"].to_numpy() == 0)).astype(float)

def zscore(x): return (x - x.mean()) / x.std()

Xparts = {
    "Strahlung": zscore(wet["weather_radiation_forecast"].to_numpy()),
    "Wind": zscore(wet["weather_wind_100m_forecast"].to_numpy()),
    "Temperatur": zscore(wet["weather_temperature_forecast"].to_numpy()),
    "sin_h": np.sin(2 * np.pi * hour / 24), "cos_h": np.cos(2 * np.pi * hour / 24),
    "S_Fruehling": (season == 1).astype(float), "S_Sommer": (season == 2).astype(float),
    "S_Herbst": (season == 3).astype(float), "Werktag": werktag,
}
X = np.column_stack([np.ones(len(IDX))] + list(Xparts.values()))
names = ["const"] + list(Xparts.keys())
rows_c = []
for m, name in MODELS.items():
    df = DATA[m][0]
    w = (df["q90_cal80"] - df["q10_cal80"]).to_numpy()
    yz = zscore(w)
    beta, res, *_ = np.linalg.lstsq(X, yz, rcond=None)
    r2 = 1 - ((yz - X @ beta) ** 2).sum() / (yz ** 2).sum()
    wetter = {n: b for n, b in zip(names, beta) if n in ("Strahlung", "Wind", "Temperatur")}
    raw = {k: np.corrcoef(Xparts[k], zscore(w))[0, 1] for k in ("Strahlung", "Wind", "Temperatur")}
    print(f"{name:11s} R2={r2:.3f}  " + "  ".join(
        f"{k}: roh r={raw[k]:+.2f} / partiell beta={wetter[k]:+.2f}" for k in wetter))
    for k in wetter:
        rows_c.append((name, k, raw[k], wetter[k], r2))
pd.DataFrame(rows_c, columns=["Modell", "Treiber", "r_roh", "beta_partiell", "R2"]).to_csv(
    OUT / "treiber_multivariat.csv", index=False)

print()
print("=" * 78)
print("D) LAENDERSPEZIFISCHE FEIERTAGE 2025 (MAE, nur Werktags-Feiertage)")
print("=" * 78)

land_ft = ["2025-01-06", "2025-06-19", "2025-08-15", "2025-10-31"]
bund_ft = ["2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-05-29",
           "2025-06-09", "2025-10-03", "2025-12-25", "2025-12-26"]
dates = pd.Series(IDX.date.astype(str), index=IDX)
is_land = dates.isin(land_ft).to_numpy()
is_werk = (dow < 5) & ~dates.isin(bund_ft).to_numpy() & ~is_land
for m, name in MODELS.items():
    ae = LOSS[m]["MAE"]
    print(f"{name:11s} MAE Landes-Feiertage ({is_land.sum()} h): {ae[is_land].mean():7.0f} MW"
          f"  | uebrige Werktage ({is_werk.sum()} h): {ae[is_werk].mean():7.0f} MW"
          f"  | Verhaeltnis {ae[is_land].mean()/ae[is_werk].mean():4.2f}")

print()
print("=" * 78)
print("E) MONATSABDECKUNG ROH / ACI (Anhangstabelle)")
print("=" * 78)
rows_e = []
for m, name in MODELS.items():
    df = DATA[m][0]
    roh = ((df["q10"] <= df["actual"]) & (df["actual"] <= df["q90"])).groupby(IDX.month).mean()
    cal = ((df["q10_cal80"] <= df["actual"]) & (df["actual"] <= df["q90_cal80"])).groupby(IDX.month).mean()
    for mo in range(1, 13):
        rows_e.append((name, mo, 100 * roh[mo], 100 * cal[mo]))
    print(f"{name:11s} roh  " + " ".join(f"{100*roh[mo]:4.0f}" for mo in range(1, 13)))
    print(f"{'':11s} ACI  " + " ".join(f"{100*cal[mo]:4.0f}" for mo in range(1, 13)))
pd.DataFrame(rows_e, columns=["Modell", "Monat", "roh", "ACI"]).to_csv(
    OUT / "monatsabdeckung.csv", index=False)

print()
print("=" * 78)
print("F) OPERATIVE REFERENZ: SMARD-RESIDUALLASTPROGNOSE (Filter 715), Testjahr")
print("=" * 78)

cache = OUT / "smard_residuallast_prognose_2025.csv"
if cache.exists():
    ref = pd.read_csv(cache, parse_dates=["timestamp"]).set_index("timestamp")["forecast"]
else:
    def get(url):
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    idx = get("https://www.smard.de/app/chart_data/715/DE/index_hour.json")["timestamps"]
    lo = pd.Timestamp("2024-12-20", tz="UTC").value // 10**6
    hi = pd.Timestamp("2026-01-05", tz="UTC").value // 10**6
    series = {}
    for ts in [t for t in idx if lo <= t <= hi]:
        for t, v in get(f"https://www.smard.de/app/chart_data/715/DE/715_DE_hour_{ts}.json")["series"]:
            if v is not None:
                series[t] = v
    ref = pd.Series(series).sort_index()
    ref.index = pd.to_datetime(ref.index, unit="ms")
    ref = ref.loc["2025-01-01":"2025-12-31 23:00"]
    ref.rename("forecast").rename_axis("timestamp").to_csv(cache)
    ref = ref.rename("forecast")

ref = ref.reindex(IDX)
ok = ref.notna()
print(f"Abgedeckte Teststunden: {ok.sum()} von {len(IDX)}")
y = ACT[ok].to_numpy(); f = ref[ok].to_numpy()
e = y - f
mae = np.abs(e).mean(); rmse = np.sqrt((e ** 2).mean())
r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"SMARD-Referenz: MAE {mae:6.0f}  RMSE {rmse:6.0f}  R2 {r2:.3f}  "
      f"mittl. Abweichung {e.mean():+6.0f} MW")
for m, name in MODELS.items():
    ae = LOSS[m]["MAE"][ok.to_numpy()]
    d = ae - np.abs(e)
    dbar, stat, p = dm_test(d)
    print(f"  {name:11s} MAE {ae.mean():6.0f}  vs. Referenz: Diff {dbar:+7.1f} MW  "
          f"DM {stat:6.2f}  p {p:.2e}")
