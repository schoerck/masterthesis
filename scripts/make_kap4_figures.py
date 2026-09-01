"""
Erzeugt die Ergebnis-Abbildungen für Kapitel 4 der Masterarbeit und druckt
alle im Kapiteltext berichteten Kennzahlen (FF1–FF3) aus den finalen
Prognosedateien in output/final/.

Ausgabe: PDF-Vektorgrafiken in Tex/img/ (deutsche Beschriftung,
farbenblindfreundliche Okabe-Ito-Palette, Stil identisch zu den
EDA-Abbildungen aus make_eda_figures.py).

Aufruf:  python scripts/make_kap4_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# --- Pfade -----------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = CODE_DIR / "output" / "final"
DATA_CSV = CODE_DIR / "data" / "processed" / "combined_data.csv"
OUT_DIR = CODE_DIR.parent / "Tex" / "img"

# --- Okabe-Ito-Palette -----------------------------------------------------
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
GREY = "#999999"

MODELS = {
    "tft": ("TFT", BLUE),
    "xgboost": ("XGBoost", ORANGE),
    "regression": ("Regression", GREEN),
}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,
})


def de_thousands(x, _pos) -> str:
    return f"{x:,.0f}".replace(",", ".")


DE_MON = {1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun",
          7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez"}


def de_month(x, _pos) -> str:
    return DE_MON[mdates.num2date(x).month]


SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Frühling", 4: "Frühling",
          5: "Frühling", 6: "Sommer", 7: "Sommer", 8: "Sommer",
          9: "Herbst", 10: "Herbst", 11: "Herbst"}
SEASON_ORDER = ["Winter", "Frühling", "Sommer", "Herbst"]


def load_model(stem: str) -> pd.DataFrame:
    df = pd.read_csv(FINAL_DIR / f"{stem}_residual_load_day_ahead_final_forecast_calibrated.csv",
                     parse_dates=["timestamp"]).set_index("timestamp")
    df.index = df.index.tz_localize("UTC")
    df["width_raw"] = df["q90"] - df["q10"]
    df["width_cal"] = df["q90_cal80"] - df["q10_cal80"]
    df["cov_raw"] = ((df["actual"] >= df["q10"]) & (df["actual"] <= df["q90"])).astype(float)
    df["cov_cal"] = ((df["actual"] >= df["q10_cal80"]) & (df["actual"] <= df["q90_cal80"])).astype(float)
    df["err"] = df["actual"] - df["prediction"]
    local = df.index.tz_convert("Europe/Berlin")
    df["hour_local"] = local.hour
    df["month"] = local.month
    df["season"] = [SEASON[m] for m in local.month]
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {m: load_model(m) for m in MODELS}
    comb = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)
    if comb.index.tz is None:
        comb.index = comb.index.tz_localize("UTC")

    # Wetter/Kalender an die Teststunden des TFT anspielen
    tft = data["tft"]
    met = comb.loc[tft.index, ["weather_radiation_forecast", "weather_wind_100m_forecast",
                               "weather_temperature_forecast", "is_weekend", "is_holiday"]].copy()
    met["frei"] = ((met["is_weekend"] > 0) | (met["is_holiday"] > 0))

    print("=" * 70)
    print("FF1 — PUNKTGENAUIGKEIT (Test 2025)")
    for m, (name, _c) in MODELS.items():
        d = data[m]
        mae = d["err"].abs().mean()
        rmse = np.sqrt((d["err"] ** 2).mean())
        r2 = 1 - (d["err"] ** 2).sum() / ((d["actual"] - d["actual"].mean()) ** 2).sum()
        bias = d["err"].mean()
        print(f"{name:11s} MAE={mae:8.1f}  RMSE={rmse:8.1f}  R2={r2:.3f}  Bias(actual-pred)={bias:+7.1f}")
    mae_tft = data["tft"]["err"].abs().mean()
    mae_xgb = data["xgboost"]["err"].abs().mean()
    mae_reg = data["regression"]["err"].abs().mean()
    print(f"TFT vs XGB: {100*(mae_xgb-mae_tft)/mae_xgb:.1f} % niedriger | "
          f"XGB vs Reg: {100*(mae_reg-mae_xgb)/mae_reg:.1f} % niedriger")

    print("\nFehlerstruktur (Lokalzeit):")
    for m, (name, _c) in MODELS.items():
        d = data[m]
        h = d.groupby("hour_local")["err"].apply(lambda e: e.abs().mean())
        s = d.groupby("season")["err"].apply(lambda e: e.abs().mean()).reindex(SEASON_ORDER)
        hb = d.groupby("hour_local")["err"].mean()
        print(f"{name:11s} MAE/h: max {h.max():.0f} um {h.idxmax()} Uhr, min {h.min():.0f} um {h.idxmin()} Uhr | "
              f"Saison: {' '.join(f'{k}={v:.0f}' for k, v in s.items())}")
        print(f"{'':11s} Bias/h: extrem {hb.min():+.0f} um {hb.idxmin()} Uhr / {hb.max():+.0f} um {hb.idxmax()} Uhr")
    neg = tft[tft["actual"] < 0]
    print(f"Stunden actual<0: n={len(neg)}, TFT-MAE dort={neg['err'].abs().mean():.0f}")

    print("\n" + "=" * 70)
    print("FF2 — KALIBRIERUNG")
    for m, (name, _c) in MODELS.items():
        d = data[m]
        mon_raw = d.groupby("month")["cov_raw"].mean()
        mon_cal = d.groupby("month")["cov_cal"].mean()
        print(f"{name:11s} Cov roh={d['cov_raw'].mean():.3f} (Monate {mon_raw.min():.2f}–{mon_raw.max():.2f}, "
              f"min Monat {mon_raw.idxmin()}) -> ACI={d['cov_cal'].mean():.3f} "
              f"(Monate {mon_cal.min():.2f}–{mon_cal.max():.2f})")
        print(f"{'':11s} Breite roh={d['width_raw'].mean():.0f} -> ACI={d['width_cal'].mean():.0f} "
              f"(+{100*(d['width_cal'].mean()/d['width_raw'].mean()-1):.0f} %) | "
              f"c: Start={d['q_hat_80'].iloc[0]:.0f}, Ende={d['q_hat_80'].iloc[-1]:.0f}, "
              f"Min={d['q_hat_80'].min():.0f} ({d['q_hat_80'].idxmin():%d.%m.}), "
              f"Max={d['q_hat_80'].max():.0f} ({d['q_hat_80'].idxmax():%d.%m.})")

    print("\n" + "=" * 70)
    print("FF3 — UNSICHERHEITSTREIBER (Bandbreite kalibriert)")
    rad_pos_med = met.loc[met["weather_radiation_forecast"] > 0, "weather_radiation_forecast"].median()
    print(f"Anteil Stunden Strahlung=0: {(met['weather_radiation_forecast'] <= 0).mean():.2f}, "
          f"Median der positiven: {rad_pos_med:.0f}")
    wind_t = met["weather_wind_100m_forecast"].quantile([1/3, 2/3])
    temp_t = met["weather_temperature_forecast"].quantile([1/3, 2/3])
    print(f"Wind-Drittelgrenzen: {wind_t.iloc[0]:.1f} / {wind_t.iloc[1]:.1f} km/h | "
          f"Temp-Drittelgrenzen: {temp_t.iloc[0]:.1f} / {temp_t.iloc[1]:.1f} °C")

    def rad_group(v):
        if v <= 0:
            return "keine"
        return "niedrig" if v <= rad_pos_med else "hoch"

    def tercile_group(v, t):
        return "niedrig" if v <= t.iloc[0] else ("mittel" if v <= t.iloc[1] else "hoch")

    groups = pd.DataFrame(index=tft.index)
    groups["Strahlung"] = met["weather_radiation_forecast"].map(rad_group)
    groups["Wind"] = met["weather_wind_100m_forecast"].map(lambda v: tercile_group(v, wind_t))
    groups["Temperatur"] = met["weather_temperature_forecast"].map(lambda v: tercile_group(v, temp_t))
    groups["Tagtyp"] = np.where(met["frei"], "frei", "Werktag")

    for m, (name, _c) in MODELS.items():
        d = data[m]
        parts = []
        for gcol, order in [("Strahlung", ["keine", "niedrig", "hoch"]),
                            ("Wind", ["niedrig", "mittel", "hoch"]),
                            ("Temperatur", ["niedrig", "mittel", "hoch"])]:
            g = d["width_cal"].groupby(groups[gcol]).mean().reindex(order)
            parts.append(f"{gcol}: " + " ".join(f"{k}={v:.0f}" for k, v in g.items()))
        seas = d["width_cal"].groupby(d["season"]).mean().reindex(SEASON_ORDER)
        tag = d["width_cal"].groupby(groups["Tagtyp"]).mean()
        print(f"{name}:")
        print(f"   Saison: {' '.join(f'{k}={v:.0f}' for k, v in seas.items())} | "
              f"Tagtyp: {' '.join(f'{k}={v:.0f}' for k, v in tag.items())}")
        print("   " + " | ".join(parts))
        r_rad = d["width_cal"].corr(met["weather_radiation_forecast"])
        r_wind = d["width_cal"].corr(met["weather_wind_100m_forecast"])
        r_temp = d["width_cal"].corr(met["weather_temperature_forecast"])
        print(f"   Korrelation Breite~Strahlung r={r_rad:+.2f}, ~Wind r={r_wind:+.2f}, ~Temp r={r_temp:+.2f}")
    h_width = tft["width_cal"].groupby(tft["hour_local"]).mean()
    print(f"TFT Breite/h: max {h_width.max():.0f} um {h_width.idxmax()} Uhr, "
          f"min {h_width.min():.0f} um {h_width.idxmin()} Uhr")

    print("\n" + "=" * 70)
    print("PERSISTENZ-REFERENZ (Vortageswert, Test 2025)")
    rl = comb["residual_load"]
    pers = rl.shift(24).loc[tft.index]
    perr = tft["actual"] - pers
    p_mae = perr.abs().mean()
    p_rmse = np.sqrt((perr ** 2).mean())
    p_r2 = 1 - (perr ** 2).sum() / ((tft["actual"] - tft["actual"].mean()) ** 2).sum()
    print(f"Persistenz  MAE={p_mae:8.1f}  RMSE={p_rmse:8.1f}  R2={p_r2:.3f} | "
          f"TFT-MAE ist {100*(1-mae_tft/p_mae):.0f} % niedriger")

    print("\n" + "=" * 70)
    print("ASYMMETRIE DER ROHEN PROGNOSEVERTEILUNG (Halbbreiten in MW)")
    for m, (name, _c) in MODELS.items():
        d = data[m]
        ol = (d["q50"] - d["q10"]).mean()
        ou = (d["q90"] - d["q50"]).mean()
        il = (d["q50"] - d["q25"]).mean()
        iu = (d["q75"] - d["q50"]).mean()
        print(f"{name:11s} außen unten={ol:6.0f} oben={ou:6.0f} (unten/oben={ol/ou:.2f}) | "
              f"innen unten={il:6.0f} oben={iu:6.0f} (unten/oben={il/iu:.2f})")
    t_ol = (tft["q50"] - tft["q10"]).groupby(tft["hour_local"]).mean()
    t_ou = (tft["q90"] - tft["q50"]).groupby(tft["hour_local"]).mean()
    ratio = t_ol / t_ou
    print(f"TFT Tagesgang unten/oben: max {ratio.max():.2f} um {ratio.idxmax()} Uhr "
          f"(unten={t_ol[ratio.idxmax()]:.0f}, oben={t_ou[ratio.idxmax()]:.0f}), "
          f"min {ratio.min():.2f} um {ratio.idxmin()} Uhr")

    # ---------------- Abbildungen ----------------
    fmt = FuncFormatter(de_thousands)

    # A0) Asymmetrie: Tagesgang der Halbbreiten des rohen TFT-Bands
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.plot(t_ol, color=VERMILLION, label="untere Halbbreite ($q_{50}-q_{10}$)")
    ax.plot(t_ou, color=BLUE, label="obere Halbbreite ($q_{90}-q_{50}$)")
    ax.set_xlabel("Stunde (Lokalzeit)")
    ax.set_ylabel("Halbbreite (MW)")
    ax.set_xticks(range(0, 24, 4))
    ax.yaxis.set_major_formatter(fmt)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "erg_asymmetrie.pdf")
    plt.close(fig)

    # A) Fehlerstruktur: MAE je Tagesstunde + je Monat (gemeinsame y-Achse)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.9), sharey=True)
    for m, (name, c) in MODELS.items():
        d = data[m]
        ax1.plot(d.groupby("hour_local")["err"].apply(lambda e: e.abs().mean()), color=c, label=name)
        ax2.plot(d.groupby("month")["err"].apply(lambda e: e.abs().mean()), color=c, label=name)
    ax1.set_xlabel("Stunde (Lokalzeit)")
    ax1.set_ylabel("MAE (MW)")
    ax1.set_xticks(range(0, 24, 4))
    ax2.set_xlabel("Monat")
    ax2.set_xticks(range(1, 13, 2))
    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(fmt)
        ax.set_ylim(bottom=0)
    ax1.legend(frameon=False)
    fig.savefig(OUT_DIR / "erg_fehlerstruktur.pdf")
    plt.close(fig)

    # A2) Streudiagramm Prognose vs. Messwert je Modell
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharex=True, sharey=True)
    lims = (-12000, 75000)
    for ax, (m, (name, c)) in zip(axes, MODELS.items()):
        d = data[m]
        ax.plot(lims, lims, color="black", ls="--", lw=0.8)
        ax.scatter(d["actual"], d["prediction"], s=2, alpha=0.12, color=c, rasterized=True)
        ax.set_title(name)
        ax.set_xlabel("Messwert (MW)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.xaxis.set_major_formatter(fmt)
        ax.set_xticks([0, 30000, 60000])
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("Prognose (MW)")
    axes[0].yaxis.set_major_formatter(fmt)
    fig.savefig(OUT_DIR / "erg_scatter.pdf", dpi=200)
    plt.close(fig)

    # A3) Kalibrierungsdiagramm: beobachteter Anteil unter jedem rohen Quantil
    taus = [0.10, 0.25, 0.50, 0.75, 0.90]
    qcols = ["q10", "q25", "q50", "q75", "q90"]
    print("\nKALIBRIERUNGSDIAGRAMM (beobachteter Anteil unter rohem Quantil):")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=0.8)
    for m, (name, c) in MODELS.items():
        d = data[m]
        obs = [(d["actual"] <= d[qc]).mean() for qc in qcols]
        ax.plot(taus, obs, marker="o", ms=4, color=c, label=name)
        print(f"{name:11s} " + "  ".join(f"q{int(t*100)}: {o:.3f}" for t, o in zip(taus, obs)))
    ax.set_xlabel("Nominalniveau des Quantils")
    ax.set_ylabel("beobachteter Anteil darunter")
    ax.set_xticks(taus)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT_DIR / "erg_reliability.pdf")
    plt.close(fig)

    # Negativstunden: kalibrierte Bandbreite im Vergleich
    print("\nNEGATIVSTUNDEN (actual < 0): kalibrierte Bandbreite")
    for m, (name, _c) in MODELS.items():
        d = data[m]
        neg = d[d["actual"] < 0]
        print(f"{name:11s} gesamt={d['width_cal'].mean():.0f}  negativ={neg['width_cal'].mean():.0f} "
              f"(Faktor {neg['width_cal'].mean()/d['width_cal'].mean():.2f}, n={len(neg)})")

    # B) Rollierende 30-Tage-Abdeckung roh vs. kalibriert
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.9), sharey=True)
    for m, (name, c) in MODELS.items():
        d = data[m]
        ax1.plot(d["cov_raw"].rolling(24 * 30).mean(), color=c, label=name, lw=1.1)
        ax2.plot(d["cov_cal"].rolling(24 * 30).mean(), color=c, label=name, lw=1.1)
    for ax, titel in [(ax1, "ohne Kalibrierung"), (ax2, "mit ACI")]:
        ax.axhline(0.8, color="black", ls="--", lw=0.9)
        ax.set_title(titel)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(FuncFormatter(de_month))
    ax1.set_ylabel("Abdeckung (rollierend, 30 Tage)")
    ax1.set_ylim(0.3, 1.0)
    ax1.legend(frameon=False, loc="lower left")
    fig.savefig(OUT_DIR / "erg_coverage.pdf")
    plt.close(fig)

    # C) Zuschlag c_t über das Testjahr
    fig, ax = plt.subplots(figsize=(8.4, 2.7))
    for m, (name, c) in MODELS.items():
        ax.plot(data[m]["q_hat_80"], color=c, label=name, lw=1.1)
    ax.set_ylabel("Zuschlag $c_t$ (MW)")
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(FuncFormatter(de_month))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.savefig(OUT_DIR / "erg_ct_verlauf.pdf")
    plt.close(fig)

    # D) Beispielwochen mit kalibriertem Band (TFT): Woche des Jahresmax/-min
    wk_max = tft["actual"].idxmax().normalize() - pd.Timedelta(days=3)
    wk_min = tft["actual"].idxmin().normalize() - pd.Timedelta(days=3)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.6), sharex=False)
    for ax, start, titel in [(axes[0], wk_max, "Um das Jahresmaximum"),
                             (axes[1], wk_min, "Um das Jahresminimum")]:
        w = tft.loc[start:start + pd.Timedelta(days=7)]
        ax.fill_between(w.index, w["q10_cal80"], w["q90_cal80"], color=SKY, alpha=0.45,
                        label="kalibriertes 80-%-Band", linewidth=0)
        ax.plot(w.index, w["q10"], color=VERMILLION, ls="--", lw=1.0, label="rohes 80-%-Band")
        ax.plot(w.index, w["q90"], color=VERMILLION, ls="--", lw=1.0)
        ax.plot(w.index, w["prediction"], color=BLUE, label="Median-Prognose")
        ax.plot(w.index, w["actual"], color="black", lw=1.0, label="Messwert")
        ax.axhline(0, color=GREY, lw=0.7)
        ax.set_title(titel)
        ax.set_ylabel("Residuallast (MW)")
        ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))
    axes[0].legend(frameon=False, loc="lower left", ncol=4, fontsize=7.2)
    fig.savefig(OUT_DIR / "erg_bandwochen.pdf")
    plt.close(fig)
    print(f"\nBeispielwochen: Max-Woche ab {wk_max:%d.%m.%Y}, Min-Woche ab {wk_min:%d.%m.%Y}")

    # D2) Jahresansicht je Modell: gleitendes Wochenmittel Band (oben) + Fehler (unten)
    def week_smooth(df):
        return df.resample("D").mean().rolling(7, center=True, min_periods=1).mean()

    wk_all = {m: week_smooth(data[m][["actual", "prediction", "q10_cal80", "q90_cal80"]])
              for m in MODELS}
    day_all = {m: data[m][["actual", "prediction"]].resample("D").mean() for m in MODELS}
    err_wk = {m: day_all[m]["actual"] - day_all[m]["prediction"] for m in MODELS}
    band_lo = min(w["q10_cal80"].min() for w in wk_all.values())
    band_hi = max(w["q90_cal80"].max() for w in wk_all.values())
    err_lim = max(e.abs().max() for e in err_wk.values()) * 1.1
    print("\nJAHRESANSICHT (Tagesmittel-Fehler = Messwert - Prognose):")
    for m, (name, _c) in MODELS.items():
        e = err_wk[m]
        print(f"{name:11s} min {e.min():+.0f} ({e.idxmin():%d.%m.}), max {e.max():+.0f} ({e.idxmax():%d.%m.}) | "
              f"Dez-Mittel {e.loc['2025-12'].mean():+.0f}")

    wkraw_all = {m: week_smooth(data[m][["q10", "q90"]]) for m in MODELS}
    for m, (name, c) in MODELS.items():
        w = wk_all[m]
        wr = wkraw_all[m]
        fig, (ax, axe) = plt.subplots(2, 1, figsize=(8.4, 3.6), sharex=True,
                                      gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.1})
        ax.fill_between(w.index, w["q10_cal80"], w["q90_cal80"], color=c, alpha=0.30,
                        linewidth=0, label="kalibriertes 80-%-Band")
        ax.plot(wr.index, wr["q10"], color=VERMILLION, ls="--", lw=0.7, label="rohes 80-%-Band")
        ax.plot(wr.index, wr["q90"], color=VERMILLION, ls="--", lw=0.7)
        ax.plot(w.index, w["prediction"], color=c, ls="--", lw=0.9, label="Median-Prognose")
        ax.plot(w.index, w["actual"], color="black", lw=0.8, label="Messwert")
        ax.set_ylim(band_lo * 0.9, band_hi * 1.04)
        ax.set_ylabel("Residuallast (MW)")
        ax.yaxis.set_major_formatter(fmt)
        ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=7.0)
        ax.text(0.012, 0.955, name, transform=ax.transAxes, ha="left", va="top",
                fontsize=9, fontweight="bold", color=c)
        axe.axhline(0, color=GREY, lw=0.7)
        axe.bar(err_wk[m].index, err_wk[m], width=1.0, color=c, alpha=0.75, linewidth=0)
        axe.set_ylim(-err_lim, err_lim)
        axe.set_ylabel("Fehler (MW)", fontsize=8)
        axe.yaxis.set_major_formatter(fmt)
        for a in (ax, axe):
            a.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            a.xaxis.set_major_formatter(FuncFormatter(de_month))
        plt.setp(ax.get_xticklabels(), visible=False)
        fig.savefig(OUT_DIR / f"erg_jahresband_{m}.pdf", bbox_inches="tight")
        plt.close(fig)

    # E0) Punktwolken: Bandbreite vs. Strahlungsvorhersage je Modell
    rad = met["weather_radiation_forecast"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharex=True, sharey=True)
    for ax, (m, (name, c)) in zip(axes, MODELS.items()):
        d = data[m]
        r = d["width_cal"].corr(rad)
        ax.scatter(rad, d["width_cal"], s=2, alpha=0.10, color=c, rasterized=True)
        ax.set_title(f"{name} ($r = {r:.2f}$)".replace(".", "{,}"))
        ax.set_xlabel("Strahlungsvorhersage (W/m²)")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Bandbreite (MW)")
    axes[0].yaxis.set_major_formatter(fmt)
    fig.savefig(OUT_DIR / "erg_treiber_scatter.pdf", dpi=200)
    plt.close(fig)

    # E) Treibergruppen: mittlere kalibrierte Bandbreite, drei Modelle
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    kal_labels = SEASON_ORDER + ["Werktag", "frei"]
    wet_labels = [("Strahlung", "keine"), ("Strahlung", "niedrig"), ("Strahlung", "hoch"),
                  ("Wind", "niedrig"), ("Wind", "mittel"), ("Wind", "hoch")]
    x1 = np.arange(len(kal_labels))
    x2 = np.arange(len(wet_labels))
    bw = 0.26
    for i, (m, (name, c)) in enumerate(MODELS.items()):
        d = data[m]
        seas = d["width_cal"].groupby(d["season"]).mean().reindex(SEASON_ORDER)
        tag = d["width_cal"].groupby(groups["Tagtyp"]).mean().reindex(["Werktag", "frei"])
        v1 = list(seas.values) + list(tag.values)
        v2 = [d["width_cal"].groupby(groups[g]).mean()[lvl] for g, lvl in wet_labels]
        ax1.bar(x1 + (i - 1) * bw, v1, bw, color=c, label=name)
        ax2.bar(x2 + (i - 1) * bw, v2, bw, color=c, label=name)
    ax1.set_xticks(x1, kal_labels, rotation=30, ha="right")
    ax2.set_xticks(x2, [f"{g}\n{lvl}" for g, lvl in wet_labels])
    ax1.set_ylabel("mittlere Bandbreite (MW)")
    ax1.yaxis.set_major_formatter(fmt)
    ax1.legend(frameon=False)
    fig.savefig(OUT_DIR / "erg_treiber.pdf")
    plt.close(fig)

    print("Abbildungen gespeichert in", OUT_DIR)


if __name__ == "__main__":
    main()
