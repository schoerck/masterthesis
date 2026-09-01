"""
Erzeugt den Vergleich TFT gegen SMARD-Residuallastprognose (Kapitel 4.1).

Oberes Feld: Messwert, TFT-Median und SMARD-Referenz im gleitenden
Wochenmittel ueber das Testjahr 2025.
Unteres Feld: mittlerer absoluter Fehler je Kalendermonat als Balkenpaar.

Datengrundlage: output/final/*.csv und der SMARD-Cache aus
scripts/make_gutachten_analysen.py.

Aufruf:  .venv/bin/python scripts/make_smard_vergleich.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parent / "Tex" / "img"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#999999"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

de_thousands = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
MON = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
       "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def main() -> None:
    tft = pd.read_csv(ROOT / "output" / "final" /
                      "tft_residual_load_day_ahead_final_forecast_calibrated.csv",
                      parse_dates=["timestamp"]).set_index("timestamp")
    ref = pd.read_csv(ROOT / "output" / "analysen" /
                      "smard_residuallast_prognose_2025.csv",
                      parse_dates=["timestamp"]).set_index("timestamp")["forecast"]
    ref = ref.reindex(tft.index)

    df = pd.DataFrame({"Messwert": tft["actual"], "TFT": tft["prediction"],
                       "SMARD": ref})
    wk = df.resample("D").mean().rolling(7, center=True, min_periods=1).mean()

    mae = pd.DataFrame({
        "TFT": (df["Messwert"] - df["TFT"]).abs().groupby(df.index.month).mean(),
        "SMARD": (df["Messwert"] - df["SMARD"]).abs().groupby(df.index.month).mean(),
    })
    print("Monats-MAE (MW):")
    print(mae.round(0).to_string())
    print("Verhaeltnis SMARD/TFT je Monat:",
          (mae["SMARD"] / mae["TFT"]).round(2).to_list())

    fig, (ax, axm) = plt.subplots(
        2, 1, figsize=(8.4, 3.6),
        gridspec_kw={"height_ratios": [2.4, 1.2], "hspace": 0.42})

    ax.plot(wk.index, wk["Messwert"], color="black", lw=0.9, label="Messwert")
    ax.plot(wk.index, wk["TFT"], color=BLUE, ls="--", lw=1.0,
            label="TFT (Median-Prognose)")
    ax.plot(wk.index, wk["SMARD"], color=VERMILLION, lw=1.0,
            label="SMARD-Referenz")
    ax.set_ylabel("Residuallast (MW)")
    ax.yaxis.set_major_formatter(de_thousands)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=7.4)
    ax.set_xlim(wk.index[0], wk.index[-1])
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: MON[plt.matplotlib.dates.num2date(v).month - 1]))

    x = np.arange(1, 13)
    axm.bar(x - 0.19, mae["TFT"], width=0.38, color=BLUE, label="TFT")
    axm.bar(x + 0.19, mae["SMARD"], width=0.38, color=VERMILLION,
            label="SMARD-Referenz")
    axm.set_ylabel("MAE (MW)", fontsize=8)
    axm.yaxis.set_major_formatter(de_thousands)
    axm.set_xticks(x)
    axm.set_xticklabels(MON)
    axm.legend(frameon=False, ncol=2, loc="upper right", fontsize=7.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "erg_smard_vergleich.pdf", bbox_inches="tight")
    plt.close(fig)
    print("gespeichert:", OUT_DIR / "erg_smard_vergleich.pdf")


if __name__ == "__main__":
    main()
