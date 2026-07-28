"""
Adaptive Conformal Inference (ACI) — Gibbs & Candès (2021)

Eine *online*-adaptive Variante der Conformal Prediction. Im Gegensatz
zum klassischen Split-CQR (Romano et al. 2019), das einen festen
Kalibrierungs-Offset q̂ aus einer Validation-Periode berechnet und blind
auf die gesamte Testperiode anwendet, passt ACI q̂ **bei jedem neuen
Datenpunkt an** — basierend auf der beobachteten Treffer-/Fehlrate.

Warum nicht klassisches CQR?
─────────────────────────────
CQR garantiert die Ziel-Coverage nur unter der Annahme der **Austauschbarkeit**
von Validation und Test (gleiche Verteilung). Bei Zeitreihen mit Verteilungs-
verschiebungen (z. B. Marktstruktur Val 2024 ≠ Test 2025) versagt diese
Annahme — die effektive Coverage weicht systematisch vom Ziel ab.

ACI braucht diese Annahme nicht. Es garantiert die Coverage **langfristig**
über jede Sequenz, indem es q̂ als Steuergröße eines geschlossenen
Regelkreises behandelt.

Mathematik (für ein (1−α)-Band, hier α = 0,2 ↔ 80 %-Band)
─────────────────────────────────────────────────────────
Für jede Test-Stunde t = 1, 2, …, T:

    1. Vorhersage-Intervall mit aktuellem q̂_t:
            I_t = [q_lower(x_t) − q̂_t,  q_upper(x_t) + q̂_t]

    2. Beobachte y_t. Indikator: 1 wenn y_t außerhalb I_t, sonst 0:
            err_t = 1{y_t ∉ I_t}

    3. Update q̂ proportional zur Abweichung vom Soll-Fehleranteil α:
            q̂_(t+1) = q̂_t + γ · (err_t − α)

Im langfristigen Erwartungswert muss (err − α) im Mittel null sein, d.h.
die empirische Miss-Rate konvergiert gegen α. Coverage = 1 − α langfristig
garantiert — ohne Stationaritäts-Annahme.

Wahl der Lernrate γ:
    Wir leiten γ adaptiv aus der Skala der Konformitätsscores ab:
        γ = 0.005 · std(scores_val)
    Das skaliert die Schrittweite mit der natürlichen Variabilität der
    Vorhersagefehler und ist über Modelle hinweg vergleichbar.

Startwert q̂_0:
    Wir initialisieren q̂_0 mit dem klassischen Split-CQR-Wert aus der
    Val-Periode. Damit startet ACI nicht bei null, sondern bei einer
    bereits informierten Schätzung — schnelle Konvergenz garantiert.

Eigenschaften
─────────────
• Funktioniert auch bei Verteilungsverschiebungen (kein Train/Val/Test-
  Austauschbarkeits-Bedarf).
• Nutzt y_t nur **nachdem** das Intervall I_t schon abgegeben wurde
  (kein Datenleck — entspricht realistischem Operieren).
• Liefert als Nebenprodukt die Zeitreihe q̂_t — diagnostisch wertvoll
  (zeigt, wann das Modell besonders übersicher oder unterkonservativ war).

Referenzen
──────────
    Gibbs, I., & Candès, E. J. (2021).
    "Adaptive Conformal Inference Under Distribution Shift."
    NeurIPS 2021. https://arxiv.org/abs/2106.00170

    Romano, Y., Patterson, E., & Candès, E. J. (2019).
    "Conformalized Quantile Regression."
    NeurIPS 2019. (Basis für die q̂_0-Initialisierung.)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Pure-Funktionen — Kernlogik
# ════════════════════════════════════════════════════════════════════


def compute_conformity_scores(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> np.ndarray:
    """
    Berechnet CQR-Konformitätsscores für ein Quantil-Paar.

    E_i = max(q_lower(x_i) − y_i, y_i − q_upper(x_i))

    Interpretation:
      E_i > 0  →  y_i liegt AUSSERHALB des Bands
      E_i ≤ 0  →  y_i liegt INNERHALB; |E_i| = Abstand zur näheren Bandgrenze
    """
    if not (len(y_true) == len(q_lower) == len(q_upper)):
        raise ValueError(
            f"Längen müssen übereinstimmen: y_true={len(y_true)}, "
            f"q_lower={len(q_lower)}, q_upper={len(q_upper)}"
        )

    y_true = np.asarray(y_true, dtype=float)
    q_lower = np.asarray(q_lower, dtype=float)
    q_upper = np.asarray(q_upper, dtype=float)

    return np.maximum(q_lower - y_true, y_true - q_upper)


def _initial_qhat_from_val(
    scores_val: np.ndarray,
    coverage_level: float = 0.8,
) -> float:
    """
    Startwert für ACI: Split-CQR-Quantil aus Validation-Scores.

    Verwendet die Finite-Sample-korrigierte Quantil-Formel aus Romano et al.
    2019 (Theorem 1): q̂₀ = ⌈(n+1)(1−α)⌉/n-Quantil der Val-Scores.
    """
    alpha = 1.0 - coverage_level
    n = len(scores_val)
    quantile_index = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat_0 = float(np.quantile(scores_val, quantile_index, method="higher"))
    return q_hat_0


def _suggest_learning_rate(scores_val: np.ndarray) -> float:
    """
    Heuristik für die Lernrate γ: 0.5 % der Standardabweichung der
    Val-Konformitätsscores. Skaliert γ mit der natürlichen Variabilität
    der Vorhersagefehler — modellübergreifend vergleichbar.
    """
    return float(0.005 * np.std(scores_val))


def adaptive_conformal_inference(
    y_true_test: np.ndarray,
    q_lower_test: np.ndarray,
    q_upper_test: np.ndarray,
    q_hat_init: float,
    target_coverage: float = 0.8,
    gamma: float | None = None,
    scores_val_for_gamma: np.ndarray | None = None,
) -> dict:
    """
    Wendet Adaptive Conformal Inference auf eine Test-Sequenz an.

    Iteriert online über alle Test-Stunden, gibt für jede Stunde ein
    kalibriertes Intervall ab und aktualisiert q̂ nach jeder Beobachtung.

    Parameters
    ----------
    y_true_test : np.ndarray, shape (T,)
        Beobachtete Ist-Werte auf der Test-Sequenz.
    q_lower_test, q_upper_test : np.ndarray, shape (T,)
        Vorhergesagte Modell-Quantile auf der Test-Sequenz.
    q_hat_init : float
        Startwert für den Kalibrierungs-Offset (typischerweise aus
        :func:`_initial_qhat_from_val`).
    target_coverage : float, default 0.8
        Gewünschtes Konfidenzniveau (1 − α).
    gamma : float, optional
        Lernrate für das Update. Falls None, wird sie aus
        `scores_val_for_gamma` heuristisch hergeleitet.
    scores_val_for_gamma : np.ndarray, optional
        Val-Konformitätsscores zur Bestimmung von γ. Nur nötig wenn
        gamma=None.

    Returns
    -------
    dict mit Keys:
        q_lower_calibrated : np.ndarray, shape (T,)
            Adaptiv kalibrierte untere Quantile.
        q_upper_calibrated : np.ndarray, shape (T,)
            Adaptiv kalibrierte obere Quantile.
        q_hat_trajectory : np.ndarray, shape (T,)
            Zeitreihe der adaptiv aktualisierten q̂-Werte.
        gamma : float
            Verwendete Lernrate.
        q_hat_init : float
            Verwendeter Startwert.
    """
    if gamma is None:
        if scores_val_for_gamma is None:
            raise ValueError(
                "Entweder gamma direkt oder scores_val_for_gamma übergeben."
            )
        gamma = _suggest_learning_rate(scores_val_for_gamma)

    y_true_test = np.asarray(y_true_test, dtype=float)
    q_lower_test = np.asarray(q_lower_test, dtype=float)
    q_upper_test = np.asarray(q_upper_test, dtype=float)

    if not (len(y_true_test) == len(q_lower_test) == len(q_upper_test)):
        raise ValueError("Test-Arrays müssen gleiche Länge haben.")

    alpha = 1.0 - target_coverage
    T = len(y_true_test)

    q_hat_trajectory = np.empty(T)
    q_lower_cal = np.empty(T)
    q_upper_cal = np.empty(T)

    q_hat = float(q_hat_init)
    for t in range(T):
        # 1. Aktuelles Intervall (mit aktuellem q̂)
        q_hat_trajectory[t] = q_hat
        q_lower_cal[t] = q_lower_test[t] - q_hat
        q_upper_cal[t] = q_upper_test[t] + q_hat

        # 2. Beobachte Treffer / Fehler
        outside = (y_true_test[t] < q_lower_cal[t]) or (
            y_true_test[t] > q_upper_cal[t]
        )
        err_t = 1.0 if outside else 0.0

        # 3. q̂-Update für nächste Stunde
        q_hat = q_hat + gamma * (err_t - alpha)

    logger.info(
        "ACI: T=%d, γ=%.2f, q̂₀=%.2f, q̂_T=%.2f, "
        "q̂-Range [%.2f, %.2f]",
        T, gamma, q_hat_init, q_hat,
        q_hat_trajectory.min(), q_hat_trajectory.max(),
    )

    return {
        "q_lower_calibrated": q_lower_cal,
        "q_upper_calibrated": q_upper_cal,
        "q_hat_trajectory": q_hat_trajectory,
        "gamma": gamma,
        "q_hat_init": q_hat_init,
    }


def evaluate_coverage(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> dict:
    """
    Bewertet ein Vorhersage-Intervall: Coverage und mittlere Breite.
    """
    y_true = np.asarray(y_true, dtype=float)
    q_lower = np.asarray(q_lower, dtype=float)
    q_upper = np.asarray(q_upper, dtype=float)

    inside = (y_true >= q_lower) & (y_true <= q_upper)
    widths = q_upper - q_lower
    return {
        "coverage": float(inside.mean()),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "n": int(len(y_true)),
    }


# ════════════════════════════════════════════════════════════════════
#  High-Level-API — Kalibrierung eines kompletten Forecast-DataFrames
# ════════════════════════════════════════════════════════════════════


def calibrate_forecast_adaptive(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantile_pairs: dict[str, tuple[str, str, float]] | None = None,
    gamma: float | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Adaptive-CQR-Kalibrierung eines Test-Forecast-DataFrames anhand eines
    Validation-Forecast-DataFrames.

    Workflow:
      1. Konformitätsscores aus Val berechnen — für Startwert q̂₀ und für
         Lernraten-Schätzung γ.
      2. ACI auf Test-Sequenz iterieren: pro Stunde Intervall + Update.
      3. Vorher/Nachher-Coverage-Vergleich auf Test als Report.

    Parameters
    ----------
    val_df : pd.DataFrame
        Forecast auf Validation-Periode. Erwartet Spalten 'actual' plus
        die in quantile_pairs referenzierten Quantil-Spalten.
    test_df : pd.DataFrame
        Forecast auf Testperiode. Gleiche Spaltenstruktur. Muss zeitlich
        SORTIERT sein — ACI iteriert sequenziell.
    quantile_pairs : dict, optional
        Mapping name -> (q_lower_col, q_upper_col, target_coverage).
        Default: {"80": ("q10", "q90", 0.8)}.
    gamma : float, optional
        Lernrate. None → automatisch aus Val-Score-Standardabweichung.

    Returns
    -------
    test_df_calibrated : pd.DataFrame
        Test-DataFrame mit ergänzten Spalten je Quantil-Paar:
          - {q_lower}_cal{name}, {q_upper}_cal{name} — kalibrierte Quantile
          - q_hat_{name} — Zeitreihe der adaptiven q̂-Werte
    calibration_report : dict
        Pro Quantil-Paar: γ, q̂-Verlaufsstatistik, Cov vorher/nachher.
    """
    if quantile_pairs is None:
        quantile_pairs = {"80": ("q10", "q90", 0.8)}

    needed_cols = {"actual"}
    for q_low, q_up, _ in quantile_pairs.values():
        needed_cols.update({q_low, q_up})
    missing_val = needed_cols - set(val_df.columns)
    missing_test = needed_cols - set(test_df.columns)
    if missing_val:
        raise ValueError(f"val_df fehlen Spalten: {missing_val}")
    if missing_test:
        raise ValueError(f"test_df fehlen Spalten: {missing_test}")

    # Test-DF zeitlich sortieren — ACI iteriert sequenziell.
    test_df_sorted = test_df.sort_values("timestamp").reset_index(drop=True)
    test_df_out = test_df_sorted.copy()

    report: dict[str, dict] = {}

    for name, (q_low_col, q_up_col, target_coverage) in quantile_pairs.items():
        # 1. Val-Konformitätsscores — für Startwert und γ-Schätzung
        scores_val = compute_conformity_scores(
            y_true=val_df["actual"].to_numpy(),
            q_lower=val_df[q_low_col].to_numpy(),
            q_upper=val_df[q_up_col].to_numpy(),
        )
        q_hat_init = _initial_qhat_from_val(scores_val, target_coverage)

        # 2. ACI auf Test anwenden
        result = adaptive_conformal_inference(
            y_true_test=test_df_sorted["actual"].to_numpy(),
            q_lower_test=test_df_sorted[q_low_col].to_numpy(),
            q_upper_test=test_df_sorted[q_up_col].to_numpy(),
            q_hat_init=q_hat_init,
            target_coverage=target_coverage,
            gamma=gamma,
            scores_val_for_gamma=scores_val if gamma is None else None,
        )

        # 3. Kalibrierte Bänder + q̂-Verlauf in Output-DF
        out_low = f"{q_low_col}_cal{name}"
        out_up = f"{q_up_col}_cal{name}"
        out_qhat = f"q_hat_{name}"
        test_df_out[out_low] = result["q_lower_calibrated"]
        test_df_out[out_up] = result["q_upper_calibrated"]
        test_df_out[out_qhat] = result["q_hat_trajectory"]

        # 4. Coverage-Reports
        val_cov = evaluate_coverage(
            val_df["actual"].to_numpy(),
            val_df[q_low_col].to_numpy(),
            val_df[q_up_col].to_numpy(),
        )
        test_cov_before = evaluate_coverage(
            test_df_sorted["actual"].to_numpy(),
            test_df_sorted[q_low_col].to_numpy(),
            test_df_sorted[q_up_col].to_numpy(),
        )
        test_cov_after = evaluate_coverage(
            test_df_sorted["actual"].to_numpy(),
            result["q_lower_calibrated"],
            result["q_upper_calibrated"],
        )

        q_hat_traj = result["q_hat_trajectory"]
        report[name] = {
            "target_coverage": target_coverage,
            "gamma": result["gamma"],
            "q_hat_init": result["q_hat_init"],
            "q_hat_min":    float(q_hat_traj.min()),
            "q_hat_max":    float(q_hat_traj.max()),
            "q_hat_mean":   float(q_hat_traj.mean()),
            "q_hat_median": float(np.median(q_hat_traj)),
            "val_coverage_uncal":        val_cov["coverage"],
            "val_width_uncal":           val_cov["mean_width"],
            "test_coverage_uncal":       test_cov_before["coverage"],
            "test_width_uncal":          test_cov_before["mean_width"],
            "test_coverage_calibrated":  test_cov_after["coverage"],
            "test_width_calibrated_mean":   test_cov_after["mean_width"],
            "test_width_calibrated_median": test_cov_after["median_width"],
            "n_val":  val_cov["n"],
            "n_test": test_cov_before["n"],
        }

        logger.info(
            "[%s%%] γ=%.2f, q̂₀=%.0f, q̂-Range [%.0f, %.0f] | "
            "Test-Cov vorher=%.3f → nachher=%.3f | "
            "Test-Breite Ø %.0f → %.0f",
            name, result["gamma"], result["q_hat_init"],
            q_hat_traj.min(), q_hat_traj.max(),
            test_cov_before["coverage"], test_cov_after["coverage"],
            test_cov_before["mean_width"], test_cov_after["mean_width"],
        )

    return test_df_out, report


if __name__ == "__main__":
    # Mini-Selbsttest: Modell mit Verteilungsverschiebung Val→Test
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(42)

    n_val, n_test = 1000, 2000

    # Val: niedrige Streuung, Modell hat sich an enge Bänder gewöhnt
    y_val = rng.normal(0, 1, n_val)
    q10_val = np.full(n_val, -0.8)
    q90_val = np.full(n_val, 0.8)

    # Test: höhere Streuung (Verteilungsshift!), aber Modell behält die
    # engen Bänder bei. Klassisches CQR würde hier struggeln, ACI passt sich an.
    y_test = rng.normal(0, 1.6, n_test)
    q10_test = np.full(n_test, -0.8)
    q90_test = np.full(n_test, 0.8)

    val_df = pd.DataFrame({"actual": y_val, "q10": q10_val, "q90": q90_val})
    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n_test, freq="h"),
        "actual": y_test, "q10": q10_test, "q90": q90_test,
    })

    test_cal, report = calibrate_forecast_adaptive(val_df, test_df)
    print("\n--- Report ---")
    for name, r in report.items():
        print(f"  Cov vor:    {r['test_coverage_uncal']:.3f}")
        print(f"  Cov nach:   {r['test_coverage_calibrated']:.3f}  "
              f"(Ziel: {r['target_coverage']:.3f})")
        print(f"  Breite Ø:   {r['test_width_uncal']:.3f} → "
              f"{r['test_width_calibrated_mean']:.3f}")
        print(f"  γ:          {r['gamma']:.4f}")
        print(f"  q̂-Range:    [{r['q_hat_min']:.3f}, {r['q_hat_max']:.3f}]")
        print(f"  q̂_init:     {r['q_hat_init']:.3f}")
