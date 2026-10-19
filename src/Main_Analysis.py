import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from Generate_Synthetic_Data import (
    generate_synthetic_data,
    ATTACK_CASES,
    EV_COUNT,
    SIM_STEPS,
)

from Compute_Trust_Value import load_cluster_scaler  # reuse scaler loader


def _predict_one_step(series: np.ndarray,
                      t: int,
                      seq_len: int,
                      model: tf.keras.Model,
                      scaler) -> float:
    """Predict series[t] using series[t-seq_len:t] (same pattern as trust version)."""
    raw_win = series[t - seq_len: t].reshape(-1, 1)
    X_scaled = scaler.transform(raw_win).reshape(1, seq_len, 1)
    y_scaled = model.predict(X_scaled, verbose=0)[0, 0]
    pred_kwh = scaler.inverse_transform([[y_scaled]])[0, 0]
    return float(pred_kwh)


def analyze_location_without_trust(
    synthetic: dict[float, dict[str, pd.DataFrame]],
    location: str,
    pct: float,
    model: tf.keras.Model,
    scaler,
    *,
    tolerance: float,
    alarm_threshold: int,
    seq_len: int = 96,
    week_len: int = 7 * 96
) -> tuple[dict, list[str], list[np.ndarray], list[int]]:
    """
    Weekly binary detection (WITHOUT trust):
      - alarm_count = number of timesteps where |reported - predicted| > tolerance
      - weekly label = "Compromised" if alarm_count >= alarm_threshold else "Benign"

    Returns:
      (summary_dict, week_labels_list, full_error_history, weekly_alarm_counts)
      - full_error_history: list of np.ndarray, each length=week_len (abs error curve per week)
      - weekly_alarm_counts: list of int, alarm_count per week
    """
    if pct not in synthetic:
        raise ValueError(f"No scenario at {pct * 100:.0f}%")
    results = synthetic[pct]

    if location not in results:
        raise KeyError(f"Location {location!r} not found in simulation results")

    reported = results[location]["reported_total_load"].to_numpy(dtype=np.float32)
    n_weeks = len(reported) // week_len

    week_labels: list[str] = []
    full_error_history: list[np.ndarray] = []
    weekly_alarm_counts: list[int] = []

    for w in range(n_weeks):
        start = w * week_len
        end = start + week_len
        week_data = reported[start:end]

        errs = np.zeros(week_len, dtype=float)
        alarm_count = 0

        # only start predicting at t >= seq_len
        for t in range(seq_len, week_len):
            pred = _predict_one_step(week_data, t, seq_len, model, scaler)
            err = abs(float(week_data[t]) - pred)
            errs[t] = err
            if err > tolerance:
                alarm_count += 1

        weekly_alarm_counts.append(alarm_count)
        full_error_history.append(errs)

        state = "Compromised" if alarm_count >= alarm_threshold else "Benign"
        week_labels.append(state)

    # count
    n_comp = week_labels.count("Compromised")
    n_benign = week_labels.count("Benign")

    # ground truth: was this location compromised?
    manual_list = ATTACK_CASES[pct]
    manual_dc = dict(manual_list)
    is_compromised = manual_dc.get(location, 0) > 0

    # compute weekly metrics
    if is_compromised:
        TP = n_comp
        FN = n_benign
        TPR = TP / (TP + FN) if (TP + FN) else np.nan
        TN = FP = FPR = np.nan
    else:
        FP = n_comp
        TN = n_benign
        FPR = FP / (FP + TN) if (FP + TN) else np.nan
        TP = FN = TPR = np.nan

    summary = {
        "Location": location,
        "Attack_%": pct * 100,
        "Tolerance": tolerance,
        "Alarm_Threshold": alarm_threshold,
        "EVs_simulated": EV_COUNT,
        "Simulation_Steps": SIM_STEPS,
        "Weeks_total": n_weeks,
        "Weeks_Benign": n_benign,
        "Weeks_Compromised": n_comp,
        "GroundTruth_Compromised": is_compromised,
        "TPR": TPR,
        "FPR": FPR,
    }

    return summary, week_labels, full_error_history, weekly_alarm_counts


if __name__ == "__main__":
    synthetic = generate_synthetic_data()

    # 1) choose location & %attack
    LOCATION = "Location Name"
    PCT = 0.30
    MODEL_PATH = ("..")
    DATA_ROOT = os.path.join("..", "Dataset")

    # 2) detection hyperparameters (WITHOUT TRUST)
    # tolerance should be the CLUSTER tolerance for this location's cluster
    TOLERANCE = # location-specific tolerance
    ALARM_THRESHOLD = 20   # e.g., 5 hours = 20 timesteps (15-min each)

    SEQ_LEN = 96
    WEEK_LEN = 7 * 96

    # 3) load your pretrained LSTM for this cluster
    model = load_model(
        MODEL_PATH,
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )
    cluster_name = os.path.basename(MODEL_PATH).split("_model")[0]
    scaler = load_cluster_scaler(cluster_name, DATA_ROOT)

    # 4) run the one-shot analysis
    start_time = time.time()
    summary, week_labels, full_error_history, weekly_alarm_counts = analyze_location_without_trust(
        synthetic, LOCATION, PCT, model, scaler,
        tolerance=TOLERANCE,
        alarm_threshold=ALARM_THRESHOLD,
        seq_len=SEQ_LEN,
        week_len=WEEK_LEN
    )
    elapsed = time.time() - start_time

    # 5) print the summary
    print("\n=== Detection Summary (Without Trust) ===")
    print(pd.DataFrame([summary]).to_string(index=False))

    print("\nWeek-by-week labels:")
    for i, (lab, cnt) in enumerate(zip(week_labels, weekly_alarm_counts), 1):
        print(f"  Week {i:2d}: {lab:11s} (alarm_count={cnt})")

    # 6) plot: absolute error trajectory over all weeks (optional, similar to trust plot)
    all_err = np.concatenate(full_error_history)
    n_weeks = len(full_error_history)

    plt.figure(figsize=(10, 4))
    plt.plot(all_err, marker="o", markersize=2, linewidth=1)
    plt.axhline(TOLERANCE, color="red", linestyle="--", label="tolerance")
    plt.title(f"Absolute Error over {n_weeks} weeks — {LOCATION}")
    plt.xlabel("Time step (15-min intervals)")
    plt.ylabel("|Reported - Predicted| (kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 7) report elapsed time in H:M:S
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"\nTotal runtime: {int(h)}h {int(m)}m {s:.2f}s")