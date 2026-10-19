import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from Generate_Synthetic_Data import generate_synthetic_data

# -----------------------------------------------------------------------------
# 1) compute trust over one week (7*seq_len steps)
# -----------------------------------------------------------------------------
def compute_weekly_alarm(reported: np.ndarray,
                         model: tf.keras.Model,
                         scaler: MinMaxScaler,
                         seq_len: int,
                         tolerance: float,
                         ) -> tuple[int, np.ndarray]: # np.ndarray
    """
    Weekly detection

    Returns:
      - alarm_count: number of timesteps where |reported - predicted| > tolerance
      - preds: raw kWh predictions (same length as reported; preds[:seq_len]=0)
    """
    T = len(reported)
    assert T == 7 * seq_len, f"Expected {7*seq_len} steps, got {T}"

    # trust = np.zeros(T, dtype=float)
    preds = np.zeros(T, dtype=float)
    alarm_count = 0

    # for t in [seq_len .. T-1], predict reported[t] from reported[t-seq_len : t]
    for t in range(seq_len, T):
        # take last seq_len raw values → scale → predict → invert scale
        raw_win = reported[t - seq_len: t].reshape(-1, 1)
        X_scaled = scaler.transform(raw_win).reshape(1, seq_len, 1)
        y_scaled = model.predict(X_scaled, verbose=0)[0, 0]
        pred_kwh = scaler.inverse_transform([[y_scaled]])[0, 0]
        preds[t] = pred_kwh
        rpt = reported[t]

        err = abs(float(rpt) - float(pred_kwh))

        # deviation tolerance threshold per location
        if err > tolerance:
            alarm_count += 1

    return alarm_count, preds

# helper to rebuild the scaler from your benign validation CSV
def load_cluster_scaler(cluster_name: str, data_root: str) -> MinMaxScaler:
    path = os.path.join(data_root, "cluster_csvs", f"{cluster_name}_val.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find validation file at {path}")
    import pandas as pd
    df     = pd.read_csv(path)
    series = df["load"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(series)
    return scaler

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(__file__)
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Dataset"))
    MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Model",
                                              "Model Name"))

    pct = 0.30 #default 0.30
    loc = "Location 4 Sports Arena"
    seq_len = 96 #default 96

    tolerance = 31.908607
    alarm_threshold = 20 # 5 hours

    # 1) load the one‐week reported series
    synthetic = generate_synthetic_data()
    df_week = synthetic[pct][loc].iloc[:7 * seq_len]
    reported = df_week["reported_total_load"].to_numpy(dtype=np.float32)

    # 2) load the cluster‐model
    model = load_model(MODEL_PATH,
                        custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

    cluster_name = os.path.basename(MODEL_PATH).split("_model")[0]
    scaler = load_cluster_scaler(cluster_name, DATA_ROOT)

    # 3) compute weekly alarms (WITHOUT trust)
    alarm_count, preds = compute_weekly_alarm(
        reported, model, scaler,
        seq_len=seq_len,
        tolerance=tolerance
    )

    # 4) final decision & variance
    if alarm_count > alarm_threshold: verdict = "Bad"
    else: verdict = "Good"

    print(f"\nWeekly alarm summary for {loc} @ {pct}%:")
    print(f"  tolerance = {tolerance:.4f}")
    print(f"  alarm_threshold    = {alarm_threshold}")
    print(f"  alarm_count     = {alarm_count}")
    print(f"  weekly label    = {verdict}\n")

    # 5) plot the trajectory
    abs_err = np.abs(reported - preds)
    plt.figure(figsize=(10, 4))
    plt.plot(abs_err, label="|Reported - Predicted|", linewidth=1)
    plt.axhline(tolerance, linestyle="--", label="Tolerance")
    plt.title(f"Absolute Error vs Tolerance — {loc} — Attack {int(pct * 100)}%")
    plt.xlabel("Time step (15-min intervals)")
    plt.ylabel("Absolute Error (kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 6) Plot Predicted vs. Reported for the same week
    plt.figure(figsize=(10, 4))
    plt.plot(reported, label="Reported Total Load", linewidth=2)
    plt.plot(preds, label="Predicted Total Load", linestyle="--", linewidth=2)
    plt.title(f"Predicted vs. Reported — {loc} — Attack {int(pct * 100)}%")
    plt.xlabel("Time step (15-min intervals)")
    plt.ylabel("Load (kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 7) Metrics
    from sklearn.metrics import mean_squared_error, r2_score
    y_true = reported[seq_len:]
    y_pred = preds[seq_len:]
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE (t >= seq_len) = {mse:.6f}")
    print(f"R²  (t >= seq_len) = {r2:.6f}")