# Compute_Tolerance_without_Trust.py
# compute deviation without normalization for theft detection without trust assessment

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from Classes.System_Setting import run_simulation
from Generate_Synthetic_Data import CHARGER_SPREAD, NUM_LOCS, DT
from Compute_Trust_Value import load_cluster_scaler
# from typing import List

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION FOR TOLERANCE COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
STEPS_PER_DAY = 96
NUM_WEEKS     = 4
SIM_STEPS     = STEPS_PER_DAY * 7 * NUM_WEEKS  # 96*7*4 = 2688
SEQ_LEN       = 96  # model window size

# ──────────────────────────────────────────────────────────────────────────────
#  CORE FUNCTION: run benign sim + compute deltas + μ/σ/τ
# ──────────────────────────────────────────────────────────────────────────────
def run_benign_simulation():
    print("\n[INFO] Running one-time benign simulation...")
    start = time.time()

    results, _ = run_simulation(
        num_locations        = NUM_LOCS,
        num_evs              = 300,
        dt                   = DT,
        simulation_steps     = SIM_STEPS,
        charger_spread       = CHARGER_SPREAD,
        energy_theft         = False,
        malicious_percentage = 0.0,
        p_under              = 1.0
    )

    elapsed = time.time() - start
    print(f"[INFO] Simulation finished in {elapsed:.2f}s.\n")

    return results

def analyze_cluster(cluster_name: str,
                    cfg: dict,
                    results: dict,
                    data_root: str,
                    k: float):

    start = time.time()
    model_path = cfg["model_path"]
    locations  = cfg["locations"]

    # load model + scaler
    model = load_model(
        model_path,
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )

    scaler_cluster_name = os.path.basename(model_path).split("_model")[0]
    scaler = load_cluster_scaler(scaler_cluster_name, data_root)

    # ---- cluster mean series ----
    series_stack = np.vstack([
        np.asarray(results[loc]["total_load"], dtype=float)
        for loc in locations
    ])
    cluster_series = series_stack.mean(axis=0)

    mu_c, sigma_c, tol_c = compute_tolerance_from_series(cluster_series, model, scaler, k)

    # ---- per-location ----
    per_loc_stats = []
    for loc in locations:
        loc_series = np.asarray(results[loc]["total_load"], dtype=float)
        mu_l, sigma_l, tol_l = compute_tolerance_from_series(loc_series, model, scaler, k)
        per_loc_stats.append((loc, mu_l, sigma_l, tol_l))

    # ---- print ----
    elapsed = time.time() - start

    print(f"\nTolerance analysis for {cluster_name} with locations {locations}")
    print(f"  [Cluster-mean] mean |e| = {mu_c:.6f}")
    print(f"  [Cluster-mean] std  |e| = {sigma_c:.6f}")
    print(f"  [Cluster-mean] tol τ    = {tol_c:.6f}")

    print("  Per-location tolerances:")
    for loc, mu_l, sigma_l, tol_l in per_loc_stats:
        print(f"    - {loc}: mean={mu_l:.6f}, std={sigma_l:.6f}, tol={tol_l:.6f}")

    print(f"Cluster runtime: {elapsed:.2f}s\n")

def compute_tolerance_from_series(series: np.ndarray,
                                 model: tf.keras.Model,
                                 scaler: MinMaxScaler,
                                 k: float) -> tuple[float, float, float]:
    deltas = []
    for t in range(SEQ_LEN, SIM_STEPS):
        window = series[t-SEQ_LEN: t].reshape(-1, 1)
        Xs     = scaler.transform(window).reshape(1, SEQ_LEN, 1)
        y_s    = model.predict(Xs, verbose=0)[0, 0]
        pred   = scaler.inverse_transform([[y_s]])[0, 0]
        rpt    = float(series[t])
        deltas.append(abs(rpt - pred))

    mu    = float(np.mean(deltas))
    sigma = float(np.std(deltas))
    tol   = mu + k * sigma
    return mu, sigma, tol


# ──────────────────────────────────────────────────────────────────────────────
#  DEMO ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SCRIPT_DIR  = os.path.dirname(__file__)
    # MODEL_PATH  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Model",
    #                                            "Cluster_1-1_model3-4.h5"))
    DATA_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Dataset"))
    # LOCATION    = "Location 6 Hospital"
    K    = 3.0

    CLUSTERS = {
        "Cluster_0-0": {
            "model_path": os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Model", "Model_Name")),
            "locations": [
                "Location 3 Sports Arena",
                "Location 4 Sports Arena"
            ]
        },
        "Cluster_0-2": {
            "model_path": os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Model", "Model_Name")),
            "locations": [
                "Location 5 Concert Venue",
                "Location 7 Church"
            ]
        },
        "Cluster_1-0": {
            "model_path": os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Model", "Model_Name")),
            "locations": [
                "Location 2 Hospital",
                "Location 9 Hospital"
            ]
        },
        "Cluster_1-1": {
            "model_path": os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Model", "Model_Name")),
            "locations": [
                "Location 10 Office Building",
                "Location 6 Hospital"
            ]
        }
    }

    # Run simulation once
    results = run_benign_simulation()

    # Analyze each cluster
    for cluster_name, cfg in CLUSTERS.items():
        analyze_cluster(cluster_name, cfg, results, DATA_ROOT, K)