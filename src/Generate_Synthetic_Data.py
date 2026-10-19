import random
import math
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Classes.System_Setting import run_simulation

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
SIM_STEPS = 36097 # 672*52=36097 one year, 672 one week
STEPS_PER_DAY = 96
NUM_DAYS = int(np.ceil(SIM_STEPS / STEPS_PER_DAY))
DT = 0.25 # hours per step
NUM_LOCS = 10 # charging locations
EV_COUNT = 300
THEFT_DAY_CHOICES = (3, 4) # default (3, 4)

# charger configuration
CHARGER_SPREAD = {
    "Location 1 Convenience Store": (0, 2),
    "Location 2 Hospital":           (6, 2),
    "Location 3 Sports Arena":       (4, 3),
    "Location 4 Sports Arena":       (4, 3),
    "Location 5 Concert Venue":      (1, 3),
    "Location 6 Hospital":           (3, 2), # note: 3 DC, 2 L2
    "Location 7 Church":             (1, 3),
    "Location 8 Housing":            (0, 0),
    "Location 9 Hospital":           (6, 2),
    "Location 10 Office Building":   (3, 2)
}

# how many DC chargers to corrupt at each attack intensity
ATTACK_CASES = {
    0.00: [],
    0.20: [("Location 2 Hospital", 6),
           ("Location 3 Sports Arena", 4)],
    0.25: [("Location 2 Hospital", 6),
           ("Location 3 Sports Arena", 4),
           ("Location 6 Hospital", 3)],
    0.30: [("Location 2 Hospital", 6),
           ("Location 3 Sports Arena", 4),
           ("Location 6 Hospital", 3),
           ("Location 10 Office Building", 2)],
    0.35: [("Location 2 Hospital", 6),
           ("Location 3 Sports Arena", 4),
           ("Location 6 Hospital", 3),
           ("Location 10 Office Building", 3),
           ("Location 4 Sports Arena", 2)],
    0.40: [("Location 2 Hospital", 6),
           ("Location 3 Sports Arena", 4),
           ("Location 6 Hospital", 3),
           ("Location 10 Office Building",3),
           ("Location 4 Sports Arena", 3),
           ("Location 5 Concert Venue", 1)]
}

# -----------------------------------------------------------------------------
# 1) Build a “benign” year—day by day
# -----------------------------------------------------------------------------
def _build_benign_full_year() -> Dict[str, pd.DataFrame]:
    # for each location (and “aggregated”) collect 1-day DataFrames
    day_dfs: Dict[str, list] = {loc: [] for loc in CHARGER_SPREAD}
    day_dfs["aggregated"] = []

    for day in range(NUM_DAYS):
        evs = EV_COUNT
        one_day, _ = run_simulation(
            num_locations        = NUM_LOCS,
            num_evs              = evs,
            dt                   = DT,
            simulation_steps     = STEPS_PER_DAY,
            charger_spread       = CHARGER_SPREAD,
            energy_theft         = False,
            malicious_percentage = 0.0,
            p_under              = 1.0,
            attack_model         = "AllTimeVar"  # One of "DayVar", "DayFixed", "AllTimeVar"
        )
        # turn each location’s dict into a DataFrame and append
        for loc, series in one_day.items():
            day_dfs[loc].append(pd.DataFrame(series))

    # concat each list into a full-year DataFrame
    return {loc: pd.concat(frames, ignore_index=True)
            for loc, frames in day_dfs.items()}

# -----------------------------------------------------------------------------
# 2) For each attack‐pct: clone the benign year and inject theft
# -----------------------------------------------------------------------------
def generate_synthetic_data(attack_model="DayVar") -> Dict[float, Dict[str, pd.DataFrame]]:
    """
    select an attack model from "DayVar" (default), "DayFixed", "AllTimeVar"
    Returns:
      { pct : { loc_name: DataFrame(full-year), …, "aggregated": DataFrame(...) } }
    """
    benign = _build_benign_full_year()

    # pick weekly theft pattern once
    stolen_days = random.choice(THEFT_DAY_CHOICES)
    theft_weekdays = set(random.sample(range(1, 7), stolen_days))
    # slots during each day to under-report:
    theft_slots = list(range(STEPS_PER_DAY)) # during 24 h

    scenarios: Dict[float, Dict[str, pd.DataFrame]] = {}
    for pct, manual_list in ATTACK_CASES.items():
        # deep-copy the benign full-year
        threat = {loc: df.copy() for loc, df in benign.items()}

        # initialize reported = actual
        for df in threat.values():
            df["reported_ev_load"]    = df["ev_load"]
            df["reported_total_load"] = df["base_load"] + df["ev_load"]

        # one random attack factor per scenario
        attack_factor = random.uniform(0.8, 1.0) # default (0.8, 1.0)

        # for each compromised location in this pct
        for loc_name, _ in manual_list:
            df = threat[loc_name]
            T  = len(df)

            # inject under-report on each theft day
            for day in range(NUM_DAYS):
                if (day % 7) in theft_weekdays:
                    start = day * STEPS_PER_DAY
                    for s in theft_slots:
                        idx = start + s
                        if idx < T:
                            # shrink reported EV load
                            df.at[idx, "reported_ev_load"] *= (1 - attack_factor)
                            df.at[idx, "reported_total_load"] = (
                                df.at[idx, "base_load"]
                                + df.at[idx, "reported_ev_load"]
                            )

        scenarios[pct] = threat

    return scenarios # both benign and theft cases

if __name__ == "__main__":
    synthetic = generate_synthetic_data()

    # pick a scenario & location to look at
    pct = 0.30
    loc = "Location 3 Sports Arena"
    df = synthetic[pct][loc]

    # --- plotting ---
    import matplotlib.dates as mdates

    for day in range(7):
        # slice out day‐by‐day
        start = day * STEPS_PER_DAY
        end = start + STEPS_PER_DAY
        day_df = df.iloc[start:end]

        print(f"\n--- Day {day + 1} at {int(pct * 100)}% attack for {loc} ---")
        print(day_df[["time", "total_load", "reported_total_load"]].to_string(index=False))

        # plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(day_df["time"], day_df["total_load"], label="Actual Total Load")
        ax.plot(day_df["time"], day_df["reported_total_load"], label="Reported Total Load")

        ticks = [0, 16, 32, 48, 64, 80, 95]
        labels = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]

        pos = day_df["time"].iloc[ticks]
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)

        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel("kWh")
        ax.set_title(f"Day {day + 1} @ {int(pct * 100)}% — {loc}")
        ax.legend(loc="upper left")
        ax.grid(True)
        fig.tight_layout()
        plt.show()