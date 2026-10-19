import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Tuple

from opt_einsum.sharing import currently_sharing

from Classes.Base_Load import Base_Load
from Classes.Charging_Location import ChargingLocation
from Classes.Electric_Vehicle import EV
from Classes.Aggregator import Aggregator
from Classes.Threat_Models import get_attack_model

# fixed charger spread per location
Default_Charger_Spread = {
    "Location 1 Convenience Store": (0, 2),
    "Location 2 Hospital": (6, 2),
    "Location 3 Sports Arena": (4, 3),
    "Location 4 Sports Arena": (4, 3),
    "Location 5 Concert Venue": (1, 3),
    "Location 6 Hospital": (2, 3),
    "Location 7 Church": (1, 3),
    "Location 8 Housing": (0, 0),
    "Location 9 Hospital": (6, 2),
    "Location 10 Office Building": (3, 2)
    #  (number of DC, number of L2)
}

def run_simulation(num_locations: int = 10,
                   num_evs: int = 100, # can be changed by user
                   dt: float = 0.25,
                   simulation_steps: int = 96, # default 96 steps for one day (15-min intervals), 36097 for one year
                   charger_spread: Dict[str, Tuple[int,int]] = None,
                   energy_theft: bool = False,
                   malicious_percentage: float = 0.0, # fraction of compromised meters
                   p_under: float = 1.0, # probability of under-report in each peak slot
                   compromised_spread: Dict[str, int] = None,
                   evs_per_day_mode: bool = False, # GUI only
                   attack_model = "Default" # Threat Models
                   ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Each charging location is paired with its corresponding base load.
    EV charging is simulated for each location; if energy_theft is True, then a fraction of
    chargers (malicious_percentage) are marked as malicious and under-report their consumption.

    Returns a dictionary with keys for each location and an "aggregated" key.
    For each location, the dictionary includes:
      "time": time axis (datetime array)
      "base_load": base load series (numpy array; length = simulation_steps)
      "ev_load": EV charging load (actual)
      "total_load": base load + EV load (actual)
    If energy_theft is True, also:
      "reported_ev_load": under-reported EV charging load
      "reported_total_load": base load + reported EV load

    Returns:
      - results (dict): per-location and aggregated loads
      - meter_dict (dict): for every SmartMeter, a dict with:
            "records":   list of (time_index, energy)
            "malicious": boolean ground truth
    """
    if charger_spread is None:
        charger_spread = Default_Charger_Spread

    # Added for different threat models
    selected_attack_model = None
    if energy_theft:
        selected_attack_model = get_attack_model(attack_model)

    # 1. Load base load data
    base_load_obj = Base_Load(read_all=True)
    all_data = base_load_obj.get_location_data()
    filtered = {k: v for k, v in all_data.items() if k.strip().lower() != "total load"}
    sorted_keys = sorted(filtered.keys())[:num_locations]

    # 2. Create EVs
    # Default mode: num_evs means total EVs for the whole simulation horizon.
    # GUI demo mode: num_evs means EVs per day.
    steps_per_day = int(24 / dt)

    if evs_per_day_mode:
        num_days = int(np.ceil(simulation_steps / steps_per_day))
    else:
        num_days = 1

    evs = []
    for day in range(num_days):
        for j in range(num_evs):
            ev_id = day * num_evs + j
            ev = EV(ev_id=ev_id, battery_capacity=None, charging_rate=None)
            ev.set_charging_demand(None)

            # Store which day this EV belongs to.
            evs.append((ev, day))

    # 3. Simulate actual charging and build location objects.
    # build all locations
    location_objects = []

    for idx, loc_key in enumerate(sorted_keys):
        # base‐load slice for this location
        df = filtered[loc_key]
        # Truncate base load data to simulation_steps (for one day, 96 rows by default)
        base_series = df['Kwh'].iloc[:simulation_steps].to_numpy()
        time_axis = pd.to_datetime(df['Time'].iloc[:simulation_steps]).values
        num_dc, num_l2 = charger_spread.get(loc_key, (2, 3))
        loc = ChargingLocation(location_id=idx, num_dc=num_dc, num_l2=num_l2)
        location_objects.append((loc_key, base_series, time_axis, loc))

    # for GUI
    for ev, day in evs:
        if ev.is_fully_charged():
            continue

        if evs_per_day_mode:
            day_start = day * steps_per_day
            remaining_steps = min(steps_per_day, simulation_steps - day_start)

            if remaining_steps <= 0:
                continue

            local_start_step = ev.choose_charging_start_time(remaining_steps)
            start_step = day_start + local_start_step
        else:
            start_step = ev.choose_charging_start_time(simulation_steps)


        # shuffle locations, try the first that has an available charger
        for (loc_key, base_series, time_axis, loc) in random.sample(location_objects, num_locations):
            charger = loc.assign_ev_to_charger(ev)
            if charger is None:
                continue # If no charger is available, skip this EV for this round.

            # Simulate charging from the start time until either complete or simulation end.
            current_step = start_step
            while not ev.is_fully_charged() and current_step < simulation_steps:
                loc.simulate_location_charging(time=current_step, time_hours=dt)
                current_step += 1

            # Record a summary event in the smart meter for this charging session.
            charger.smart_meter.record_event(
                start_time = start_step * dt,
                end_time = (current_step - 1) * dt, # Last time step used for charging
                ev_id = ev.ev_id,
                energy = ev.current_charge,
            )

            # Release the EV from the charger.
            if charger.charger_type == 'DC':
                charger.release_ev(ev)
            else:
                charger.release_ev()

            break  # EV assigned & simulated → move to next EV

    # 4. Select malicious meters and initialize their attack behavior - Added for different threat models
    if energy_theft:

        # --------------------------------------------------
        # Case 1: GUI mode
        #
        # compromised_spread specifies the number of
        # compromised meters at each location.
        # --------------------------------------------------
        if compromised_spread is not None:

            for loc_key, _, _, loc in location_objects:
                chargers = loc.chargers

                # Number of compromised meters assigned to this location
                num_mal = compromised_spread.get(loc_key, 0)

                # Prevent the number of compromised meters from exceeding
                # the number of chargers at this location.
                num_mal = min(num_mal, len(chargers))

                malicious_chargers = random.sample(
                    chargers,
                    num_mal
                )

                for charger in malicious_chargers:
                    meter = charger.smart_meter
                    meter.malicious = True

                    selected_attack_model.initialize_meter(
                        meter=meter,
                        simulation_steps=simulation_steps,
                        dt=dt,
                        p_under=p_under
                    )

        # --------------------------------------------------
        # Case 2: Original mode
        #
        # Select compromised DC chargers globally according
        # to malicious_percentage.
        # --------------------------------------------------
        else:

            dc_list = []

            for _, _, _, loc in location_objects:
                for charger in loc.chargers:
                    if charger.charger_type == "DC":
                        dc_list.append(charger)

            # Preserve your original calculation.
            # malicious_percentage represents a fraction of 50 meters.
            num_mal = int(malicious_percentage * 50)

            # Do not select more meters than the available DC chargers.
            num_mal = min(num_mal, len(dc_list))

            malicious_chargers = random.sample(
                dc_list,
                num_mal
            )

            for charger in malicious_chargers:
                meter = charger.smart_meter
                meter.malicious = True

                selected_attack_model.initialize_meter(
                    meter=meter,
                    simulation_steps=simulation_steps,
                    dt=dt,
                    p_under=p_under
                )



    # 5. Build results and reported loads
    results = {}
    agg_base = np.zeros(simulation_steps)
    agg_ev = np.zeros(simulation_steps)
    agg_rep = np.zeros(simulation_steps)

    for loc_key, base_series, time_axis, loc in location_objects:
        ev_load = np.zeros(simulation_steps)
        reported_load = np.zeros(simulation_steps)

        # Added for different threat models
        for charger in loc.chargers:
            meter = charger.smart_meter

            for t, energy in meter.records:
                ev_load[t] += energy

                if energy_theft and meter.malicious:
                    reported_energy = selected_attack_model.apply_attack(
                        actual_energy=energy,
                        meter=meter,
                        time_step=t
                    )
                else:
                    reported_energy = energy

                reported_load[t] += reported_energy


        total_load = base_series + ev_load

        results[loc_key] = {
            "time": time_axis,
            "base_load": base_series,
            "ev_load": ev_load,
            "total_load": total_load,
        }
        if energy_theft:
            results[loc_key]["reported_ev_load"] = reported_load
            results[loc_key]["reported_total_load"] = base_series + reported_load

        agg_base += base_series
        agg_ev += ev_load
        if energy_theft:
            agg_rep += reported_load

    aggregated = {
        "base_load": agg_base,
        "ev_load": agg_ev,
        "total_load": agg_base + agg_ev,
    }
    if energy_theft:
        aggregated["reported_ev_load"] = agg_rep
        aggregated["reported_total_load"] = agg_base + agg_rep
    results["aggregated"] = aggregated

    # 6. Build meter-level ground truth
    meter_dict = {}
    for _, _, _, loc in location_objects:
        for charger in loc.chargers:
            m = charger.smart_meter
            meter_dict[m.meter_id] = {
                "records": m.records.copy(),
                "malicious": m.malicious,
                "attack_factor": m.attack_factor,
                "theft_schedule": m.theft_schedule.copy(),
                "attack_model": attack_model if m.malicious else None
            }

    # 7. Print Report
    aggregator = Aggregator(
        aggregator_id="Region-1",
        charging_locations=[loc for _, _, _, loc in location_objects]
    )
    total_consumption = aggregator.aggregate_consumption()
    location_report = aggregator.get_location_report(simulation_steps, dt)

    print("----- Simulation Report -----")
    print(f"Aggregator ID: {aggregator.aggregator_id}")
    print(f"Total Energy Consumption across all locations: {total_consumption:.2f} kW")
    for location_id, rep in location_report.items():
        print(f"\nLocation {location_id}:")
        print("Consumption Profile (kW per dt interval):", rep["consumption_profile"])
        print("Detailed Charging Events:")
        for event in rep["events"]:
            print(f"  Meter {event['meter_id']} | EV {event['ev_id']} | "
                  f"Start: {event['start_time']}h, End: {event['end_time']}h | "
                  f"Consumption: {event['energy']:.2f} kWh")

    return results, meter_dict

#########################
# Plotting for benign results
#########################
def plot_results_per_location(results: Dict[str, Dict]) -> None:
    """
    For each location (excluding the key "aggregated"), plot a figure showing:
      - Base load per dt interval.
      - EV charging load (actual) per dt interval.
      - Total grid load (base load + EV load) per dt interval.
    """
    for loc, res in results.items():
        if loc == "aggregated":
            continue

        # ensure time axis is datetime
        T = len(res["base_load"])
        steps = np.arange(T)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(steps, res["base_load"], label="Base Load", linestyle=":", color="green")
        ax.plot(steps, res["ev_load"], label="EV Charging", linestyle="--", color="orange")
        ax.plot(steps, res["total_load"], label="Total Grid", linestyle="-", color="blue")
        ax.set_title(f"Grid Load at {loc}")
        ax.set_ylabel("Load (kWh)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        # format x-axis for 1-day
        if T == 96:
            # map 96 points to 0–24 h
            ticks = [0,
                     int(4 * (T / 24)),
                     int(8 * (T / 24)),
                     int(12 * (T / 24)),
                     int(16 * (T / 24)),
                     int(20 * (T / 24)),
                     T - 1]
            labels = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Time (Hours)")
        plt.show()

def plot_aggregated_results(results: Dict[str, Dict]) -> None:
    """
    Plot a single aggregated figure (across all locations) showing:
      - Aggregated Base Load.
      - Aggregated EV Charging Load.
      - Aggregated Total Grid Load.
    """
    agg = results["aggregated"]
    T = len(agg["base_load"])
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, agg["base_load"],  label="Base Load",    linestyle=":",  color="green")
    ax.plot(steps, agg["ev_load"],    label="EV Charging",  linestyle="--", color="orange")
    ax.plot(steps, agg["total_load"], label="Total Grid",   linestyle="-",  color="blue")
    ax.set_title("Aggregated Grid Load Across All Locations")
    ax.set_ylabel("Load (kW)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    # format x-axis for 1-day
    if T == 96:
        # map 96 points to 0–24 h
        ticks = [0,
                 int(4 * (T / 24)),
                 int(8 * (T / 24)),
                 int(12 * (T / 24)),
                 int(16 * (T / 24)),
                 int(20 * (T / 24)),
                 T - 1]
        labels = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Time (Hours)")
    plt.show()

##################################
# Threat Model Plotting Functions
##################################
def plot_threat_results_per_location(results: Dict[str, Dict]) -> None:
    """
    For each location (except "aggregated"), plot two figures:
      1. One figure that shows:
           - Base load (kW) in green dotted line.
           - Actual total grid load (base load + actual EV load) in blue solid line.
           - Reported total grid load (base load + under-reported EV load) in red dashed line.
      2. A second figure that shows a bar chart of the under-report percentage.
    """
    for loc, res in results.items():
        if loc == "aggregated" or "reported_total_load" not in res:
            continue
        T = len(res["base_load"])
        steps = np.arange(T)

        # Figure 1: Load curves.
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(steps, res["base_load"], label="Base Load", linestyle=":", color="green")
        ax.plot(steps, res["total_load"], label="Actual Total Grid Load", linestyle="-", color="blue")
        ax.plot(steps, res["reported_total_load"], label="Reported Total Grid Load", linestyle="--", color="red")
        ax.set_title(f"Grid Load at {loc} (Energy Theft)")
        ax.set_ylabel("Load (kW)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        # format x-axis for 1-day
        if T == 96:
            # map 96 points to 0–24 h
            ticks = [0,
                     int(4 * (T / 24)),
                     int(8 * (T / 24)),
                     int(12 * (T / 24)),
                     int(16 * (T / 24)),
                     int(20 * (T / 24)),
                     T - 1]
            labels = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Time (Hours)")
        plt.show()

        # Figure 2: Under-report percentage.
        actual = res["ev_load"]
        reported = res["reported_ev_load"]
        under_pct = np.divide(actual - reported,
                              actual,
                              out=np.zeros_like(actual),
                              where=actual != 0) * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        # use time numeric or datetime; bar width ~ dt
        ax.bar(steps, under_pct, color="pink", width=1.0)
        ax.set_title(f"Under-report Percentage at {loc}")
        ax.set_ylabel("Under-report (%)")
        ax.grid(True)
        plt.tight_layout()
        # format x-axis for 1-day
        if T == 96:
            # map 96 points to 0–24 h
            ticks = [0,
                     int(4 * (T / 24)),
                     int(8 * (T / 24)),
                     int(12 * (T / 24)),
                     int(16 * (T / 24)),
                     int(20 * (T / 24)),
                     T - 1]
            labels = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Time (Hours)")
        plt.show()


def plot_threat_aggregated_results(results: Dict[str, Dict]) -> None:
    """
    Plot a single aggregated figure (across all locations) for the threat model.
    Displays:
      - Aggregated Base Load (green dotted)
      - Aggregated Actual Total Grid Load (blue solid)
      - Aggregated Reported Total Grid Load (red dashed)
    Also plots the aggregated under-report percentage in a separate bar chart.
    """
    agg = results["aggregated"]
    T = len(agg["base_load"])
    steps = np.arange(T)

    # Figure 1: loads
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, agg["base_load"], label="Base Load", linestyle=":", color="green")
    ax.plot(steps, agg["total_load"], label="Actual Total", linestyle="-", color="blue")
    ax.plot(steps, agg["reported_total_load"], label="Reported Total", linestyle="--", color="red")
    ax.set_title("Aggregated Grid Load (Energy Theft)")
    ax.set_ylabel("Load (kW)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Figure 2: under-report %
    under_agg = np.divide(agg["ev_load"] - agg["reported_ev_load"],
                          agg["ev_load"],
                          out=np.zeros_like(agg["ev_load"]),
                          where=agg["ev_load"] != 0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(steps, under_agg, color="pink", width=1.0)
    ax.set_title("Aggregated Under-report Percentage of EV Charging Load")
    ax.set_ylabel("Under-report (%)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # For testing: run both benign and threat mode if desired.
    np.random.seed(42)
    random.seed(42)
    # Uncomment one of the following to test.

    '''
    For a single day analysis, use simulation_steps=96.
    For one year analysis, use simulation_steps=36097.
    '''

    '''For benign simulation:'''
    results_benign, meter_dict = run_simulation(num_locations=10,
                                                num_evs=300,
                                                dt=0.25,
                                                simulation_steps=96,
                                                energy_theft=False)
    plot_aggregated_results(results_benign)
    plot_results_per_location(results_benign)

    '''For threat simulation:'''
    results_threat, meter_dict = run_simulation(num_locations=10,
                                                num_evs=300,
                                                dt=0.25,
                                                simulation_steps=96,
                                                energy_theft=True,
                                                malicious_percentage=0.3,
                                                p_under=1.0,
                                                attack_model="AllTime") #One of "Default", "FixedAtk", "AllTime"
    plot_threat_aggregated_results(results_threat)
    plot_threat_results_per_location(results_threat)