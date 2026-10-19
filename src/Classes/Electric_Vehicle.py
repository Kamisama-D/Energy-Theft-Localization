"""
EV – an electric vehicle with its
    ev_id: Unique identifier for the EV.
    battery_capacity: Total battery capacity in kWh.
    charging_rate: Charging rate in kW.
    current_charge: Current state of charge in kWh.
    charging_demand: Total charging demand during current session in kWh.
"""
import numpy as np

class EV:
    def __init__(self, ev_id:int, battery_capacity:float, charging_rate:float):
        self.ev_id = ev_id

        # Total battery capacity in kWh.
        # Drawn from a Poisson distribution with average 70 kWh,then clipped to the range [40, 100] kWh.
        if battery_capacity is None:
            cap = np.random.poisson(lam=70)
            self.battery_capacity = max(40, min(100, cap))
        else:
            self.battery_capacity = battery_capacity # in kWh

        # Choose charging rate in kW if not provided.
        if charging_rate is None:
            self.charging_rate = np.random.choice([3.3, 7.2, 10, 19.2])
        else:
            self.charging_rate = charging_rate

        self.current_charge = 0.0 # in kWh
        self.charging_demand = 0.0 # in kWh

    def set_charging_demand(self, demand:float) -> None:
        """
        Sets the energy (in kWh) the EV requires to charge.
        drawn from a Poisson distribution with average 30 kWh, then clipped to the range [10, 50] kWh.
        Cannot exceed the battery's remaining capacity.
        """
        if demand is None:
            d = np.random.poisson(lam=30)
            self.charging_demand = max(10, min(50, d))
        else:
            # ensure demand does not exceed battery capacity
            self.charging_demand = min(demand, self.battery_capacity - self.current_charge)

    def charge(self, time_hours = 1, rate = None) -> float:
        """
        Simulate charging for a given duration.
        time_hours (float): Duration of the time step in hours.
        rate (float, optional): Effective charging rate in kW. If None, use self.charging_rate.
        Returns the amount of energy (in kWh) delivered during this time step.
        """
        if rate is None:
            rate = self.charging_rate
        if self.current_charge >= self.battery_capacity:
            return 0.0 # EV is already fully charged for this session.

        # energy that can be delivered in the given time
        energy_possible = rate * time_hours # in kWh
        energy_needed = self.charging_demand - self.current_charge # in kWh
        energy_delivered = min(energy_needed, energy_possible) # in kWh
        self.current_charge += energy_delivered # in kWh
        return energy_delivered

    def is_fully_charged(self) -> bool:
        """
        Checks if the EV has met its charging demand.
        """
        return self.current_charge >= self.charging_demand # True if fully charged, False otherwise.

    def choose_charging_start_time(self, simulation_steps: int) -> int:
        """
        Choose a charging start time at 15-min resolution.

        With 90% probability, the arrival time is drawn from a Poisson distribution (λ = 12) plus ±7.5 min jitter
        and then clipped to the range 9 to 17 (i.e., 9:00 AM to 5:00 PM).
        With 10% probability, the arrival hour is drawn uniformly from 0 to 24 (any time during the day).

        Parameters:
            simulation_steps (int): Total number of simulation steps over a 24-hour period.
                                    For example, if dt = 0.25, simulation_steps would be 96.

        Returns:
            int: The simulation step index at which the EV will start charging.
        """
        if np.random.rand() < 0.9:
            # Most EVs: use Poisson(λ = 12) distribution and clip to [9, 17]
            h = np.random.poisson(lam=12)
            h = max(9, min(17, h))
            # jitter up to ±7.5 minutes = ±0.125 hours
            jitter = np.random.uniform(-0.125, 0.125)
            arrival_time = h + jitter
            # q = np.random.choice([0, 15, 30, 45]) # 15-min slots per hour = 4
            # arrival_time = h + q / 60.0

        else:
            # Minority: choose uniformly over 24 hours
            arrival_time = np.random.randint(0, 24)

        # Convert the chosen hour to a simulation step.
        step = int(arrival_time / 24 * simulation_steps)
        return min(step, simulation_steps - 1)