"""
Aggregator – Collects data from all the charging locations (via the smart meters) and produces an aggregated consumption report for the entire region.
    aggregator_id: unique identifier for the aggregator.
    charging_locations: list of ChargingLocation objects to aggregate data from.
"""
from Classes.Charging_Location import ChargingLocation
from typing import List, Dict


class Aggregator:
    def __init__(self, aggregator_id:str, charging_locations:List[ChargingLocation]):
        self.aggregator_id = aggregator_id
        self.charging_locations = charging_locations

    def aggregate_consumption(self) -> float:
        """
        Sums up the total energy consumption recorded by all smart meters across all locations.
        """
        total = 0.0
        for location in self.charging_locations:
            for charger in location.chargers:
                total += charger.smart_meter.get_total_consumption()
        return total # total energy consumption in kWh

    def get_location_report(self, simulation_steps:int, dt:float) -> Dict:
        """
        Generate a detailed report for each charging location.
        simulation_steps (int): Total number of time steps.
        dt (float): Duration (in hours) of each time step.

        The report includes:
          - location_id
          - A list of events from each charger (each event contains meter_id, start_time, end_time, ev_id, energy)
          - Hourly aggregated energy consumption for the location.
        Returns:
            Dict: A dictionary keyed by location_id with detailed reports.
        """
        report = {}
        for location in self.charging_locations:
            location_events = []
            consumption_profile = [0.0] * simulation_steps # one value per time step
            for charger in location.chargers:
                # Append each event recorded by the charger’s smart meter.
                location_events.extend(charger.smart_meter.events)
                # For each recorded time-step, add the energy to the proper time step.
                for (t, energy) in charger.smart_meter.records:
                    if 0 <= t < simulation_steps:
                        consumption_profile[t] += energy
            report[location.location_id] = {
                "events" : location_events,
                "consumption_profile" : consumption_profile
            }
        return report