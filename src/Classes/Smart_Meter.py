"""
SmartMeter – a metering device attached to a charger;
it records the energy drawn during each charging event.
    meter_id: unique identifier for the smart meter
    records = [(time, energy)]: records the energy drawn during each charging event.
    events = [{meter_id, start_time, end_time, ev_id, energy}]: records complete charging event for an EV
    malicious: A boolean indicating if this meter is malicious or benign (True = malicious, False = benign)
    attack_factor: if malicious, under-report energy consumption by 80% - 100%
    theft_schedule: list of time step indices when to under-report
"""
class SmartMeter:
    def __init__(self, meter_id:str):
        self.meter_id = meter_id
        self.records = [] # List to store (time, energy) tuples
        self.events = [] # list to store detailed events for each EV charging session (each event is a dict)
        self.malicious = False # Flag indicating if this meter is malicious (default benign)
        self.attack_factor = 0.0 # If malicious, the under-reporting factor (0.1 to 0.7)
        self.theft_schedule = [] # time slots to apply theft

    def record_consumption(self, time:float, energy:float) -> None:
        """
        Records the energy consumption (in kWh) at a specific time step.
        """
        self.records.append((time, energy))

    def record_event(self, start_time:float, end_time:float, ev_id:int, energy:float) -> None:
        """
        Record a complete charging event for an EV.
        """
        event = {
            "meter_id": self.meter_id,
            "start_time": start_time, # in hours
            "end_time": end_time, # in hours
            "ev_id": ev_id,
            "energy": energy # in kWh
        }
        self.events.append(event)

    def get_total_consumption(self) -> float:
        """
        Compute the total energy consumption from the recorded time steps.
        If the meter is malicious, under-report the consumption.
        """
        total = sum(energy for _, energy in self.records)
        if self.malicious:
            # Under-report consumption by reducing the total by the attack factor.
            total *= (1 - self.attack_factor)
        return total