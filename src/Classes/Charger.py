"""
Charger – Each charger has its own smart meter and may have one EV connected at a time.
    charger_id: unique identifier for the charger.
    charger_type (str): Type of charger; "L2" for Level 2 or "DC" for DC fast charger. For DC chargers, we allow simultaneous charging for up to two EVs.
    charger_rate (float, optional): For DC fast chargers, the output rate in kW.
    smart_meter: each charger comes with its own smart meter.
    current_ev: the EV currently connected to the charger, if any.
"""
from typing import List

from Classes.Smart_Meter import SmartMeter
from Classes.Electric_Vehicle import EV

class Charger:
    def __init__(self, charger_id:str, charger_type:str, charger_rate:float):
        self.charger_id = charger_id
        self.charger_type = charger_type # "L2" or "DC"
        self.charger_rate = charger_rate # kW if DC fast charger
        self.smart_meter = SmartMeter(meter_id=f"Meter-{charger_id}") # Each charger has an associated SmartMeter.
        if self.charger_type == "DC":
            self.current_ev: List[EV] = [] # Allow up to 2 EVs simultaneously
        else:
            self.current_ev = None # The EV currently charging, if any

    def assign_ev(self, ev:EV) -> bool:
        """
        Assign an EV to the charger if it is available.
        """
        if self.charger_type == "DC":
            if len(self.current_ev) < 2:
                self.current_ev.append(ev)
                return True
            else:
                return False
        else:
            if self.current_ev is None:
                self.current_ev = ev
                return True
            return False

    def release_ev(self, ev:EV = None) -> None:
        """
        Release the EV from the charger.
        """
        if self.charger_type == "DC":
            if ev is not None:
                if ev in self.current_ev:
                    self.current_ev.remove(ev)
            else:
                self.current_ev = []
        else:
            self.current_ev = None

    def simulate_charging(self, time:float, time_hours:float = 1) -> float:
        """
        Simulate a charging event for the current time step and record consumption.
        - For DC chargers:
                • If one EV is charging, effective rate is 100 kW.
                • If two EVs are charging simultaneously, each EV gets 50 kW.

        time (float): Current simulation time (in hours or time step units).
        time_hours (float): Duration of the time step in hours.

        Returns:
            float: Total energy delivered (kWh) during this time step.
        """
        energy_delivered = 0
        if self.charger_type == "DC":
            if len(self.current_ev) == 0:
                return 0.0
            # Determine effective rate per EV.
            if len(self.current_ev) == 1:
                effective_rate = self.charger_rate # 100 kW when one EV is connected
            else:
                effective_rate = self.charger_rate / 2 # 50 kW per EV when two EVs are connected
            for ev in self.current_ev:
                energy = ev.charge(time_hours, rate = effective_rate)
                energy_delivered += energy
                self.smart_meter.record_consumption(time, energy)
            return energy_delivered
        else:
            if self.current_ev is None:
                return 0.0
            effective_rate = self.current_ev.charging_rate
            energy = self.current_ev.charge(time_hours, rate = effective_rate)
            energy_delivered += energy
            self.smart_meter.record_consumption(time, energy)
            return energy_delivered