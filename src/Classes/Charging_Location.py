"""
ChargingLocation – a charging location contains several chargers (each with a smart meter).
It also includes methods to assign EVs to available chargers and to simulate charging.
    location_id: unique identifier for the location.
    num_chargers: number of chargers at the location. 5 chargers per location with a mix of 2 DC fast chargers and 3 Level 2 chargers.
    location_capacity: overall capacity of the location (e.g., maximum simultaneous kW draw).
"""
from Classes.Charger import Charger
from Classes.Electric_Vehicle import EV
from typing import List
import random

class ChargingLocation:
    def __init__(
            self,
            location_id:int,
            # num_chargers:int = 5,
            # location_capacity:float = None
            num_dc: int=2,
            num_l2: int=3
    ):
        self.location_id = location_id
        self.num_dc = num_dc
        self.num_l2 = num_l2
        #self.num_chargers = num_chargers
        # total station capacity in kW: 100 kW per DC, 10 kW per L2
        self.location_capacity = self.num_dc * 100 + self.num_l2 * 10 # in kW
        # Create chargers for this location.
        self.chargers: List[Charger] = []
        # DC fast chargers
        for i in range(num_dc):
            self.chargers.append(
                Charger(
                    charger_id=f"{location_id}-DC_{i}",
                    charger_type="DC",
                    charger_rate=100
                )
            )
        # level-2 chargers
        for i in range(num_l2):
            self.chargers.append(
                Charger(
                    charger_id=f"{location_id}-L2_{i}",
                    charger_type="L2",
                    charger_rate=None
                )
            )
        # for i in range(num_chargers):
        #     if i < num_dc:
        #         # Create a DC fast charger with 100 kW
        #         charger = Charger(charger_id=f"{location_id}-{i}", charger_type="DC", charger_rate=100)
        #     else:
        #         # Create a Level 2 charger; rate will be defined by the EV.
        #         charger = Charger(charger_id=f"{location_id}-{i}", charger_type="L2", charger_rate=None)
        #     self.chargers.append(charger)

    def get_available_chargers(self) -> List[Charger]:
        """
        Returns a list of chargers that are currently available.
        """
        available = []
        for charger in self.chargers:
            if charger.charger_type == "DC":
                if len(charger.current_ev) < 2:
                    available.append(charger)
            else:
                if charger.current_ev is None:
                    available.append(charger)
        return available

    def assign_ev_to_charger(self, ev:EV) -> None:
        """
        Randomly assigns an EV to one of the available chargers.
        Returns the assigned charger or None if none available.
        """
        available = self.get_available_chargers()
        if available:
            charger = random.choice(available)
            charger.assign_ev(ev)
            return charger
        return None

    def simulate_location_charging(self, time:float, time_hours:float = 1) -> float:
        """
        Simulates a charging cycle for all chargers in the location.
        Returns the total energy delivered at this location during this time slot.
        """
        total_energy = 0.0
        for charger in self.chargers:
            """
            time (float): Simulation time (hour).
            time_hours (float): Duration of the time step (in hours).
            """
            total_energy += charger.simulate_charging(time, time_hours)
        return total_energy

if __name__ == "__main__":
    from Classes.Base_Load import Base_Load

    # location:(num_dc, num_l2)
    Charger_Spread = {
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
    }

    print("\n>>> Verifying charger spread and station capacities:")
    base_obj = Base_Load(read_all=True)
    all_data = base_obj.get_location_data()
    loc_keys = sorted(
        [name for name in all_data if name.strip().lower() != "total load"]
    )[:10]

    for idx, name in enumerate(loc_keys):
        dc, l2 = Charger_Spread.get(name, (2, 3))
        loc = ChargingLocation(location_id=idx, num_dc=dc, num_l2=l2)
        print(
            f" • {name:<25} → DC={loc.num_dc}, "
            f"L2={loc.num_l2}, capacity={loc.location_capacity} kW"
        )