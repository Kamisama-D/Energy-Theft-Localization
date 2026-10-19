import random
from abc import ABC, abstractmethod
from typing import Dict


class BaseAttackModel(ABC):
    """
    Base class for all energy-theft attack models.

    Each attack model must define initialize_meter(), which configures:
        meter.attack_factor
        meter.theft_schedule

    The default apply_attack() method calculates:
        reported_energy = actual_energy * (1 - attack_factor) # during theft_schedule
    """

    name = "Base"

    @abstractmethod
    def initialize_meter(
            self,
            meter,
            simulation_steps: int,
            dt: float,
            p_under: float
    ) -> None:
        """
        Initialize the attack parameters for one malicious smart meter.

        Parameters
        ----------
        meter:
            Smart-meter object associated with a compromised charger.

        simulation_steps:
            Total number of simulation time steps.

        dt:
            Duration of one simulation step in hours.

        p_under:
            Probability that an eligible time step is attacked.
            For example:
                p_under = 1.0 means every eligible time step is attacked.
                p_under = 0.5 means each eligible time step has a 50% chance
                of being attacked.
        """
        pass

    def apply_attack(
            self,
            actual_energy: float,
            meter,
            time_step: int
    ) -> float:
        """
        Return the reported energy for one meter record.

        If the current time step belongs to the theft schedule, the meter
        under-reports its actual consumption. Otherwise, it reports the
        actual value.
        """

        if time_step in meter.theft_schedule:
            return actual_energy * (1.0 - meter.attack_factor)

        return actual_energy


class DefaultEnergyTheft(BaseAttackModel):
    """
    Original/default energy-theft attack. [Threat_Model_1]

    Attack factor:
        Randomly selected from [0.8, 1.0].

    Attack time:
        09:00-17:00 every day.
    """

    name = "DayVar"

    def initialize_meter(
            self,
            meter,
            simulation_steps: int,
            dt: float,
            p_under: float
    ) -> None:

        steps_per_day = int(24 / dt)

        peak_start = int(9 / dt)
        peak_end = int(17 / dt)

        meter.attack_factor = random.uniform(0.8, 1.0)

        meter.theft_schedule = [
            t
            for t in range(simulation_steps)
            if (
                peak_start <= (t % steps_per_day) < peak_end
                and random.random() < p_under
            )
        ]



ATTACK_MODELS: Dict[str, BaseAttackModel] = {
    "DayVar": DefaultEnergyTheft(),
}


def get_attack_model(model_name: str) -> BaseAttackModel:
    """
    Return an attack-model object based on its registered name.

    Parameters
    ----------
    model_name:
        One of:
            "DayVar" (default)
            "DayFixed"
            "AllTimeVar"
    """

    if model_name not in ATTACK_MODELS:
        available_models = ", ".join(ATTACK_MODELS.keys())

        raise ValueError(
            f"Unknown attack model '{model_name}'. "
            f"Available attack models: {available_models}"
        )

    return ATTACK_MODELS[model_name]