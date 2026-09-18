"""Physical properties of air, as functions of temperature.

All quantities are SI. The formulas are the standard engineering
approximations used throughout the musical-acoustics literature
(Fletcher & Rossing, *The Physics of Musical Instruments*, App. A);
they are accurate to well under 1% over the range an instrument
actually sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Air:
    """Thermodynamic and transport properties of air at a given temperature.

    Attributes
    ----------
    temperature:
        Air temperature in degrees Celsius. 20 C is the usual reference;
        a warmed-up wind instrument sits nearer 30 C, which raises the
        speed of sound by roughly 2% and sharpens every resonance by
        about 30 cents. That is a large enough effect to matter when
        comparing model output against a real instrument.
    """

    temperature: float = 20.0

    @property
    def speed_of_sound(self) -> float:
        """Speed of sound, m/s."""
        return 331.45 * np.sqrt(1.0 + self.temperature / 273.15)

    @property
    def density(self) -> float:
        """Density, kg/m^3, at one atmosphere."""
        return 1.2929 * 273.15 / (273.15 + self.temperature)

    @property
    def viscosity(self) -> float:
        """Dynamic (shear) viscosity, Pa.s."""
        return 1.708e-5 * (1.0 + 0.0029 * self.temperature)

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity nu = mu / rho, m^2/s."""
        return self.viscosity / self.density

    @property
    def heat_capacity_ratio(self) -> float:
        """Ratio of specific heats, gamma. Essentially constant for air."""
        return 1.4017

    @property
    def prandtl_number(self) -> float:
        """Prandtl number. Essentially constant for air."""
        return 0.708


AIR_20C = Air(20.0)
