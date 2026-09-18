"""A playable Trombolese voice: reed, bore and the controls between them.

Ties :mod:`trombolese.reed` to :mod:`trombolese.waveguide` and exposes the
instrument as something you can actually play. Controls are read at block rate
and the model runs at audio rate.

The control set
---------------
Five continuous dimensions, of which only three have any precedent:

``pressure``
    Breath, in pascals. As on any wind instrument, it sets loudness, brightness
    and -- past a threshold -- whether the note speaks at all.
``lip_frequency``
    What the player's embouchure is tuned to. At the brass end of the reed
    morph this selects which regime of the bore speaks, exactly as lipping into
    a partial does on a trombone.
``slide``
    Extra tubing, in metres. Continuous pitch, as a trombone slide.
``alpha``
    The bore morph, cylinder to cone. **No precedent.**
``beta``
    The reed morph, lips to double reed. **No precedent.**

``alpha`` and ``beta`` are independent, so a double reed can be put on a
cylindrical bore or brass lips on a conical one -- pairings no instrument
family has ever had to make a decision about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bore import Trombolese
from .reed import Reed, ReedParameters
from .waveguide import SectionedBore, Waveguide, section_bore

__all__ = ["Controls", "Voice"]


@dataclass
class Controls:
    """The instrument's five continuous inputs, plus the blowing pressure."""

    pressure: float = 0.0
    lip_frequency: float = 60.0
    slide: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0


@dataclass
class Voice:
    """One sounding Trombolese.

    Parameters
    ----------
    instrument:
        The geometry. Pass a waveguide-tuned, pitch-compensated instrument if
        the morph is meant to hold its pitch.
    block_size:
        How often controls are read, in samples. The geometry is re-sectioned
        once per block, so this trades control latency against the cost of
        rebuilding the coefficient vectors.
    """

    instrument: Trombolese
    sample_rate: float = 48_000.0
    block_size: int = 64
    reed_parameters: ReedParameters = field(default_factory=ReedParameters)
    dispersion_reference_hz: float = 150.0

    def __post_init__(self) -> None:
        self._bore = self._section(0.0, 0.0)
        self._guide = Waveguide(self._bore)
        self._reed = Reed(self.sample_rate, self.reed_parameters,
                          self.instrument.air)
        self._dc_state = 0.0
        self._previous_radiated = 0.0
        self._geometry_key: tuple[float, float] | None = None

    def _section(self, alpha: float, slide: float) -> SectionedBore:
        return section_bore(
            self.instrument, alpha, slide, self.sample_rate,
            dispersion_reference_hz=self.dispersion_reference_hz,
        )

    def _apply_geometry(self, alpha: float, slide: float) -> None:
        """Re-section the bore, but only when the controls have actually moved."""
        key = (round(alpha, 4), round(slide, 5))
        if key == self._geometry_key:
            return
        self._geometry_key = key
        self._bore = self._section(alpha, slide)
        self._guide.adopt(self._bore)

    def render(
        self,
        n_samples: int,
        controls: Controls | None = None,
        automation=None,
    ) -> np.ndarray:
        """Render ``n_samples`` of audio.

        ``automation`` is an optional callable taking a normalised position in
        ``[0, 1]`` and returning a :class:`Controls`, which is how the sweeping
        gestures are driven.
        """
        controls = controls or Controls()
        output = np.empty(n_samples)
        impedance = self._bore.characteristic_impedance(0)

        for start in range(0, n_samples, self.block_size):
            stop = min(start + self.block_size, n_samples)

            if automation is not None:
                controls = automation(start / max(n_samples - 1, 1))

            self._apply_geometry(controls.alpha, controls.slide)
            impedance = self._bore.characteristic_impedance(0)
            self._reed.set_morph(controls.beta, controls.lip_frequency)

            for n in range(start, stop):
                injected = self._reed.step(
                    controls.pressure, self._guide.mouthpiece_return, impedance
                )
                radiated = self._guide.step(injected)

                # Radiation from the mouth goes as the time derivative of the
                # wave, which is most of why a bell sounds bright; the leaky
                # integrator that follows removes the DC the reed injects.
                differentiated = radiated - self._previous_radiated
                self._previous_radiated = radiated
                self._dc_state = 0.995 * self._dc_state + differentiated
                output[n] = self._dc_state

        return output

    def reset(self) -> None:
        self._guide.reset()
        self._reed.reset()
        self._dc_state = 0.0
        self._previous_radiated = 0.0
