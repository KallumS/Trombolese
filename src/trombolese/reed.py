"""The excitation: a reed that morphs from brass lips to a double reed.

A wind instrument is a resonator plus a pressure-controlled valve, and the two
parents of this instrument disagree about which kind of valve. Brass lips are
**outward-striking**: raising the pressure difference across them pushes them
apart, so they blow *open*. A double reed is **inward-striking**: the same
pressure difference pushes the blades together, so it blows *closed*. Nearly
everything that distinguishes how the two families speak, slur and crack
follows from that single sign.

So the excitation morph is, at its heart, that sign made continuous. ``beta``
runs from 0 (lips) to 1 (double reed), and the coupling coefficient runs from
+1 to -1 with it, passing through zero -- a valve that barely responds to
pressure at all and lets the bore ring almost undriven. Alongside the sign, the
valve's resonance, damping, aperture and stiffness interpolate between
plausible values for the two instruments.

``beta`` is deliberately independent of the bore morph ``alpha``. A real
instrument has to pair a lip reed with a brass bore; this one can put a double
reed on a cylinder, or lips on a cone, and those combinations have no acoustic
precedent at all.

Model
-----
A single-degree-of-freedom valve, which is the standard reduction for both
families:

    y'' + (w/Q) y' + w^2 (y - y_rest) = sigma * dp / mu

with ``y`` the opening, ``dp = p_mouth - p_bore`` the pressure across it, and
``sigma`` the striking sign. Flow through the opening is quasi-static Bernoulli,

    U = width * max(y, 0) * sgn(dp) * sqrt(2 |dp| / rho)

Coupling to the bore is what makes this delicate. The bore presents a wave
impedance ``Zc``, so the pressure the valve sees depends on the flow it is
itself producing -- ``dp`` and ``U`` are mutually defined. Rather than iterate,
this solves the pair exactly: substituting ``dp = A - Zc U`` into Bernoulli
gives a quadratic in ``U`` with a closed-form root. That is what keeps the
model stable at high blowing pressure, where an explicit one-sample-delayed
guess would blow up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import AIR_20C, Air

__all__ = ["ReedParameters", "Reed"]


def _lerp(beta: float, at_zero: float, at_one: float) -> float:
    return (1.0 - beta) * at_zero + beta * at_one


@dataclass
class ReedParameters:
    """Endpoint values for the excitation morph.

    Defaults are plausible rather than measured: brass lips are heavy, slack
    and tuned near the note being played, while a double reed is small, stiff
    and resonates far above the bore's regimes.

    Attributes
    ----------
    lip_frequency:
        Default natural frequency of the valve at the brass end, in Hz. It is a
        *control*, not a constant -- a brass player chooses which regime speaks
        by setting it -- so it is normally overridden per note.
    reed_frequency_ratio:
        How far above the embouchure control the valve sits at the double-reed
        end. A double reed resonates well above the regimes it drives, but it
        must still answer to the player: biting raises a reed's effective
        stiffness and favours the upper register, just as tightening the lips
        does on brass. Keeping the valve frequency proportional to the
        embouchure control at *both* ends is what preserves register control
        across the whole morph -- pinning it to an absolute frequency at the
        reed end leaves the player with no pitch control whatsoever, which the
        model duly demonstrated.
    lip_q, reed_q:
        Quality factor of the valve. These are higher than a physical lip's,
        and deliberately so: below about 7 the valve is so damped that the
        model stops self-oscillating altogether, and the instrument only rings
        down from its attack transient.
    lip_width, reed_width:
        Effective width of the opening, in metres.
    lip_rest_opening, reed_rest_opening:
        Opening at rest, in metres.
    lip_closing_pressure, reed_closing_pressure:
        The static pressure difference that would just close the valve, in
        pascals. This sets the valve's effective mass, and with it the blowing
        pressure the instrument wants. It must stay well above the blowing
        pressure: when it does not, the valve is driven into its end stops
        every cycle and degenerates into a relaxation oscillator that the
        bore alone controls, at which point the embouchure stops selecting
        regimes and the instrument is unplayable.
    """

    lip_frequency: float = 60.0
    reed_frequency_ratio: float = 3.5
    lip_q: float = 15.0
    reed_q: float = 12.0
    lip_width: float = 0.012
    reed_width: float = 0.010
    lip_rest_opening: float = 3.0e-4
    reed_rest_opening: float = 3.0e-4
    lip_closing_pressure: float = 12000.0
    reed_closing_pressure: float = 7000.0


class Reed:
    """A morphable pressure-controlled valve driving a waveguide.

    Call :meth:`set_morph` at control rate and :meth:`step` once per sample.
    """

    def __init__(
        self,
        sample_rate: float = 48_000.0,
        parameters: ReedParameters | None = None,
        air: Air = AIR_20C,
    ) -> None:
        self.sample_rate = sample_rate
        self.parameters = parameters or ReedParameters()
        self.air = air

        self.opening = 0.0
        self.velocity = 0.0
        self._frequency = self.parameters.lip_frequency
        self.set_morph(0.0)
        self.reset()

    def reset(self) -> None:
        self.opening = self._rest_opening
        self.velocity = 0.0

    def set_morph(self, beta: float, frequency: float | None = None) -> None:
        """Interpolate the valve between lips (``beta = 0``) and reed (``1``).

        ``frequency`` overrides the valve's natural frequency, which is how a
        player selects a regime at the brass end of the morph. It is scaled
        toward the double reed's own stiff resonance as ``beta`` rises, since a
        cane reed's pitch is its own business and not the player's.
        """
        beta = float(np.clip(beta, 0.0, 1.0))
        self.beta = beta
        p = self.parameters

        # +1 blows open (lips), -1 blows closed (double reed). The zero
        # crossing is a valve that scarcely responds to pressure at all.
        self.striking = 1.0 - 2.0 * beta

        embouchure = p.lip_frequency if frequency is None else frequency
        self._frequency = embouchure * _lerp(beta, 1.0, p.reed_frequency_ratio)
        self._q = _lerp(beta, p.lip_q, p.reed_q)
        self._width = _lerp(beta, p.lip_width, p.reed_width)
        self._rest_opening = _lerp(beta, p.lip_rest_opening, p.reed_rest_opening)

        closing = _lerp(beta, p.lip_closing_pressure, p.reed_closing_pressure)
        omega = 2.0 * np.pi * self._frequency
        # mu follows from "the pressure that just closes the valve":
        # closing = mu * omega^2 * rest_opening.
        self._inverse_mass = (omega**2 * self._rest_opening) / closing

    def step(self, mouth_pressure: float, returning_wave: float,
             impedance: float) -> float:
        """Advance one sample and return the wave to inject into the bore.

        Parameters
        ----------
        mouth_pressure:
            Blowing pressure behind the valve, in pascals.
        returning_wave:
            The bore's backward-travelling wave arriving at the mouthpiece.
        impedance:
            Wave impedance of the bore's first section.
        """
        # Pressure difference the valve would see at zero flow.
        incident = mouth_pressure - 2.0 * returning_wave

        area = self._width * max(self.opening, 0.0)
        if area > 0.0:
            # U = area sqrt(2 |dp| / rho) with dp = incident - Zc U, solved
            # exactly as a quadratic rather than iterated.
            k = 2.0 * area**2 / self.air.density
            magnitude = abs(incident)
            discriminant = (k * impedance) ** 2 + 4.0 * k * magnitude
            flow = 0.5 * (-k * impedance + np.sqrt(discriminant))
            flow = np.copysign(flow, incident)
        else:
            flow = 0.0

        pressure_difference = incident - impedance * flow

        # Trapezoidal-ish integration of the valve; stable at these rates.
        dt = 1.0 / self.sample_rate
        omega = 2.0 * np.pi * self._frequency
        acceleration = (
            self.striking * pressure_difference * self._inverse_mass
            - (omega / self._q) * self.velocity
            - omega**2 * (self.opening - self._rest_opening)
        )
        self.velocity += acceleration * dt
        self.opening += self.velocity * dt

        # The valve cannot open past a hard stop, nor close past shut.
        if self.opening < 0.0:
            self.opening = 0.0
            if self.velocity < 0.0:
                self.velocity = 0.0
        elif self.opening > 3.0 * self._rest_opening:
            self.opening = 3.0 * self._rest_opening
            if self.velocity > 0.0:
                self.velocity = 0.0

        return returning_wave + impedance * flow
