"""The excitation: a reed that morphs from brass lips to a double reed.

A wind instrument is a resonator plus a pressure-controlled valve, and the two
parents of this instrument disagree about which kind of valve. Brass lips are
**outward-striking**: raising the pressure difference across them pushes them
apart, so they blow *open*. A double reed is **inward-striking**: the same
pressure difference pushes the blades together, so it blows *closed*. Nearly
everything that distinguishes how the two families speak, slur and crack
follows from that single sign.

So the excitation morph is, at its heart, that sign made continuous. The
obvious way to do it -- run one valve's coupling coefficient from +1 to -1 --
turns out to be a trap, and the model said so plainly: at the midpoint the
coefficient is zero, the valve stops responding to pressure altogether, and
the embouchure loses all authority over the instrument. Measured, the pitch
sat 324 cents off target at ``beta = 0.5`` and could not be lipped back by any
embouchure whatsoever, while every other position on the morph came within
about 50 cents.

So the morph is instead a **crossfade between two valves**, one blowing open
and one blowing closed, each keeping its coupling at full strength. Their
openings are blended into a single effective aperture, so the middle of the
morph is a valve that is half-lip and half-reed and answers vigorously to
pressure in both characters at once, rather than a valve that answers to
nothing. Alongside the blend, each valve's resonance, damping, aperture and
stiffness interpolate between plausible values for the two instruments.

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
        How far above the embouchure control the double reed sits. A double
        reed resonates well above the regimes it drives, but it must still
        answer to the player: biting raises a reed's effective stiffness and
        favours the upper register, just as tightening the lips does on brass.
        Keeping the valve frequency proportional to the embouchure at *both*
        ends is what preserves register control across the whole morph --
        pinning the reed end to an absolute frequency leaves the player with no
        pitch control whatsoever, which the model duly demonstrated.
    reed_frequency_ratio:
        See :meth:`Reed.set_morph`.
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


class _Valve:
    """One pressure-controlled valve: a damped oscillator with a hard stop."""

    __slots__ = ("striking", "frequency", "q", "width", "rest_opening",
                 "inverse_mass", "opening", "velocity")

    def __init__(self, striking: float) -> None:
        self.striking = striking
        self.frequency = 100.0
        self.q = 10.0
        self.width = 0.010
        self.rest_opening = 3.0e-4
        self.inverse_mass = 0.0
        self.opening = self.rest_opening
        self.velocity = 0.0

    def tune(self, frequency: float, q: float, width: float,
             rest_opening: float, closing_pressure: float) -> None:
        self.frequency = frequency
        self.q = q
        self.width = width
        self.rest_opening = rest_opening
        omega = 2.0 * np.pi * frequency
        # mu follows from "the pressure that just closes the valve":
        # closing = mu * omega^2 * rest_opening.
        self.inverse_mass = (omega**2 * rest_opening) / closing_pressure

    def reset(self) -> None:
        self.opening = self.rest_opening
        self.velocity = 0.0

    @property
    def area(self) -> float:
        return self.width * max(self.opening, 0.0)

    def advance(self, pressure_difference: float, dt: float) -> None:
        omega = 2.0 * np.pi * self.frequency
        acceleration = (
            self.striking * pressure_difference * self.inverse_mass
            - (omega / self.q) * self.velocity
            - omega**2 * (self.opening - self.rest_opening)
        )
        self.velocity += acceleration * dt
        self.opening += self.velocity * dt

        if self.opening < 0.0:
            self.opening = 0.0
            if self.velocity < 0.0:
                self.velocity = 0.0
        elif self.opening > 3.0 * self.rest_opening:
            self.opening = 3.0 * self.rest_opening
            if self.velocity > 0.0:
                self.velocity = 0.0


class Reed:
    """A morphable pressure-controlled valve driving a waveguide.

    Holds two valves -- one blowing open, one blowing closed -- and crossfades
    between them. Call :meth:`set_morph` at control rate and :meth:`step` once
    per sample.
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

        self.lip = _Valve(striking=1.0)     # blows open, as brass lips do
        self.double = _Valve(striking=-1.0)  # blows closed, as a double reed does
        self.beta = 0.0
        self.set_morph(0.0)
        self.reset()

    def reset(self) -> None:
        self.lip.reset()
        self.double.reset()

    @property
    def opening(self) -> float:
        """Effective opening: the blend the bore actually sees."""
        return (1.0 - self.beta) * self.lip.opening + self.beta * self.double.opening

    @property
    def effective_area(self) -> float:
        """Blended aperture of the two valves."""
        return (1.0 - self.beta) * self.lip.area + self.beta * self.double.area

    def set_morph(self, beta: float, frequency: float | None = None) -> None:
        """Crossfade the valve from lips (``beta = 0``) to double reed (``1``).

        ``frequency`` overrides the embouchure, which is how a player selects a
        regime. It stays live at both ends: the double reed's own resonance is
        set as a multiple of it rather than as an absolute, because pinning it
        would leave the player no register control at the conical end.
        """
        beta = float(np.clip(beta, 0.0, 1.0))
        self.beta = beta
        p = self.parameters

        embouchure = p.lip_frequency if frequency is None else frequency
        self.lip.tune(
            embouchure, p.lip_q, p.lip_width, p.lip_rest_opening,
            p.lip_closing_pressure,
        )
        self.double.tune(
            embouchure * p.reed_frequency_ratio, p.reed_q, p.reed_width,
            p.reed_rest_opening, p.reed_closing_pressure,
        )

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

        # Both valves see the same pressure difference and vent into the same
        # bore, so they behave as one opening of the blended area -- which keeps
        # the coupled solve a single quadratic rather than a pair of them.
        area = self.effective_area
        if area > 0.0:
            k = 2.0 * area**2 / self.air.density
            discriminant = (k * impedance) ** 2 + 4.0 * k * abs(incident)
            flow = 0.5 * (-k * impedance + np.sqrt(discriminant))
            flow = np.copysign(flow, incident)
        else:
            flow = 0.0

        pressure_difference = incident - impedance * flow

        dt = 1.0 / self.sample_rate
        self.lip.advance(pressure_difference, dt)
        self.double.advance(pressure_difference, dt)

        return returning_wave + impedance * flow
