"""Trombolese bore geometry and its input impedance.

The instrument is a chain of three parts, from the player's lips outward:

    mouthpiece -> bore -> bell

and a single parameter ``alpha`` morphs it continuously between a trombone-like
and an oboe-like acoustic regime.

The morph
---------
``alpha = 0`` is the trombone limit: a brass mouthpiece cup feeding a
cylindrical bore of 6.95 mm radius, flaring into the bell.

``alpha = 1`` is the conical limit: a narrow reed staple feeding a bore that
opens from 2 mm to 15 mm before the bell.

Every radius interpolates linearly between those two instruments. Because the
bore's blend is between a constant profile and a linear one, the blend is
itself exactly linear -- a single conical frustum at any ``alpha``, computed
exactly rather than sliced.

Two things this geometry got wrong on the first pass
----------------------------------------------------
Both were found by sweeping the model and noticing the resonances did not move
the way the textbook says they should. They are recorded here because they are
design constraints on the instrument, not incidental bugs:

1. **The throat cannot carry a long cylindrical section.** An earlier version
   kept a 0.55 m cylindrical slide, at the throat radius, ahead of the taper.
   A truncated cone only behaves like a complete one if whatever replaces its
   missing apex has roughly the volume of that apex; 0.55 m of narrow tube has
   several times too much, and behaves as a transmission line rather than a
   compliance besides. The result was an instrument whose upper resonances
   ignored the morph entirely -- only the fundamental moved. The slide is now
   part of the taper itself.

2. **A cone cannot reach the bell at a trombone's bore radius.** Opening from
   2 mm to 6.95 mm over 2.15 m truncates the cone at 29% of its apex distance,
   which is far too stubby to produce a harmonic series: the regimes come out
   at 1, 2.21, 3.51 rather than 1, 2, 3. Reaching 15 mm instead puts the
   truncation at 13% and the regimes at 1, 2.05, 3.14. So the conical limit of
   this instrument necessarily has a wider bore than its cylindrical limit --
   nearer a euphonium than a trombone. The bell's *mouth* stays fixed at
   108 mm so that the radiating aperture, and hence the radiation impedance,
   is constant across the morph; only the bell's entry follows the bore.

Why this is the parameter worth exposing to a player
----------------------------------------------------
A cylinder closed at the driven end resonates on odd harmonics and overblows
at the twelfth; a cone resonates on the complete harmonic series and overblows
at the octave. Sweeping ``alpha`` does not merely change timbre -- it changes
which pitches the instrument *can* sound for a given lip tension, and moves
the regimes continuously between the two families. No physical wind instrument
can do that, which is the whole reason for building this one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .acoustics import (
    chain,
    cone_matrix,
    cylinder_matrix,
    input_impedance,
    radiation_impedance,
)
from .constants import AIR_20C, Air

__all__ = ["Trombolese", "BoreResponse", "Segment"]


def _lerp(alpha: float, at_zero: float, at_one: float) -> float:
    return (1.0 - alpha) * at_zero + alpha * at_one


@dataclass(frozen=True)
class Segment:
    """One conical frustum of the bore. ``radius_in == radius_out`` is a cylinder."""

    name: str
    radius_in: float
    radius_out: float
    length: float

    @property
    def is_cylindrical(self) -> bool:
        return self.radius_in == self.radius_out


@dataclass
class BoreResponse:
    """Input impedance of a bore over a frequency grid."""

    freqs: np.ndarray
    impedance: np.ndarray
    alpha: float
    slide: float

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.impedance)

    @property
    def magnitude_db(self) -> np.ndarray:
        return 20.0 * np.log10(np.abs(self.impedance))


@dataclass
class Trombolese:
    """Geometry of the Trombolese, in metres.

    Defaults put the cylindrical limit close to a tenor trombone -- a 13.9 mm
    bore, a 216 mm bell, about 2.8 m of tubing, so a B-flat instrument -- and
    the conical limit at the widest taper that still shares that bell.

    Parameters
    ----------
    bore_length:
        Length of the morphable taper, before any slide extension.
    cyl_throat_radius, cyl_bell_entry_radius:
        The bore's two end radii at ``alpha = 0``. Equal, so the bore is a
        cylinder there.
    cone_throat_radius, cone_bell_entry_radius:
        The bore's two end radii at ``alpha = 1``. See the module docstring for
        why the conical limit has to open much wider than the cylindrical one.
    bell_length, bell_radius, bell_flare:
        The bell is a power-law horn from the bore's bell-entry radius out to
        ``bell_radius``, which stays fixed across the morph. ``bell_flare = 1``
        is a plain cone; larger values concentrate the flare near the mouth, as
        a real brass bell does.
    bell_segments:
        Number of conical frusta approximating the bell. Resonance frequencies
        are converged to under a tenth of a cent by 48 slices; going to 320
        moves nothing audible.
    cup_radius, cup_length, cup_throat_radius, cup_throat_length:
        A deliberately crude two-cylinder brass mouthpiece at ``alpha = 0``: a
        cup of roughly 10 ml followed by a narrow throat. A real mouthpiece is
        what pulls a brass instrument's stretched modes into an almost-harmonic
        series, so omitting it would misrepresent the trombone limit. Two
        cylinders capture the broad effect and not the detail -- see the
        README's limitations section.
    staple_radius, staple_length:
        What the mouthpiece becomes at ``alpha = 1``: a narrow reed staple.
    """

    bore_length: float = 2.15
    cyl_throat_radius: float = 0.00695
    cyl_bell_entry_radius: float = 0.00695
    cone_throat_radius: float = 0.00200
    cone_bell_entry_radius: float = 0.01500

    bell_length: float = 0.60
    bell_radius: float = 0.1080
    bell_flare: float = 3.5
    bell_segments: int = 48

    cup_radius: float = 0.01250
    cup_length: float = 0.02000
    cup_throat_radius: float = 0.00350
    cup_throat_length: float = 0.02500

    staple_radius: float = 0.00230
    staple_length: float = 0.02500

    include_mouthpiece: bool = True

    air: Air = field(default_factory=lambda: AIR_20C)

    # -- geometry ---------------------------------------------------------

    def throat_radius(self, alpha: float) -> float:
        """Radius at the bore's mouthpiece end."""
        self._check_alpha(alpha)
        return _lerp(alpha, self.cyl_throat_radius, self.cone_throat_radius)

    def bell_entry_radius(self, alpha: float) -> float:
        """Radius where the bore meets the bell."""
        self._check_alpha(alpha)
        return _lerp(alpha, self.cyl_bell_entry_radius, self.cone_bell_entry_radius)

    def truncation_ratio(self, alpha: float, slide: float = 0.0) -> float:
        """How stubby the cone is: apex distance over total distance to the bell.

        0 would be a complete cone (apex at the lips) and gives an exactly
        harmonic series; values above roughly 0.2 are too truncated to produce
        one. Returns NaN at ``alpha = 0``, where the bore is cylindrical and
        has no apex.
        """
        throat = self.throat_radius(alpha)
        entry = self.bell_entry_radius(alpha)
        if entry <= throat:
            return float("nan")
        length = self.bore_length + slide
        apex = throat * length / (entry - throat)
        return apex / (apex + length)

    def bell_radii(self, alpha: float = 0.0) -> np.ndarray:
        """Radii at the boundaries of the bell's conical slices."""
        entry = self.bell_entry_radius(alpha)
        s = np.linspace(0.0, 1.0, self.bell_segments + 1)
        ratio = self.bell_radius / entry
        return entry * (1.0 + (ratio ** (1.0 / self.bell_flare) - 1.0) * s) ** self.bell_flare

    def segments(self, alpha: float = 0.0, slide: float = 0.0) -> list[Segment]:
        """The bore as an ordered list of frusta, mouthpiece end first.

        This is the single source of truth for the geometry: :meth:`profile`
        and :meth:`sweep` are both built from it.

        A slide extension lengthens the taper rather than inserting cylindrical
        tubing at its throat. On a real trombone the slide has to be a
        cylinder, but inserting one here would reintroduce exactly the problem
        described in the module docstring; stretching the taper keeps the bore
        a single clean frustum at every slide position, so the slide changes
        pitch without disturbing the mode structure.
        """
        self._check_alpha(alpha)
        if slide < 0.0:
            raise ValueError("slide extension must be non-negative")

        parts: list[Segment] = []

        if self.include_mouthpiece:
            # The cup collapses toward a reed staple as the bore turns conical.
            cup_r = _lerp(alpha, self.cup_radius, self.staple_radius)
            cup_l = _lerp(alpha, self.cup_length, self.staple_length)
            throat_r = _lerp(alpha, self.cup_throat_radius, self.cone_throat_radius)
            parts.append(Segment("cup", cup_r, cup_r, cup_l))
            parts.append(
                Segment("cup_throat", throat_r, throat_r, self.cup_throat_length)
            )

        parts.append(
            Segment(
                "bore",
                self.throat_radius(alpha),
                self.bell_entry_radius(alpha),
                self.bore_length + slide,
            )
        )

        radii = self.bell_radii(alpha)
        step = self.bell_length / self.bell_segments
        parts.extend(
            Segment("bell", float(r_in), float(r_out), step)
            for r_in, r_out in zip(radii[:-1], radii[1:])
        )
        return parts

    def profile(
        self, alpha: float = 0.0, slide: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bore radius against axial distance from the lips, for plotting.

        Step changes in radius appear as repeated x values.
        """
        xs: list[float] = []
        radii: list[float] = []
        x = 0.0
        for seg in self.segments(alpha, slide):
            xs.extend([x, x + seg.length])
            radii.extend([seg.radius_in, seg.radius_out])
            x += seg.length
        return np.asarray(xs), np.asarray(radii)

    def total_length(self, alpha: float = 0.0, slide: float = 0.0) -> float:
        """Total acoustic length from lips to mouth."""
        return float(sum(seg.length for seg in self.segments(alpha, slide)))

    # -- acoustics --------------------------------------------------------

    def _segment_matrix(self, freqs: np.ndarray, seg: Segment) -> np.ndarray:
        """Transfer matrix of one segment."""
        if seg.is_cylindrical:
            return cylinder_matrix(freqs, seg.radius_in, seg.length, self.air)
        return cone_matrix(freqs, seg.radius_in, seg.radius_out, seg.length, self.air)

    def response(
        self,
        freqs: np.ndarray,
        alpha: float = 0.0,
        slide: float = 0.0,
        ideal_open_end: bool = False,
    ) -> BoreResponse:
        """Input impedance seen by the reed, over ``freqs``.

        Peaks of ``|Z_in|`` are the instrument's playing regimes: both a lip
        reed and a double reed are pressure-controlled valves behaving close
        to a flow source, so they cooperate with maxima of input impedance.
        This is why the analysis tracks impedance peaks rather than
        transmission.

        Parameters
        ----------
        ideal_open_end:
            Terminate in zero impedance instead of a radiation impedance.
            Unphysical, but it reproduces textbook results exactly and so is
            useful in tests.
        """
        return self.sweep(freqs, (alpha,), slide, ideal_open_end)[0]

    def sweep(
        self,
        freqs: np.ndarray,
        alphas: Sequence[float],
        slide: float = 0.0,
        ideal_open_end: bool = False,
    ) -> list[BoreResponse]:
        """Input impedance at several morph positions.

        Equivalent to calling :meth:`response` once per value of ``alpha``,
        but it builds the frequency-dependent terminating impedance once.
        """
        freqs = np.asarray(freqs, dtype=float)
        load = (
            np.zeros_like(freqs, dtype=complex)
            if ideal_open_end
            else radiation_impedance(freqs, self.bell_radius, self.air)
        )

        responses: list[BoreResponse] = []
        for alpha in alphas:
            matrices = [
                self._segment_matrix(freqs, seg)
                for seg in self.segments(float(alpha), slide)
            ]
            responses.append(
                BoreResponse(
                    freqs=freqs,
                    impedance=input_impedance(chain(matrices), load),
                    alpha=float(alpha),
                    slide=slide,
                )
            )
        return responses

    @staticmethod
    def _check_alpha(alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
