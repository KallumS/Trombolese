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
``vent``
    The register vent, shut to open. A woodwind's octave key, except that this
    one travels along the bore as the morph changes where the pressure node is.

``alpha`` and ``beta`` are independent, so a double reed can be put on a
cylindrical bore or brass lips on a conical one -- pairings no instrument
family has ever had to make a decision about.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .bore import Trombolese
from .reed import Reed, ReedParameters
from .waveguide import SectionedBore, Waveguide, section_bore

__all__ = ["Controls", "Voice", "ReedCompensation", "compensate_reed_morph"]


@dataclass
class Controls:
    """The instrument's five continuous inputs, plus the blowing pressure."""

    pressure: float = 0.0
    lip_frequency: float = 60.0
    slide: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    vent: float = 0.0


@dataclass(frozen=True)
class ReedCompensation:
    """Embouchure correction that stops the reed morph dragging the pitch.

    ``beta`` changes the valve's natural frequency, because the embouchure
    control is deliberately kept live at both ends of the morph -- pinning the
    double-reed end to an absolute frequency would leave the player no register
    control at all. The cost is that sweeping ``beta`` alone walks the pitch,
    by well over an octave across the full range.

    The correction has two parts, because the drift has two causes.

    **Which regime speaks** is set by the embouchure, so that part is corrected
    by scaling it. Dividing by the valve's frequency ratio removes most of the
    error, and the rest is measured rather than derived: what remains is not a
    smooth detuning but steps between regimes, and only playing the instrument
    reveals where they fall.

    **Where inside that regime the note settles** cannot be corrected by
    embouchure at all. An outward-striking valve sounds a bore resonance sharp
    and an inward-striking one sounds it flat -- on this instrument, 217 cents
    apart on the same regime -- because that is what the striking sign does.
    Only a change of length moves it, so the second part of the correction is a
    slide trim, and it goes negative: the reed end has to be shortened to meet
    the pitch the lips were sounding.

    Both parts are tabulated over ``alpha`` as well as ``beta``. That is not
    caution: a correction calibrated at one bore shape and applied at another
    is worse than none, missing by over an octave, because which regime the
    reed grabs depends on the bore it is driving. The two morphs are not
    separable and the correction is a surface, not a curve.

    Built by :func:`compensate_reed_morph`.
    """

    alphas: np.ndarray
    betas: np.ndarray
    multipliers: np.ndarray
    slide_offsets: np.ndarray
    target_frequency: float

    #: Two entries further apart than this (as a ratio) are taken to have
    #: chosen different regimes. Within a regime the correction varies by a few
    #: percent; the smallest step between neighbouring regimes on this
    #: instrument is about 30%.
    PLATEAU_TOLERANCE: ClassVar[float] = 1.12

    def multiplier_at(self, alpha: float, beta: float) -> float:
        """Embouchure scaling, which keeps the same regime selected.

        **Interpolated within a plateau, snapped across a cliff.**

        This entry's job is to pick a regime, and a regime is discrete: between
        two grid points that chose different ones there is no meaningful value
        in between, and blending them lands on a third and misses by an octave.
        A table whose every grid point was within 26 cents produced 1200-cent
        errors at the points between them for exactly this reason.

        But snapping everywhere is not right either. Most of this table is a
        smooth plateau where the correction really is continuous, and snapping
        there steps the pitch audibly as a control sweeps. So the four
        bracketing entries are inspected: if they agree to within
        ``PLATEAU_TOLERANCE`` they are one regime's worth of correction and get
        interpolated, and if they do not, the choice is a real one and the
        nearest is taken.
        """
        return self._look_up(self.multipliers, alpha, beta, snap_across_cliffs=True)

    def _bracket(self, grid: np.ndarray, value: float) -> tuple[int, int, float]:
        """Indices either side of ``value`` in ``grid``, and the fraction between."""
        if len(grid) == 1:
            return 0, 0, 0.0
        upper = int(np.clip(np.searchsorted(grid, value), 1, len(grid) - 1))
        lower = upper - 1
        span = grid[upper] - grid[lower]
        fraction = 0.0 if span == 0 else float((value - grid[lower]) / span)
        return lower, upper, float(np.clip(fraction, 0.0, 1.0))

    def _look_up(self, table: np.ndarray, alpha: float, beta: float,
                 snap_across_cliffs: bool = False) -> float:
        row_lo, row_hi, row_f = self._bracket(self.alphas, alpha)
        col_lo, col_hi, col_f = self._bracket(self.betas, beta)

        corners = np.array([
            table[row_lo, col_lo], table[row_lo, col_hi],
            table[row_hi, col_lo], table[row_hi, col_hi],
        ])

        if snap_across_cliffs:
            smallest = float(np.min(np.abs(corners)))
            largest = float(np.max(np.abs(corners)))
            if smallest <= 0.0 or largest / smallest > self.PLATEAU_TOLERANCE:
                row = row_hi if row_f > 0.5 else row_lo
                column = col_hi if col_f > 0.5 else col_lo
                return float(table[row, column])

        lower = corners[0] + col_f * (corners[1] - corners[0])
        upper = corners[2] + col_f * (corners[3] - corners[2])
        return float(lower + row_f * (upper - lower))

    def slide_at(self, alpha: float, beta: float) -> float:
        """Length trim, which absorbs the reed's pull on that regime.

        Always bilinear: once the regime is fixed, pitch varies smoothly with
        length, so interpolating here is not only safe but what keeps the
        correction from stepping audibly between grid points.
        """
        return self._look_up(self.slide_offsets, alpha, beta)


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

    #: Applied to the embouchure so that morphing the reed does not move the
    #: pitch. Build one with :func:`compensate_reed_morph`.
    reed_compensation: ReedCompensation | None = None

    def __post_init__(self) -> None:
        self._bore = self._section(0.0, 0.0)
        self._guide = Waveguide(self._bore)
        self._reed = Reed(self.sample_rate, self.reed_parameters,
                          self.instrument.air)
        self._dc_state = 0.0
        self._previous_radiated = 0.0
        self._geometry_key: tuple[float, float, float] | None = None

    def _section(self, alpha: float, slide: float,
                 vent: float = 0.0) -> SectionedBore:
        return section_bore(
            self.instrument, alpha, slide, self.sample_rate,
            dispersion_reference_hz=self.dispersion_reference_hz,
            vent_opening=vent,
        )

    def _apply_geometry(self, alpha: float, slide: float, vent: float) -> None:
        """Re-section the bore, but only when the controls have actually moved."""
        key = (round(alpha, 4), round(slide, 5), round(vent, 3))
        if key == self._geometry_key:
            return
        self._geometry_key = key
        self._bore = self._section(alpha, slide, vent)
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

            embouchure = controls.lip_frequency
            slide = controls.slide
            if self.reed_compensation is not None:
                embouchure *= self.reed_compensation.multiplier_at(
                    controls.alpha, controls.beta
                )
                slide += self.reed_compensation.slide_at(
                    controls.alpha, controls.beta
                )

            self._apply_geometry(controls.alpha, slide, controls.vent)
            impedance = self._bore.characteristic_impedance(0)
            self._reed.set_morph(controls.beta, embouchure)

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


def _sounding_frequency(
    voice: "Voice",
    controls: Controls,
    seconds: float = 0.8,
    silence: float = 1e-3,
) -> float:
    """Play a note and report the frequency it settled on.

    The default is long for a reason. Some notes sound one regime for the first
    third of a second and then jump to the octave above and stay there, so a
    shorter probe measures a transient and calls it the pitch. A calibration
    built on those measurements looks perfect at its own grid points and is an
    octave out when the note is actually held.
    """
    voice.reset()
    output = voice.render(int(seconds * voice.sample_rate), controls)
    tail = output[-8192:]
    if np.sqrt(np.mean(tail**2)) < silence:
        return float("nan")
    spectrum = np.abs(np.fft.rfft(tail * np.hanning(len(tail))))
    freqs = np.fft.rfftfreq(len(tail), 1.0 / voice.sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


def compensate_reed_morph(
    instrument,
    embouchure: float = 176.0,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    pressure: float = 4200.0,
    n_beta: int = 9,
    n_candidates: int = 19,
    span: float = 3.2,
    seconds: float = 0.3,
    sample_rate: float = 48_000.0,
    reed_parameters: ReedParameters | None = None,
) -> ReedCompensation:
    """Measure the correction that holds pitch across the reed morph.

    For each ``(alpha, beta)`` this plays the instrument at a spread of
    embouchures and keeps the one landing nearest the target, then trims the
    length to remove whatever the embouchure could not reach.

    The embouchure search is a scan rather than a root-find, deliberately: the
    sounding pitch is not a continuous function of the embouchure, because the
    note steps from one regime to the next, and a bisection would happily
    converge on the wrong side of a step. The length trim afterwards *is*
    solved directly, because within a regime frequency does go smoothly as one
    over length.

    Calibration takes a few minutes. It is an offline step whose result is a
    small table.
    """
    parameters = reed_parameters or ReedParameters()
    probe = Voice(
        instrument, sample_rate=sample_rate, reed_parameters=parameters,
        reed_compensation=None,
    )

    alphas = np.asarray(alphas, dtype=float)
    betas = np.linspace(0.0, 1.0, n_beta)
    multipliers = np.ones((len(alphas), n_beta))
    slide_offsets = np.zeros((len(alphas), n_beta))

    reference_alpha = float(alphas[len(alphas) // 2])
    target = _sounding_frequency(
        probe, Controls(pressure=pressure, lip_frequency=embouchure,
                        alpha=reference_alpha, beta=0.0),
        seconds=seconds,
    )
    if not np.isfinite(target):
        raise ValueError("the instrument did not speak at beta = 0")

    for row, alpha in enumerate(alphas):
        nominal = instrument.bore_length_at(float(alpha))

        for column, beta in enumerate(betas):
            analytic = 1.0 / (
                1.0 + float(beta) * (parameters.reed_frequency_ratio - 1.0)
            )
            candidates = analytic * np.geomspace(1.0 / span, span, n_candidates)

            errors = np.full(len(candidates), np.inf)
            for index, multiplier in enumerate(candidates):
                sounded = _sounding_frequency(
                    probe,
                    Controls(pressure=pressure,
                             lip_frequency=embouchure * multiplier,
                             alpha=float(alpha), beta=float(beta)),
                    seconds=seconds,
                )
                if np.isfinite(sounded):
                    errors[index] = abs(1200.0 * np.log2(sounded / target))

            if not np.any(np.isfinite(errors)):
                multipliers[row, column] = analytic
                continue

            # Several embouchures can land equally near the target, on
            # different regimes. Picking the outright winner each time makes
            # the table jump between them, and the damage shows up not at the
            # grid points -- each individually fine -- but *between* them,
            # where interpolating across such a jump lands on a third regime
            # and misses by an octave. So among those doing essentially as
            # well, take the one nearest its already-solved neighbours.
            #
            # Preferring the middle of a run of acceptable candidates, rather
            # than the one nearest the neighbours, was tried on the theory that
            # an edge candidate sits against a regime boundary. It verified
            # worse (three off-grid failures against one) and was dropped.
            acceptable = errors <= float(np.min(errors)) + 25.0
            contenders = candidates[acceptable]

            neighbours = []
            if column > 0:
                neighbours.append(multipliers[row, column - 1])
            if row > 0:
                neighbours.append(multipliers[row - 1, column])
            anchor = (
                float(np.exp(np.mean(np.log(neighbours)))) if neighbours
                else analytic
            )
            multipliers[row, column] = float(
                min(contenders, key=lambda m: abs(np.log(m / anchor)))
            )

            # Whatever the embouchure could not reach is the reed's pull on the
            # regime. Frequency goes as one over length, so the trim follows in
            # closed form; one refinement covers the nonlinearity.
            for _ in range(2):
                sounded = _sounding_frequency(
                    probe,
                    Controls(pressure=pressure,
                             lip_frequency=embouchure * multipliers[row, column],
                             alpha=float(alpha), beta=float(beta),
                             slide=float(slide_offsets[row, column])),
                    seconds=seconds,
                )
                if not np.isfinite(sounded):
                    break
                residual = 1200.0 * np.log2(sounded / target)
                length = nominal + slide_offsets[row, column]
                slide_offsets[row, column] = float(np.clip(
                    length * 2.0 ** (residual / 1200.0) - nominal,
                    -0.6 * nominal, 1.5 * nominal,
                ))

    return ReedCompensation(
        alphas=alphas, betas=betas, multipliers=multipliers,
        slide_offsets=slide_offsets, target_frequency=float(target),
    )
