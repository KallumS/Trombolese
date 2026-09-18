"""Digital waveguide of the Trombolese bore.

Stage 2's resonator. The bore is sliced into cylindrical sections one sample
of propagation long, and pressure waves travel along them in both directions,
scattering wherever the cross-section changes. This is the Kelly-Lochbaum
ladder, and it is the natural home for the bore morph: changing ``alpha``
changes nothing structural, only the reflection coefficient at each junction.

Why a scattering ladder rather than a single delay loop
------------------------------------------------------
Most waveguide brass models use one delay-line pair with a lumped reflection
filter standing in for the whole bell. That is far cheaper, and it is the right
choice when the bore is fixed -- but it hides the bore shape inside a filter
fit, which is exactly the thing this instrument needs to vary continuously. A
ladder keeps the geometry explicit: the shape *is* the coefficient vector, so
morphing the bore is a control-rate recomputation of that vector and nothing
else.

Relationship to stage 1
-----------------------
The transfer-matrix model in :mod:`trombolese.bore` is the reference. This
module must reproduce its resonance frequencies from the same geometry, and
:func:`measure_input_impedance` exists to check exactly that: it probes the
ladder with a flow impulse and recovers the input impedance the same way the
frequency-domain model computes it. The tests hold the two within a few cents.

Scattering convention
---------------------
At a junction between a section of area ``S1`` and the next of area ``S2``,
with ``p+`` arriving from the left and ``p-`` from the right, continuity of
pressure and flow gives

    r  = (S1 - S2) / (S1 + S2)
    d  = r * (p+ - p-)
    p+ out = p+ + d        (into the next section)
    p- out = p- + d        (back into the previous one)

which is the one-multiply form of the junction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bore import Trombolese
from .constants import Air

__all__ = [
    "SectionedBore",
    "Waveguide",
    "section_bore",
    "measure_input_impedance",
    "regime_of",
    "tune_to_waveguide",
]


@dataclass(frozen=True)
class SectionedBore:
    """A bore resampled onto a uniform grid of one-sample-long sections."""

    radii: np.ndarray
    reflections: np.ndarray
    losses: np.ndarray
    dispersion: np.ndarray
    vent_index: int | None
    vent_cutoff: float
    section_length: float
    sample_rate: float
    air: Air

    @property
    def n_sections(self) -> int:
        return len(self.radii)

    @property
    def length(self) -> float:
        return self.n_sections * self.section_length

    @property
    def areas(self) -> np.ndarray:
        return np.pi * self.radii**2

    def characteristic_impedance(self, index: int = 0) -> float:
        """Wave impedance of one section, in acoustic ohms."""
        return float(
            self.air.density * self.air.speed_of_sound / self.areas[index]
        )


def section_bore(
    instrument: Trombolese,
    alpha: float = 0.0,
    slide: float = 0.0,
    sample_rate: float = 48_000.0,
    loss_reference_hz: float = 250.0,
    dispersion_reference_hz: float | None = None,
    vent_opening: float = 0.0,
) -> SectionedBore:
    """Resample an instrument's bore onto the waveguide's section grid.

    Each section is one sample of propagation long, ``c / fs`` -- about 7.2 mm
    at 48 kHz -- so the section count follows the instrument's length and grows
    as the pitch-compensated bore lengthens.

    Radii are sampled at section midpoints from the exact geometry, so the
    conical parts stay conical rather than inheriting the bell's slicing.

    Losses
    ------
    Viscothermal wall loss varies as the square root of frequency, which a
    single per-section scalar cannot represent. Since each section is only
    millimetres long its loss is tiny (order 1e-4), and the audible effect is
    the *accumulated* round-trip loss. This applies a frequency-independent
    per-section gain evaluated at ``loss_reference_hz``, which makes the model
    slightly too bright at the top of the range and slightly too damped at the
    bottom.

    Dispersion
    ----------
    Boundary-layer loss does not only attenuate -- it slows the wave. Stage 1's
    complex wavenumber has ``Re(k) = omega / c + a``, so waves travel a few
    percent slower than ``c``, and every resonance sits correspondingly flat. A
    plain ladder propagates at exactly ``c`` and so plays **sharp by some 65
    cents at the fundamental** -- a tuning error no player would accept, and by
    far the largest discrepancy between the two models.

    The correction has to be **distributed, not lumped**. Folding it into the
    termination does nothing for the low regimes, because at low frequency the
    bell flare reflects the wave long before it reaches the mouth -- that is
    what a bell cutoff is -- so a delay at the mouth is never traversed. The
    excess delay belongs where the wave actually travels: in every section.

    Each section therefore gets a one-pole whose group delay is the excess
    delay that section owes, ``a_i dx fs / omega``, evaluated at
    ``dispersion_reference_hz``. Well below Nyquist a one-pole's group delay is
    essentially flat at ``g / (1 - g)``, so this reproduces the slowing without
    reproducing its ``1 / sqrt(f)`` shape: exact at the reference frequency,
    progressively too slow above it. Referencing near the second regime
    balances the residual across the playing range rather than nailing the
    pedal note and leaving everything above it flat. Matching the ``sqrt`` law
    properly needs a fitted fractional-order filter, which is the obvious next
    refinement.
    """
    c = instrument.air.speed_of_sound
    section_length = c / sample_rate

    total = instrument.total_length(alpha, slide)
    n_sections = max(int(round(total / section_length)), 2)

    # Sample the geometry at section midpoints.
    x_profile, r_profile = instrument.profile(alpha, slide)
    midpoints = (np.arange(n_sections) + 0.5) * section_length
    radii = np.interp(midpoints, x_profile, r_profile)

    areas = np.pi * radii**2
    reflections = (areas[:-1] - areas[1:]) / (areas[:-1] + areas[1:])

    # Per-section amplitude loss, exp(-alpha_wall * dx), at the reference
    # frequency and the local radius.
    air = instrument.air
    omega = 2.0 * np.pi * loss_reference_hz
    boundary_layer = np.sqrt(air.kinematic_viscosity) + (
        air.heat_capacity_ratio - 1.0
    ) * np.sqrt(air.kinematic_viscosity / air.prandtl_number)
    attenuation = np.sqrt(omega / 2.0) * boundary_layer / (radii * c)
    losses = np.exp(-attenuation * section_length)

    # Per-section excess delay, in samples, from the slowed wave; converted to
    # the one-pole coefficient with that group delay.
    if dispersion_reference_hz is None:
        dispersion_reference_hz = loss_reference_hz
    omega_ref = 2.0 * np.pi * dispersion_reference_hz
    attenuation_ref = np.sqrt(omega_ref / 2.0) * boundary_layer / (radii * c)
    excess = attenuation_ref * section_length * sample_rate / omega_ref
    dispersion = excess / (1.0 + excess)

    vent_index, vent_cutoff = _vent_coefficients(
        instrument, alpha, slide, radii, section_length, n_sections, vent_opening
    )

    return SectionedBore(
        radii=radii,
        reflections=reflections,
        losses=losses,
        dispersion=dispersion,
        vent_index=vent_index,
        vent_cutoff=vent_cutoff,
        section_length=section_length,
        sample_rate=sample_rate,
        air=air,
    )


def _vent_coefficients(
    instrument: Trombolese,
    alpha: float,
    slide: float,
    radii: np.ndarray,
    section_length: float,
    n_sections: int,
    opening: float,
) -> tuple[int | None, float]:
    """Locate the register vent in the ladder and find its corner frequency.

    A side hole shunts the bore through the inertance of the air in it,
    ``Z_h = j omega rho t_e / S_h``. Solving the three-port junction with that
    shunt gives, for the junction pressure,

        p = a * j omega / (j omega + Zc / (2 L_h))

    -- a first-order **high-pass**, with ``a`` the sum of the two arriving
    waves. Which is the register hole's whole behaviour in one line: low
    frequencies are shorted to the outside and lost, high ones sail past. The
    corner is proportional to the open area, so a partly open vent simply moves
    it down, and a shut one puts it at zero, where the high-pass becomes a
    wire.
    """
    if instrument.vent is None or opening <= 0.0:
        return None, 0.0

    air = instrument.air
    lead_in = 0.0
    if instrument.include_mouthpiece:
        cup_length = (1.0 - alpha) * instrument.cup_length + alpha * instrument.staple_length
        lead_in = cup_length + instrument.cup_throat_length

    bore_length = instrument.bore_length_at(alpha) + slide
    distance = lead_in + bore_length * instrument.vent_position_at(alpha)
    index = int(round(distance / section_length))
    index = int(np.clip(index, 1, n_sections - 2))

    hole_area = opening * np.pi * instrument.vent.radius**2
    effective_height = instrument.vent.height + 1.5 * instrument.vent.radius
    inertance = air.density * effective_height / hole_area

    z_char = air.density * air.speed_of_sound / (np.pi * radii[index] ** 2)
    return index, float(z_char / (2.0 * inertance))


class Waveguide:
    """Bidirectional pressure-wave ladder with a radiating termination.

    The state is two delay lines, one per direction of travel, each holding one
    sample per section. :meth:`step` advances the whole ladder by one sample:
    it scatters at every junction at once, shifts, reflects at the mouth and
    accepts an injected wave at the mouthpiece.

    The termination models radiation from the bell mouth as a one-pole filter
    fitted to the true reflection coefficient ``(Zr - Zc) / (Zr + Zc)``: near
    -1 at low frequency, where the open end acts as a pressure release and
    almost everything comes back, falling toward 0 above the bell's cutoff,
    where the mouth radiates freely and little returns. That transition is what
    makes a bell a bell.

    The termination also carries a delay, which is not a detail. Radiation from
    an open end behaves as though the tube continued past it by an end
    correction of ``0.6133 a`` -- 66 mm for this bell, some 2.4% of the whole
    instrument. Leaving it out makes every resonance sharp by tens of cents.
    The reflection path is therefore delayed by ``2 * 0.6133 a`` (out and
    back), interpolated so the correction need not land on a whole sample.
    """

    def __init__(self, bore: SectionedBore) -> None:
        self.bore = bore
        self.forward = np.zeros(bore.n_sections)
        self.backward = np.zeros(bore.n_sections)
        self._mouth_state = 0.0
        self._reflection_gain, self._reflection_pole = _fit_mouth_reflection(bore)

        # Round-trip end correction, in samples. One sample is given back
        # because the ladder's own reflection step already costs one.
        mouth_radius = float(bore.radii[-1])
        correction = 2.0 * 0.6133 * mouth_radius / bore.section_length - 1.0
        self._delay_samples = max(correction, 0.0)
        self._delay = np.zeros(int(np.ceil(self._delay_samples)) + 2)
        self._write = 0

        # One-pole state per section, per direction of travel.
        self._dispersion_forward = np.zeros(bore.n_sections)
        self._dispersion_backward = np.zeros(bore.n_sections)

        self._vent_input = 0.0
        self._vent_output = 0.0

    def reset(self) -> None:
        self.forward[:] = 0.0
        self.backward[:] = 0.0
        self._mouth_state = 0.0
        self._delay[:] = 0.0
        self._write = 0
        self._dispersion_forward[:] = 0.0
        self._dispersion_backward[:] = 0.0
        self._vent_input = 0.0
        self._vent_output = 0.0

    def adopt(self, bore: SectionedBore) -> None:
        """Take on new geometry mid-note, carrying the wave state across.

        The instrument's length changes whenever the morph or the slide moves,
        and under pitch compensation the morph changes it a great deal -- the
        ladder grows from 391 sections to 636 between the two limits. Rebuilding
        it would silence the note, so instead the delay lines are resampled onto
        the new section count. Sweeping a control at audio rates changes the
        count by a section at a time, and stretching the contents by one section
        is a gentle operation; it is what lets the bore morph happen *while a
        note sustains*, which is the whole point of the instrument.

        The termination fit is reused: the bell mouth is the one radius the
        morph never touches.
        """
        old_n = self.bore.n_sections
        new_n = bore.n_sections

        if new_n != old_n:
            source = np.linspace(0.0, 1.0, old_n)
            destination = np.linspace(0.0, 1.0, new_n)
            self.forward = np.interp(destination, source, self.forward)
            self.backward = np.interp(destination, source, self.backward)
            self._dispersion_forward = np.interp(
                destination, source, self._dispersion_forward
            )
            self._dispersion_backward = np.interp(
                destination, source, self._dispersion_backward
            )

        self.bore = bore

    def _delayed(self, sample: float) -> float:
        """Push a sample into the end-correction delay and read it back out."""
        buffer = self._delay
        size = len(buffer)
        buffer[self._write] = sample

        read = self._write - self._delay_samples
        whole = int(np.floor(read))
        frac = read - whole
        earlier = buffer[whole % size]
        later = buffer[(whole + 1) % size]

        self._write = (self._write + 1) % size
        return float(earlier + frac * (later - earlier))

    @property
    def mouthpiece_pressure(self) -> float:
        """Total acoustic pressure at the mouthpiece end."""
        return float(self.forward[0] + self.backward[0])

    @property
    def mouthpiece_return(self) -> float:
        """The wave arriving back at the reed, which drives the excitation."""
        return float(self.backward[0])

    def step(self, injected: float) -> float:
        """Advance one sample. Returns the wave radiated from the mouth."""
        forward = self.forward
        backward = self.backward

        # Scatter at every junction simultaneously.
        delta = self.bore.reflections * (forward[:-1] - backward[1:])

        new_forward = np.empty_like(forward)
        new_backward = np.empty_like(backward)
        new_forward[1:] = forward[:-1] + delta
        new_backward[:-1] = backward[1:] + delta

        # Mouth: delay by the end correction, then a one-pole reflection.
        # What is not reflected radiates.
        arriving = forward[-1]
        delayed = self._delayed(arriving)
        self._mouth_state = (
            self._reflection_pole * self._mouth_state
            + (1.0 - self._reflection_pole) * delayed
        )
        reflected = -self._reflection_gain * self._mouth_state
        new_backward[-1] = reflected
        radiated = arriving + reflected

        # Mouthpiece: whatever the excitation injects.
        new_forward[0] = injected

        # Register vent: a high-pass three-port, replacing the plain junction
        # at that one section. The taper's own reflection there is a part in a
        # thousand and is simply given up in exchange.
        vent = self.bore.vent_index
        if vent is not None:
            arriving_right = forward[vent]
            arriving_left = backward[vent + 1]
            total = arriving_right + arriving_left
            pole = float(np.exp(-self.bore.vent_cutoff / self.bore.sample_rate))
            self._vent_output = pole * (
                self._vent_output + total - self._vent_input
            )
            self._vent_input = total
            new_forward[vent + 1] = self._vent_output - arriving_left
            new_backward[vent] = self._vent_output - arriving_right

        # Each section's one-pole: the wall drag that slows the wave.
        g = self.bore.dispersion
        self._dispersion_forward *= g
        self._dispersion_forward += (1.0 - g) * new_forward
        self._dispersion_backward *= g
        self._dispersion_backward += (1.0 - g) * new_backward
        new_forward = self._dispersion_forward.copy()
        new_backward = self._dispersion_backward.copy()

        np.multiply(new_forward, self.bore.losses, out=new_forward)
        np.multiply(new_backward, self.bore.losses, out=new_backward)

        self.forward = new_forward
        self.backward = new_backward
        return float(radiated)


def _fit_mouth_reflection(bore: SectionedBore) -> tuple[float, float]:
    """Least-squares fit of a one-pole to the mouth's reflection coefficient.

    The true coefficient is ``(Zr - Zc) / (Zr + Zc)`` with ``Zr`` the radiation
    impedance of the bell mouth. Fitting magnitude only -- a one-pole cannot
    match both magnitude and phase -- since the phase contribution is an end
    correction of well under one section's length.
    """
    from .acoustics import radiation_impedance

    freqs = np.linspace(20.0, min(4000.0, 0.4 * bore.sample_rate), 400)
    radius = float(bore.radii[-1])
    z_char = bore.air.density * bore.air.speed_of_sound / (np.pi * radius**2)
    z_rad = radiation_impedance(freqs, radius, bore.air)
    target = np.abs((z_rad - z_char) / (z_rad + z_char))

    omega = 2.0 * np.pi * freqs / bore.sample_rate
    best = (1.0, 0.0)
    best_error = np.inf
    for pole in np.linspace(0.0, 0.98, 99):
        response = np.abs(
            (1.0 - pole) / (1.0 - pole * np.exp(-1j * omega))
        )
        gain = float(np.sum(response * target) / np.sum(response**2))
        gain = min(gain, 1.0)  # never amplify; the termination must stay passive
        error = float(np.sum((gain * response - target) ** 2))
        if error < best_error:
            best_error = error
            best = (gain, float(pole))
    return best


def measure_input_impedance(
    bore: SectionedBore,
    n_samples: int = 1 << 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Probe the ladder with a flow impulse and recover its input impedance.

    This is the bridge back to stage 1. A flow source has infinite internal
    impedance, so the wave injected at the mouthpiece is
    ``p+ = p- + Zc * U``; driving ``U`` with a unit impulse and transforming
    the resulting mouthpiece pressure gives the input impedance directly,
    which can then be compared against the transfer-matrix result computed
    from the same geometry.

    Returns ``(freqs, impedance)`` over the positive-frequency half.
    """
    guide = Waveguide(bore)
    z_char = bore.characteristic_impedance(0)

    pressure = np.empty(n_samples)
    for n in range(n_samples):
        flow = z_char if n == 0 else 0.0
        injected = guide.mouthpiece_return + flow
        pressure[n] = injected + guide.mouthpiece_return
        guide.step(injected)

    spectrum = np.fft.rfft(pressure)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / bore.sample_rate)
    return freqs, spectrum


def regime_of(bore: SectionedBore, regime: int = 1, n_samples: int = 1 << 16,
              band: tuple[float, float] = (20.0, 600.0)) -> float:
    """Frequency of the ``regime``-th impedance peak, measured from the ladder."""
    from .analysis import find_resonances
    from .bore import BoreResponse

    freqs, spectrum = measure_input_impedance(bore, n_samples)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    found = find_resonances(
        BoreResponse(freqs[mask], spectrum[mask], 0.0, 0.0),
        fmin=band[0], fmax=band[1], max_count=regime,
    )
    return float(found.freqs[regime - 1]) if len(found) >= regime else float("nan")


def tune_to_waveguide(
    instrument: Trombolese,
    target_f1: float | None = None,
    n_alpha: int = 5,
    iterations: int = 3,
    regime: int = 1,
    sample_rate: float = 48_000.0,
    dispersion_reference_hz: float = 150.0,
    slide: float = 0.0,
) -> Trombolese:
    """Re-tune the pitch compensation against the waveguide's own resonances.

    Stage 1's compensation holds the fundamental exactly for the
    transfer-matrix model, but the waveguide approximates that model -- chiefly
    in how it handles the wall drag that slows the wave -- and the residual is
    not constant across the morph. Inheriting stage 1's lengths therefore
    leaves the *sounding* instrument drifting by some tens of cents as the bore
    morphs, which is precisely what the compensation existed to prevent.

    The waveguide is the instrument, so the instrument is what should be tuned.
    This measures the ladder's own fundamental and corrects the bore length
    toward the target, exploiting the fact that frequency is very nearly
    inversely proportional to length: a couple of multiplicative steps
    converge to a fraction of a cent, far cheaper than bisecting.

    Returns a copy carrying the refined compensation.
    """
    from dataclasses import replace

    from .bore import PitchCompensation

    base = instrument
    alphas = np.linspace(0.0, 1.0, n_alpha)
    lengths = np.array([base.bore_length_at(float(a)) for a in alphas])

    if target_f1 is None:
        probe = replace(base, pitch_compensation=None, bore_length=float(lengths[0]))
        target_f1 = regime_of(
            section_bore(probe, 0.0, slide, sample_rate,
                         dispersion_reference_hz=dispersion_reference_hz),
            regime,
        )

    for _ in range(iterations):
        for i, alpha in enumerate(alphas):
            probe = replace(
                base, pitch_compensation=None, bore_length=float(lengths[i])
            )
            measured = regime_of(
                section_bore(probe, float(alpha), slide, sample_rate,
                             dispersion_reference_hz=dispersion_reference_hz),
                regime,
            )
            if not np.isfinite(measured):
                continue
            # f ~ 1/L, so scaling the length by the frequency ratio lands close.
            lengths[i] *= measured / target_f1

    return replace(
        base,
        pitch_compensation=PitchCompensation(
            alphas=alphas, lengths=lengths, target_f1=float(target_f1)
        ),
    )
