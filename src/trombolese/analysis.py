"""Extracting playing behaviour from an input-impedance curve.

Two questions matter at this stage, and both are answered from the positions
of the impedance peaks:

1. **What does the instrument overblow by?** The ratio of the second regime to
   the first. A cylinder gives 3 (the twelfth), a cone gives 2 (the octave).
   Watching this quantity move as the bore morphs is the single most
   informative number the model produces.
2. **How harmonic is the set of regimes?** A reed only locks several
   resonances together into one strong, stable, timbrally rich note when those
   resonances are near-harmonically related. A morph position whose regimes
   are badly inharmonic will be audibly unstable and hard to play -- which may
   be exactly what you want in the middle of the sweep, but you should know
   where it happens rather than discover it later in the DSP.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import find_peaks

from .bore import BoreResponse, PitchCompensation, Trombolese

__all__ = [
    "Resonances",
    "find_resonances",
    "HarmonicFit",
    "harmonic_fit",
    "MorphScan",
    "scan_morph",
    "compensate_pitch",
    "pitch_neutral",
]


@dataclass
class Resonances:
    """Impedance maxima of a bore, in ascending frequency order."""

    freqs: np.ndarray
    magnitudes_db: np.ndarray

    def __len__(self) -> int:
        return len(self.freqs)

    @property
    def ratios(self) -> np.ndarray:
        """Resonance frequencies as multiples of the first."""
        if len(self.freqs) == 0:
            return np.array([])
        return self.freqs / self.freqs[0]

    @property
    def overblow_ratio(self) -> float:
        """Ratio of the second regime to the first.

        3 is a twelfth (cylindrical behaviour), 2 an octave (conical).
        Returns NaN if fewer than two resonances were found.
        """
        if len(self.freqs) < 2:
            return float("nan")
        return float(self.freqs[1] / self.freqs[0])


def find_resonances(
    response: BoreResponse,
    fmin: float = 30.0,
    fmax: float = 1200.0,
    prominence_db: float = 1.0,
    max_count: int | None = None,
) -> Resonances:
    """Locate impedance maxima, refined to sub-bin accuracy.

    Peaks are found on the decibel magnitude and then refined by fitting a
    parabola through the peak bin and its neighbours. With a reasonably dense
    frequency grid this puts the resonance frequencies well inside a cent,
    which matters because the whole point is to compare intervals.

    ``prominence_db`` rejects the shallow ripple that the bell's sliced
    approximation and the radiation model can leave on the curve.
    """
    freqs = response.freqs
    mag_db = response.magnitude_db

    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return Resonances(np.array([]), np.array([]))

    f_band = freqs[band]
    m_band = mag_db[band]

    indices, _ = find_peaks(m_band, prominence=prominence_db)

    refined_f: list[float] = []
    refined_m: list[float] = []
    for i in indices:
        if i == 0 or i == len(m_band) - 1:
            refined_f.append(float(f_band[i]))
            refined_m.append(float(m_band[i]))
            continue
        # Parabolic interpolation through three points about the peak.
        y0, y1, y2 = m_band[i - 1], m_band[i], m_band[i + 1]
        denom = y0 - 2.0 * y1 + y2
        offset = 0.0 if denom == 0.0 else 0.5 * (y0 - y2) / denom
        step = f_band[i + 1] - f_band[i]
        refined_f.append(float(f_band[i] + offset * step))
        refined_m.append(float(y1 - 0.25 * (y0 - y2) * offset))

    result = Resonances(np.asarray(refined_f), np.asarray(refined_m))
    if max_count is not None:
        result = Resonances(result.freqs[:max_count], result.magnitudes_db[:max_count])
    return result


@dataclass
class HarmonicFit:
    """Best-fit harmonic series through a set of resonance frequencies."""

    f0: float
    partials: np.ndarray
    deviations_cents: np.ndarray
    converged: bool

    @property
    def rms_cents(self) -> float:
        if len(self.deviations_cents) == 0:
            return float("nan")
        return float(np.sqrt(np.mean(self.deviations_cents**2)))

    @property
    def max_cents(self) -> float:
        if len(self.deviations_cents) == 0:
            return float("nan")
        return float(np.max(np.abs(self.deviations_cents)))


def harmonic_fit(
    freqs: np.ndarray,
    tolerance_cents: float = 60.0,
    grid: int = 4000,
) -> HarmonicFit:
    """Fit ``f_n ~ n * f0`` to a set of resonances and report the misfit.

    The assignment of partial numbers is not known in advance -- a cylindrical
    bore's resonances are partials 1, 3, 5 of a series whose even members
    simply do not exist -- so ``f0`` and the assignment must be found together.

    Any sufficiently small ``f0`` fits any set of frequencies arbitrarily well
    (every frequency lands near *some* multiple of it), so minimising error
    alone is ill-posed. This instead searches ``f0`` downward from the highest
    plausible value and accepts the **largest** one that fits within
    ``tolerance_cents`` -- the most constrained explanation of the data. If
    nothing meets the tolerance it falls back to the assignment minimising
    RMS error and reports ``converged = False``.
    """
    freqs = np.asarray(freqs, dtype=float)
    if len(freqs) < 2:
        return HarmonicFit(float("nan"), np.array([]), np.array([]), False)

    candidates = np.linspace(freqs[0] * 1.05, freqs[0] / 6.0, grid)
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None

    for f0_guess in candidates:
        partials = np.maximum(np.round(freqs / f0_guess), 1.0)
        if len(np.unique(partials)) != len(partials):
            continue
        # With the assignment fixed, the least-squares f0 is available in closed form.
        f0 = float(np.sum(partials * freqs) / np.sum(partials**2))
        if f0 <= 0.0:
            continue
        cents = 1200.0 * np.log2(freqs / (partials * f0))
        rms = float(np.sqrt(np.mean(cents**2)))

        if best is None or rms < best[0]:
            best = (rms, f0, partials, cents)
        if np.max(np.abs(cents)) <= tolerance_cents:
            return HarmonicFit(f0, partials.astype(int), cents, True)

    if best is None:
        return HarmonicFit(float("nan"), np.array([]), np.array([]), False)

    _, f0, partials, cents = best
    return HarmonicFit(f0, partials.astype(int), cents, False)


@dataclass
class MorphScan:
    """Resonance structure sampled across the whole morph.

    Computed once and shared by every figure that plots against ``alpha``,
    since finding the resonances is far more expensive than drawing them.
    """

    alphas: np.ndarray
    fundamentals: np.ndarray
    ratios: np.ndarray

    @property
    def overblow_ratios(self) -> np.ndarray:
        """The second regime as a multiple of the first, per morph position."""
        return self.ratios[:, 1]


def scan_morph(
    instrument: Trombolese,
    freqs: np.ndarray,
    n_alpha: int = 41,
    n_partials: int = 5,
    slide: float = 0.0,
    fmax: float = 1200.0,
) -> MorphScan:
    """Find the first ``n_partials`` regimes at ``n_alpha`` morph positions.

    Missing resonances -- a regime that has drifted above ``fmax``, or one too
    shallow to register -- are left as NaN rather than silently dropped, so a
    gap in a plotted line means the model genuinely lost the regime there
    instead of the curve quietly shifting onto the next one up.
    """
    alphas = np.linspace(0.0, 1.0, n_alpha)
    responses = instrument.sweep(freqs, alphas, slide)

    fundamentals = np.full(n_alpha, np.nan)
    ratios = np.full((n_alpha, n_partials), np.nan)

    for i, response in enumerate(responses):
        found = find_resonances(response, fmax=fmax, max_count=n_partials)
        if len(found) == 0:
            continue
        fundamentals[i] = found.freqs[0]
        ratios[i, : len(found)] = found.ratios

    return MorphScan(alphas=alphas, fundamentals=fundamentals, ratios=ratios)


def _regime_frequency(
    instrument: Trombolese,
    alpha: float,
    slide: float,
    freqs: np.ndarray,
    regime: int = 1,
) -> float:
    """Frequency of the ``regime``-th resonance, or NaN if it is not in band."""
    found = find_resonances(
        instrument.response(freqs, alpha=alpha, slide=slide),
        fmin=float(freqs[0]),
        fmax=float(freqs[-1]),
        max_count=regime,
    )
    return float(found.freqs[regime - 1]) if len(found) >= regime else float("nan")


def compensate_pitch(
    instrument: Trombolese,
    target_f1: float | None = None,
    n_alpha: int = 21,
    slide: float = 0.0,
    regime: int = 1,
    search_band: tuple[float, float] = (8.0, 600.0),
    length_bracket: tuple[float, float] = (0.4, 4.0),
) -> PitchCompensation:
    """Solve for the bore length that holds one regime steady across the morph.

    At each of ``n_alpha`` morph positions this finds, by bisection, the bore
    length whose ``regime``-th impedance peak lands on ``target_f1``. The result
    is a lookup that :class:`~trombolese.bore.Trombolese` consults instead of
    its fixed ``bore_length``.

    Which regime to target
    ----------------------
    This choice is not a detail, and only one regime can be held at a time.
    The morph's entire purpose is to change the *ratios* between regimes -- from
    1, 3, 5, 7 to 1, 2, 3, 4 -- so holding all of them still is a contradiction.
    Compensating for the fundamental leaves the second regime sliding from
    3.04 to 2.05 times it, a drop of well over an octave; a player sustaining a
    note on that regime hears it lurch, or jump to a neighbouring one.

    So compensate for the regime that will actually be played. ``regime=1``
    holds the pedal note; ``regime=2`` or ``3`` holds a normal playing register
    and lets the pedal move instead.

    Compensation cannot disturb the instrument's harmonicity, because the
    cone's truncation ratio depends only on its end radii and not on its length
    (see :meth:`~trombolese.bore.Trombolese.truncation_ratio`). Lengthening
    therefore moves every resonance together without rearranging them.

    Parameters
    ----------
    target_f1:
        Fundamental to hold, in Hz. Defaults to whatever the uncompensated
        instrument sounds at ``alpha = 0``, so the cylindrical limit keeps the
        pitch it already had and the rest of the morph is brought to meet it.
    length_bracket:
        Multiples of the instrument's nominal ``bore_length`` to search
        between.
    """
    from scipy.optimize import brentq

    # Search against an uncompensated copy, or the solve would consult the
    # very table it is building.
    base = replace(instrument, pitch_compensation=None)
    freqs = np.linspace(search_band[0], search_band[1], 4000)

    if target_f1 is None:
        target_f1 = _regime_frequency(base, 0.0, slide, freqs, regime)
        if not np.isfinite(target_f1):
            raise ValueError(
                f"regime {regime} not found for the cylindrical limit"
            )

    low = base.bore_length * length_bracket[0]
    high = base.bore_length * length_bracket[1]

    alphas = np.linspace(0.0, 1.0, n_alpha)
    lengths = np.empty(n_alpha)

    for i, alpha in enumerate(alphas):
        def error(length: float, alpha: float = float(alpha)) -> float:
            candidate = replace(base, bore_length=length)
            f1 = _regime_frequency(candidate, alpha, slide, freqs, regime)
            if not np.isfinite(f1):
                raise ValueError(
                    f"regime {regime} not in {search_band} Hz at "
                    f"alpha={alpha:.3f}, bore length {length:.3f} m"
                )
            # Compare in cents: the solve is then equally tight at every pitch.
            return 1200.0 * np.log2(f1 / target_f1)

        lengths[i] = brentq(error, low, high, xtol=1e-6)

    return PitchCompensation(alphas=alphas, lengths=lengths, target_f1=target_f1)


def pitch_neutral(instrument: Trombolese, **kwargs) -> Trombolese:
    """A copy of ``instrument`` whose morph does not change the pitch.

    Convenience wrapper over :func:`compensate_pitch`. Keyword arguments are
    passed straight through.

    >>> from trombolese import Trombolese, pitch_neutral
    >>> instrument = pitch_neutral(Trombolese())
    """
    return replace(
        instrument, pitch_compensation=compensate_pitch(instrument, **kwargs)
    )
