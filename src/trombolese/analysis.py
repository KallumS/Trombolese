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

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .bore import BoreResponse, Trombolese

__all__ = [
    "Resonances",
    "find_resonances",
    "HarmonicFit",
    "harmonic_fit",
    "MorphScan",
    "scan_morph",
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
