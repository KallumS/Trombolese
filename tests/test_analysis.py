"""Checks on resonance extraction and the harmonic fit."""

from __future__ import annotations

import numpy as np
import pytest

from trombolese import Trombolese, find_resonances, harmonic_fit, scan_morph
from trombolese.bore import BoreResponse

FREQS = np.linspace(20.0, 900.0, 20_000)


def synthetic(peaks: list[float], q: float = 80.0) -> BoreResponse:
    """A response with Lorentzian peaks at known frequencies."""
    impedance = np.full_like(FREQS, 0.01, dtype=complex)
    for peak in peaks:
        impedance += 1.0 / (1.0 + 1j * q * (FREQS / peak - peak / FREQS))
    return BoreResponse(freqs=FREQS, impedance=impedance, alpha=0.0, slide=0.0)


class TestFindResonances:
    def test_recovers_known_peaks(self) -> None:
        peaks = [100.0, 300.0, 500.0]
        found = find_resonances(synthetic(peaks), fmax=900.0)
        assert np.allclose(found.freqs, peaks, rtol=1e-3)

    def test_interpolation_beats_the_grid_spacing(self) -> None:
        """Peaks must be located to well under a bin, or interval work is noise."""
        coarse = np.linspace(20.0, 900.0, 2_000)
        impedance = 1.0 / (1.0 + 1j * 80.0 * (coarse / 137.3 - 137.3 / coarse))
        response = BoreResponse(coarse, impedance + 0.01, 0.0, 0.0)
        found = find_resonances(response, fmax=900.0)
        spacing = coarse[1] - coarse[0]
        assert abs(found.freqs[0] - 137.3) < 0.2 * spacing

    def test_respects_the_band(self) -> None:
        found = find_resonances(synthetic([100.0, 300.0, 500.0]), fmin=200.0, fmax=400.0)
        assert len(found) == 1

    def test_respects_max_count(self) -> None:
        found = find_resonances(synthetic([100.0, 300.0, 500.0]), max_count=2)
        assert len(found) == 2

    def test_empty_band_is_not_an_error(self) -> None:
        found = find_resonances(synthetic([100.0]), fmin=1000.0, fmax=1100.0)
        assert len(found) == 0
        assert np.isnan(found.overblow_ratio)

    def test_ratios_are_relative_to_the_first(self) -> None:
        found = find_resonances(synthetic([100.0, 300.0, 500.0]), fmax=900.0)
        assert np.allclose(found.ratios, [1.0, 3.0, 5.0], rtol=1e-3)


class TestHarmonicFit:
    def test_exact_complete_series(self) -> None:
        freqs = np.array([100.0, 200.0, 300.0, 400.0])
        fit = harmonic_fit(freqs)
        assert fit.converged
        assert fit.f0 == pytest.approx(100.0, rel=1e-6)
        assert list(fit.partials) == [1, 2, 3, 4]
        assert fit.rms_cents < 1e-6

    def test_odd_only_series(self) -> None:
        """A cylinder's resonances are partials 1, 3, 5 of a series, not 1, 2, 3."""
        freqs = np.array([100.0, 300.0, 500.0, 700.0])
        fit = harmonic_fit(freqs)
        assert fit.converged
        assert list(fit.partials) == [1, 3, 5, 7]
        assert fit.f0 == pytest.approx(100.0, rel=1e-6)

    def test_prefers_the_largest_admissible_fundamental(self) -> None:
        """Guards the degenerate fit: a tiny f0 fits anything."""
        freqs = np.array([100.0, 200.0, 300.0])
        fit = harmonic_fit(freqs)
        assert fit.f0 == pytest.approx(100.0, rel=1e-3)

    def test_detects_an_inharmonic_set(self) -> None:
        fit = harmonic_fit(np.array([100.0, 213.0, 347.0]), tolerance_cents=10.0)
        assert not fit.converged or fit.max_cents > 10.0

    def test_needs_at_least_two_resonances(self) -> None:
        fit = harmonic_fit(np.array([100.0]))
        assert not fit.converged
        assert np.isnan(fit.rms_cents)

    def test_deviation_sign_convention(self) -> None:
        """A sharp partial must report a positive cent deviation."""
        fit = harmonic_fit(np.array([100.0, 202.0, 300.0]))
        assert fit.deviations_cents[1] > 0.0


class TestScanMorph:
    def test_shape_and_endpoints(self) -> None:
        scan = scan_morph(Trombolese(), FREQS, n_alpha=5, n_partials=4)
        assert scan.ratios.shape == (5, 4)
        assert scan.alphas[0] == 0.0
        assert scan.alphas[-1] == 1.0
        assert np.allclose(scan.ratios[:, 0], 1.0)

    def test_overblow_travels_from_a_twelfth_to_an_octave(self) -> None:
        scan = scan_morph(Trombolese(), FREQS, n_alpha=5, n_partials=3)
        assert scan.overblow_ratios[0] == pytest.approx(3.0, abs=0.1)
        assert scan.overblow_ratios[-1] == pytest.approx(2.0, abs=0.1)

    def test_missing_regimes_are_nan_not_shifted(self) -> None:
        """A regime that falls outside the band must leave a hole, not renumber."""
        scan = scan_morph(Trombolese(), FREQS, n_alpha=3, n_partials=3, fmax=120.0)
        assert np.isnan(scan.ratios[:, 2]).all()
