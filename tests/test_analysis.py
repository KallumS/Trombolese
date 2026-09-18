"""Checks on resonance extraction and the harmonic fit."""

from __future__ import annotations

import numpy as np
import pytest

from trombolese import (
    Trombolese,
    compensate_pitch,
    find_resonances,
    harmonic_fit,
    pitch_neutral,
    scan_morph,
)
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


class TestPitchCompensation:
    """The morph must be usable as a purely timbral control."""

    def test_holds_the_fundamental_across_the_morph(self) -> None:
        instrument = pitch_neutral(Trombolese(), n_alpha=5)
        target = instrument.pitch_compensation.target_f1
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            f1 = find_resonances(
                instrument.response(FREQS, alpha=alpha), fmax=900.0
            ).freqs[0]
            cents = 1200.0 * np.log2(f1 / target)
            assert abs(cents) < 2.0

    def test_defaults_to_the_cylindrical_pitch(self) -> None:
        plain = Trombolese()
        expected = find_resonances(plain.response(FREQS, alpha=0.0), fmax=900.0).freqs[0]
        compensation = compensate_pitch(plain, n_alpha=3)
        assert compensation.target_f1 == pytest.approx(expected, rel=2e-3)

    def test_accepts_an_explicit_target(self) -> None:
        instrument = pitch_neutral(Trombolese(), target_f1=50.0, n_alpha=3)
        f1 = find_resonances(instrument.response(FREQS, alpha=0.5), fmax=900.0).freqs[0]
        assert f1 == pytest.approx(50.0, rel=2e-3)

    def test_the_bore_lengthens_monotonically(self) -> None:
        compensation = compensate_pitch(Trombolese(), n_alpha=5)
        assert np.all(np.diff(compensation.lengths) > 0.0)
        # A cone sounds c/2L where a cylinder sounds c/4L, so compensating
        # costs most of a factor of two in length.
        assert 1.5 < compensation.length_ratio < 2.0

    def test_compensation_preserves_harmonicity(self) -> None:
        """Length cancels out of the truncation ratio, so it must not rearrange modes."""
        plain = Trombolese()
        tuned = pitch_neutral(plain, n_alpha=5)
        assert tuned.truncation_ratio(1.0) == pytest.approx(plain.truncation_ratio(1.0))
        found = find_resonances(tuned.response(FREQS, alpha=1.0), fmax=900.0, max_count=3)
        assert np.allclose(found.ratios, [1, 2, 3], rtol=0.06)

    def test_still_overblows_from_a_twelfth_to_an_octave(self) -> None:
        instrument = pitch_neutral(Trombolese(), n_alpha=5)
        low = find_resonances(instrument.response(FREQS, alpha=0.0), fmax=900.0)
        high = find_resonances(instrument.response(FREQS, alpha=1.0), fmax=900.0)
        assert low.overblow_ratio == pytest.approx(3.0, abs=0.1)
        assert high.overblow_ratio == pytest.approx(2.05, abs=0.1)

    def test_interpolates_between_solved_points(self) -> None:
        compensation = compensate_pitch(Trombolese(), n_alpha=5)
        midpoint = compensation.length_at(0.125)
        assert compensation.lengths[0] < midpoint < compensation.lengths[1]

    def test_uncompensated_instrument_ignores_alpha_for_length(self) -> None:
        plain = Trombolese()
        assert plain.bore_length_at(0.0) == plain.bore_length_at(1.0)
