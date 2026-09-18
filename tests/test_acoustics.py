"""Checks on the duct-acoustics primitives against results known in closed form."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from trombolese.acoustics import (
    cone_matrix,
    cylinder_matrix,
    input_impedance,
    invert,
    radiation_impedance,
    wavenumber,
)
from trombolese.constants import Air

AIR = Air(20.0)
FREQS = np.linspace(20.0, 2000.0, 40_000)


def peak_freqs(impedance: np.ndarray, count: int = 5) -> np.ndarray:
    indices, _ = find_peaks(np.log(np.abs(impedance)))
    return FREQS[indices][:count]


class TestWavenumber:
    def test_decays_in_the_direction_of_travel(self) -> None:
        """With an exp(-jkx) convention, loss means a negative imaginary part."""
        k = wavenumber(FREQS, 0.007, AIR)
        assert np.all(k.imag < 0.0)

    def test_approaches_the_lossless_value(self) -> None:
        k = wavenumber(FREQS, 0.007, AIR)
        lossless = 2.0 * np.pi * FREQS / AIR.speed_of_sound
        # Losses perturb the real part by a few percent at the bottom of the
        # band and less above it.
        assert np.all(k.real > lossless)
        assert np.all(k.real < lossless * 1.08)

    def test_attenuation_scales_as_one_over_radius(self) -> None:
        narrow = wavenumber(FREQS, 0.002, AIR)
        wide = wavenumber(FREQS, 0.008, AIR)
        assert np.allclose(narrow.imag / wide.imag, 4.0, rtol=1e-12)

    def test_attenuation_scales_as_root_frequency(self) -> None:
        k = wavenumber(np.array([100.0, 400.0]), 0.007, AIR)
        assert np.isclose(k.imag[1] / k.imag[0], 2.0, rtol=1e-12)

    def test_rejects_non_positive_frequencies(self) -> None:
        with pytest.raises(ValueError):
            wavenumber(np.array([0.0, 100.0]), 0.007, AIR)


class TestRadiationImpedance:
    def test_is_passive(self) -> None:
        """A termination that absorbs must never have negative resistance."""
        z = radiation_impedance(FREQS, 0.1, AIR)
        assert np.all(z.real > 0.0)

    def test_low_frequency_limit(self) -> None:
        """At ka << 1 it must reduce to the Levine-Schwinger form."""
        radius = 0.01
        freqs = np.array([5.0, 10.0])
        z = radiation_impedance(freqs, radius, AIR)
        ka = 2.0 * np.pi * freqs * radius / AIR.speed_of_sound
        z_char = AIR.density * AIR.speed_of_sound / (np.pi * radius**2)
        assert np.allclose(z.real / z_char, 0.25 * ka**2, rtol=2e-3)
        assert np.allclose(z.imag / z_char, 0.6133 * ka, rtol=2e-3)

    def test_tends_to_the_characteristic_impedance(self) -> None:
        """At ka >> 1 the mouth stops reflecting."""
        radius = 0.1
        z = radiation_impedance(np.array([40_000.0]), radius, AIR)
        z_char = AIR.density * AIR.speed_of_sound / (np.pi * radius**2)
        assert z.real[0] / z_char == pytest.approx(1.0, abs=0.02)


class TestTransferMatrices:
    def test_invert_matches_numpy(self) -> None:
        matrix = cone_matrix(FREQS, 0.004, 0.05, 1.0, AIR)
        assert np.allclose(invert(matrix), np.linalg.inv(matrix), atol=1e-9)

    @pytest.mark.parametrize(
        "matrix_factory",
        [
            lambda: cylinder_matrix(FREQS, 0.007, 0.5, AIR),
            lambda: cone_matrix(FREQS, 0.004, 0.05, 1.0, AIR),
            lambda: cone_matrix(FREQS, 0.05, 0.004, 1.0, AIR),
        ],
        ids=["cylinder", "expanding-cone", "contracting-cone"],
    )
    def test_determinant_is_unity(self, matrix_factory) -> None:
        """Reciprocity. A unit determinant is the strongest cheap check there is."""
        assert np.allclose(np.linalg.det(matrix_factory()), 1.0, atol=1e-10)

    def test_cone_reduces_to_cylinder(self) -> None:
        """As the apex recedes, the spherical-wave form must become the plane one.

        Compared against the largest matrix entry rather than elementwise: the
        entries span twelve orders of magnitude (B carries a characteristic
        impedance, C its reciprocal), so an elementwise relative tolerance
        would be testing floating-point noise in the small ones.
        """
        cone = cone_matrix(FREQS, 0.007, 0.0070001, 0.5, AIR)
        cylinder = cylinder_matrix(FREQS, 0.00700005, 0.5, AIR)
        assert np.max(np.abs(cone - cylinder)) / np.max(np.abs(cylinder)) < 1e-8

    def test_a_gentle_taper_does_not_overflow(self) -> None:
        """A near-cylindrical cone puts its apex kilometres away; that must be safe."""
        matrix = cone_matrix(FREQS, 0.007, 0.007 * (1 + 1e-7), 0.5, AIR)
        assert np.all(np.isfinite(matrix))

    def test_rejects_degenerate_segments(self) -> None:
        with pytest.raises(ValueError):
            cone_matrix(FREQS, 0.004, 0.006, 0.0, AIR)
        with pytest.raises(ValueError):
            cone_matrix(FREQS, 0.0, 0.006, 1.0, AIR)


class TestTextbookResonances:
    """The two results the whole model has to reproduce."""

    LENGTH = 1.0

    def test_cylinder_gives_odd_harmonics(self) -> None:
        """A cylinder driven at a closed end resonates at (2n-1) c / 4L."""
        matrix = cylinder_matrix(FREQS, 0.007, self.LENGTH, AIR)
        found = peak_freqs(input_impedance(matrix, 0.0))
        # Boundary-layer loss slows low frequencies more than high ones, so the
        # series stretches slightly rather than landing exactly on the odds.
        assert np.allclose(found / found[0], [1, 3, 5, 7, 9], rtol=2e-2)

        quarter_wave = AIR.speed_of_sound / (4.0 * self.LENGTH)
        # Boundary-layer losses slow the wave, flattening everything ~2%.
        assert found[0] == pytest.approx(quarter_wave, rel=0.03)

    def test_cone_gives_a_complete_harmonic_series(self) -> None:
        """A near-complete cone resonates at n c / 2L."""
        apex = 0.02
        matrix = cone_matrix(
            FREQS, 0.004, 0.004 * (apex + self.LENGTH) / apex, self.LENGTH, AIR
        )
        found = peak_freqs(input_impedance(matrix, 0.0))
        assert np.allclose(found / found[0], [1, 2, 3, 4, 5], rtol=1e-2)

        half_wave = AIR.speed_of_sound / (2.0 * self.LENGTH)
        assert found[0] == pytest.approx(half_wave, rel=0.03)

    def test_the_two_families_are_distinguishable(self) -> None:
        """The premise of the instrument: a twelfth versus an octave."""
        cylinder = peak_freqs(
            input_impedance(cylinder_matrix(FREQS, 0.007, self.LENGTH, AIR), 0.0)
        )
        apex = 0.02
        cone = peak_freqs(
            input_impedance(
                cone_matrix(
                    FREQS, 0.004, 0.004 * (apex + self.LENGTH) / apex,
                    self.LENGTH, AIR,
                ),
                0.0,
            )
        )
        assert cylinder[1] / cylinder[0] == pytest.approx(3.0, rel=0.05)
        assert cone[1] / cone[0] == pytest.approx(2.0, rel=0.05)
