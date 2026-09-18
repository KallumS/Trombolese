"""Checks on the Trombolese geometry and its morph."""

from __future__ import annotations

import numpy as np
import pytest

from trombolese import Trombolese, find_resonances

FREQS = np.linspace(20.0, 900.0, 20_000)


@pytest.fixture
def instrument() -> Trombolese:
    return Trombolese()


class TestGeometry:
    def test_morph_endpoints(self, instrument: Trombolese) -> None:
        assert instrument.throat_radius(0.0) == instrument.cyl_throat_radius
        assert instrument.throat_radius(1.0) == instrument.cone_throat_radius
        assert instrument.bell_entry_radius(0.0) == instrument.cyl_bell_entry_radius
        assert instrument.bell_entry_radius(1.0) == instrument.cone_bell_entry_radius

    def test_cylindrical_limit_has_a_uniform_bore(self, instrument: Trombolese) -> None:
        bore = [s for s in instrument.segments(0.0) if s.name == "bore"][0]
        assert bore.is_cylindrical

    def test_conical_limit_tapers_outward(self, instrument: Trombolese) -> None:
        bore = [s for s in instrument.segments(1.0) if s.name == "bore"][0]
        assert bore.radius_out > bore.radius_in

    def test_bore_is_a_single_frustum_at_every_morph_position(
        self, instrument: Trombolese
    ) -> None:
        """The morph must stay exact rather than becoming a sliced approximation."""
        for alpha in np.linspace(0.0, 1.0, 11):
            bores = [s for s in instrument.segments(float(alpha)) if s.name == "bore"]
            assert len(bores) == 1

    def test_segments_join_without_gaps_in_the_bell(
        self, instrument: Trombolese
    ) -> None:
        bell = [s for s in instrument.segments(0.6) if s.name == "bell"]
        for earlier, later in zip(bell[:-1], bell[1:]):
            assert earlier.radius_out == pytest.approx(later.radius_in)

    def test_bell_mouth_is_fixed_across_the_morph(self, instrument: Trombolese) -> None:
        """The radiating aperture must not move, or the sweep confounds two changes."""
        for alpha in (0.0, 0.5, 1.0):
            assert instrument.bell_radii(alpha)[-1] == pytest.approx(
                instrument.bell_radius
            )

    def test_profile_matches_segments(self, instrument: Trombolese) -> None:
        x, radius = instrument.profile(0.5)
        assert x[-1] == pytest.approx(instrument.total_length(0.5))
        assert radius[-1] == pytest.approx(instrument.bell_radius)

    def test_slide_lengthens_the_instrument(self, instrument: Trombolese) -> None:
        assert instrument.total_length(0.5, 0.3) == pytest.approx(
            instrument.total_length(0.5) + 0.3
        )

    def test_truncation_ratio_falls_as_the_bore_turns_conical(
        self, instrument: Trombolese
    ) -> None:
        ratios = [instrument.truncation_ratio(a) for a in np.linspace(0.1, 1.0, 10)]
        assert np.all(np.diff(ratios) < 0.0)
        assert ratios[-1] < 0.2  # complete enough to give a harmonic series

    def test_rejects_out_of_range_morph(self, instrument: Trombolese) -> None:
        for bad in (-0.01, 1.01):
            with pytest.raises(ValueError):
                instrument.segments(bad)

    def test_rejects_negative_slide(self, instrument: Trombolese) -> None:
        with pytest.raises(ValueError):
            instrument.segments(0.5, -0.1)


class TestMorphAcoustics:
    def test_cylindrical_limit_overblows_at_the_twelfth(
        self, instrument: Trombolese
    ) -> None:
        found = find_resonances(instrument.response(FREQS, alpha=0.0), fmax=900.0)
        assert found.overblow_ratio == pytest.approx(3.0, abs=0.1)

    def test_conical_limit_overblows_at_the_octave(
        self, instrument: Trombolese
    ) -> None:
        found = find_resonances(instrument.response(FREQS, alpha=1.0), fmax=900.0)
        assert found.overblow_ratio == pytest.approx(2.0, abs=0.1)

    def test_conical_limit_has_a_near_complete_harmonic_series(
        self, instrument: Trombolese
    ) -> None:
        found = find_resonances(
            instrument.response(FREQS, alpha=1.0), fmax=900.0, max_count=4
        )
        assert np.allclose(found.ratios, [1, 2, 3, 4], rtol=0.04)

    def test_cylindrical_limit_suppresses_even_partials(
        self, instrument: Trombolese
    ) -> None:
        found = find_resonances(
            instrument.response(FREQS, alpha=0.0), fmax=900.0, max_count=4
        )
        assert np.allclose(found.ratios, [1, 3, 5, 7], rtol=0.04)

    def test_overblow_ratio_falls_monotonically(self, instrument: Trombolese) -> None:
        """The morph must be a usable continuous control, not a jump between two states."""
        alphas = np.linspace(0.0, 1.0, 21)
        ratios = [
            find_resonances(r, fmax=900.0).overblow_ratio
            for r in instrument.sweep(FREQS, alphas)
        ]
        assert np.all(np.diff(ratios) < 0.0)

    def test_upper_regimes_actually_move(self, instrument: Trombolese) -> None:
        """Guards the bug where only the fundamental responded to the morph."""
        low = find_resonances(
            instrument.response(FREQS, alpha=0.0), fmax=900.0, max_count=3
        )
        high = find_resonances(
            instrument.response(FREQS, alpha=1.0), fmax=900.0, max_count=3
        )
        assert high.freqs[1] > low.freqs[1] * 1.10
        assert high.freqs[2] > low.freqs[2] * 1.05

    def test_slide_lowers_the_pitch(self, instrument: Trombolese) -> None:
        closed = find_resonances(instrument.response(FREQS, 0.0, 0.0), fmax=900.0)
        extended = find_resonances(instrument.response(FREQS, 0.0, 0.3), fmax=900.0)
        assert extended.freqs[0] < closed.freqs[0]

    def test_sweep_matches_individual_responses(self, instrument: Trombolese) -> None:
        alphas = (0.0, 0.4, 1.0)
        swept = instrument.sweep(FREQS, alphas)
        for alpha, response in zip(alphas, swept):
            direct = instrument.response(FREQS, alpha=alpha)
            assert np.allclose(response.impedance, direct.impedance)

    def test_warmer_air_raises_the_pitch(self) -> None:
        from trombolese.constants import Air

        cold = Trombolese(air=Air(15.0))
        warm = Trombolese(air=Air(30.0))
        f_cold = find_resonances(cold.response(FREQS, 0.5), fmax=900.0).freqs[0]
        f_warm = find_resonances(warm.response(FREQS, 0.5), fmax=900.0).freqs[0]
        assert f_warm > f_cold
