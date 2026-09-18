"""Checks on the waveguide, against stage 1 and against textbook limits."""

from __future__ import annotations

import numpy as np
import pytest

from trombolese import Trombolese, find_resonances, pitch_neutral
from trombolese.bore import BoreResponse
from trombolese.constants import AIR_20C
from trombolese.waveguide import (
    SectionedBore,
    Waveguide,
    measure_input_impedance,
    regime_of,
    section_bore,
    tune_to_waveguide,
)

SAMPLE_RATE = 48_000.0
REFERENCE_FREQS = np.linspace(20.0, 900.0, 20_000)


def ladder_resonances(bore: SectionedBore, count: int = 4,
                      n_samples: int = 1 << 16) -> np.ndarray:
    freqs, spectrum = measure_input_impedance(bore, n_samples)
    mask = (freqs >= 20.0) & (freqs <= 900.0)
    found = find_resonances(
        BoreResponse(freqs[mask], spectrum[mask], 0.0, 0.0),
        fmax=900.0, max_count=count,
    )
    return found.freqs


class TestSectioning:
    def test_section_count_follows_length(self) -> None:
        instrument = Trombolese()
        bore = section_bore(instrument, alpha=0.0, sample_rate=SAMPLE_RATE)
        expected = instrument.total_length(0.0) / (
            instrument.air.speed_of_sound / SAMPLE_RATE
        )
        assert bore.n_sections == pytest.approx(expected, abs=1.0)

    def test_sections_are_one_sample_long(self) -> None:
        bore = section_bore(Trombolese(), sample_rate=SAMPLE_RATE)
        assert bore.section_length == pytest.approx(
            AIR_20C.speed_of_sound / SAMPLE_RATE
        )

    def test_cylindrical_limit_has_no_taper_scattering(self) -> None:
        """At alpha = 0 the bore proper is uniform, so only the bell scatters."""
        bore = section_bore(Trombolese(include_mouthpiece=False), alpha=0.0)
        n = bore.n_sections
        # The first two-thirds is the cylindrical bore; the bell is at the end.
        assert np.allclose(bore.reflections[: n // 2], 0.0, atol=1e-9)

    def test_conical_limit_scatters_throughout(self) -> None:
        bore = section_bore(Trombolese(include_mouthpiece=False), alpha=1.0)
        n = bore.n_sections
        assert np.any(np.abs(bore.reflections[: n // 2]) > 1e-6)

    def test_reflections_are_bounded(self) -> None:
        """A junction cannot reflect more than everything."""
        for alpha in (0.0, 0.5, 1.0):
            bore = section_bore(Trombolese(), alpha=alpha)
            assert np.all(np.abs(bore.reflections) <= 1.0)

    def test_losses_are_passive(self) -> None:
        bore = section_bore(Trombolese(), alpha=0.5)
        assert np.all(bore.losses > 0.0)
        assert np.all(bore.losses <= 1.0)

    def test_narrow_sections_lose_more(self) -> None:
        """Wall loss scales as 1 / radius, so the throat damps hardest."""
        bore = section_bore(Trombolese(include_mouthpiece=False), alpha=1.0)
        assert bore.losses[0] < bore.losses[-1]


class TestLadder:
    def test_uniform_ladder_matches_the_textbook(self) -> None:
        """A bare cylinder with an ideal open end resonates at (2n-1) c / 4L."""
        n = 400
        bore = SectionedBore(
            radii=np.full(n, 0.007), reflections=np.zeros(n - 1),
            losses=np.ones(n), dispersion=np.zeros(n),
            section_length=AIR_20C.speed_of_sound / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE, air=AIR_20C,
        )
        guide = Waveguide(bore)
        guide._reflection_gain, guide._reflection_pole = 1.0, 0.0
        guide._delay_samples = 0.0
        guide._delay = np.zeros(2)

        samples = 1 << 16
        pressure = np.empty(samples)
        z_char = bore.characteristic_impedance(0)
        for i in range(samples):
            injected = guide.mouthpiece_return + (z_char if i == 0 else 0.0)
            pressure[i] = injected + guide.mouthpiece_return
            guide.step(injected)

        spectrum = np.fft.rfft(pressure)
        freqs = np.fft.rfftfreq(samples, 1.0 / SAMPLE_RATE)
        mask = (freqs >= 10.0) & (freqs <= 900.0)
        found = find_resonances(
            BoreResponse(freqs[mask], spectrum[mask], 0.0, 0.0),
            fmin=10.0, fmax=900.0, max_count=4,
        )
        quarter = SAMPLE_RATE / (4.0 * n)
        expected = quarter * np.array([1, 3, 5, 7])
        assert np.allclose(
            1200.0 * np.log2(found.freqs / expected), 0.0, atol=3.0
        )

    def test_energy_decays(self) -> None:
        """With losses and a radiating end, a struck bore must ring down."""
        guide = Waveguide(section_bore(Trombolese(), alpha=0.5))
        guide.step(1.0)
        for _ in range(2000):
            guide.step(0.0)
        early = np.max(np.abs(guide.forward)) + np.max(np.abs(guide.backward))
        for _ in range(200_000):
            guide.step(0.0)
        late = np.max(np.abs(guide.forward)) + np.max(np.abs(guide.backward))
        assert late < early

    def test_adopt_preserves_the_sounding_state(self) -> None:
        """Morphing mid-note must not silence or explode the ladder."""
        instrument = pitch_neutral(Trombolese(), n_alpha=5)
        guide = Waveguide(section_bore(instrument, alpha=0.0))
        for i in range(4000):
            guide.step(1.0 if i < 20 else 0.0)

        before = float(np.sqrt(np.mean(guide.forward**2)))
        guide.adopt(section_bore(instrument, alpha=0.5))
        after = float(np.sqrt(np.mean(guide.forward**2)))

        assert np.all(np.isfinite(guide.forward))
        assert after == pytest.approx(before, rel=0.5)
        assert after > 0.0

    def test_adopt_changes_the_section_count(self) -> None:
        instrument = pitch_neutral(Trombolese(), n_alpha=5)
        guide = Waveguide(section_bore(instrument, alpha=0.0))
        first = guide.bore.n_sections
        guide.adopt(section_bore(instrument, alpha=1.0))
        assert guide.bore.n_sections > first
        assert len(guide.forward) == guide.bore.n_sections


class TestAgreementWithStageOne:
    """The waveguide must reproduce the instrument the transfer matrix designed."""

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_overblow_ratio_agrees(self, alpha: float) -> None:
        instrument = Trombolese()
        ladder = ladder_resonances(
            section_bore(instrument, alpha=alpha, dispersion_reference_hz=150.0)
        )
        reference = find_resonances(
            instrument.response(REFERENCE_FREQS, alpha=alpha), fmax=900.0, max_count=4
        )
        assert ladder[1] / ladder[0] == pytest.approx(
            reference.freqs[1] / reference.freqs[0], abs=0.08
        )

    def test_morph_still_runs_from_a_twelfth_to_an_octave(self) -> None:
        instrument = Trombolese()
        low = ladder_resonances(section_bore(instrument, alpha=0.0))
        high = ladder_resonances(section_bore(instrument, alpha=1.0))
        assert low[1] / low[0] == pytest.approx(3.0, abs=0.15)
        assert high[1] / high[0] == pytest.approx(2.05, abs=0.15)

    def test_dispersion_is_what_closes_the_tuning_gap(self) -> None:
        """Without the slowed wave the ladder plays sharp by tens of cents."""
        instrument = Trombolese()
        reference = find_resonances(
            instrument.response(REFERENCE_FREQS, alpha=0.0), fmax=900.0, max_count=1
        ).freqs[0]

        corrected = section_bore(instrument, alpha=0.0, dispersion_reference_hz=36.0)
        uncorrected = SectionedBore(
            radii=corrected.radii, reflections=corrected.reflections,
            losses=corrected.losses, dispersion=np.zeros_like(corrected.dispersion),
            section_length=corrected.section_length,
            sample_rate=corrected.sample_rate, air=corrected.air,
        )
        with_dispersion = ladder_resonances(corrected, count=1)[0]
        without = ladder_resonances(uncorrected, count=1)[0]

        assert 1200.0 * np.log2(without / reference) > 40.0
        assert abs(1200.0 * np.log2(with_dispersion / reference)) < 12.0


class TestWaveguideTuning:
    def test_holds_a_regime_across_the_morph(self) -> None:
        instrument = tune_to_waveguide(
            pitch_neutral(Trombolese(), regime=3, n_alpha=3),
            regime=3, n_alpha=3, iterations=3,
        )
        target = instrument.pitch_compensation.target_f1
        for alpha in (0.0, 0.5, 1.0):
            measured = regime_of(
                section_bore(instrument, alpha=alpha,
                             dispersion_reference_hz=150.0),
                regime=3,
            )
            assert abs(1200.0 * np.log2(measured / target)) < 15.0

    def test_holding_a_playing_regime_is_cheaper_than_the_pedal(self) -> None:
        """Only one regime can be held, and the upper ones cost far less length."""
        pedal = pitch_neutral(Trombolese(), regime=1, n_alpha=5)
        playing = pitch_neutral(Trombolese(), regime=3, n_alpha=5)
        assert pedal.pitch_compensation.length_ratio > 1.5
        assert playing.pitch_compensation.length_ratio < 1.2
