"""Checks on the correction that stops the reed morph dragging the pitch."""

from __future__ import annotations

import numpy as np
import pytest

from trombolese import Trombolese, pitch_neutral
from trombolese.reed import ReedParameters
from trombolese.synth import (
    Controls,
    ReedCompensation,
    Voice,
    _sounding_frequency,
    compensate_reed_morph,
)

SAMPLE_RATE = 48_000.0
PARAMETERS = ReedParameters(reed_frequency_ratio=2.0)


@pytest.fixture(scope="module")
def instrument() -> Trombolese:
    return pitch_neutral(Trombolese(), regime=3, n_alpha=3)


@pytest.fixture(scope="module")
def compensation(instrument: Trombolese) -> ReedCompensation:
    # A small grid: enough to exercise the machinery without paying for the
    # full calibration, which takes minutes.
    return compensate_reed_morph(
        instrument, embouchure=176.0, alphas=(0.0, 0.5, 1.0), n_beta=5,
        n_candidates=9, reed_parameters=PARAMETERS,
    )


class TestLookup:
    def test_multiplier_snaps_rather_than_blends(self) -> None:
        """Regime choice is discrete; blending two choices picks a third."""
        table = ReedCompensation(
            alphas=np.array([0.0, 1.0]), betas=np.array([0.0, 1.0]),
            multipliers=np.array([[1.0, 0.25], [1.0, 0.25]]),
            slide_offsets=np.zeros((2, 2)), target_frequency=200.0,
        )
        assert table.multiplier_at(0.0, 0.4) == pytest.approx(1.0)
        assert table.multiplier_at(0.0, 0.6) == pytest.approx(0.25)
        # Never a value that was not measured.
        for beta in np.linspace(0.0, 1.0, 21):
            assert table.multiplier_at(0.0, beta) in (1.0, 0.25)

    def test_slide_trim_is_interpolated(self) -> None:
        """Within a regime, pitch goes smoothly with length, so this may blend."""
        table = ReedCompensation(
            alphas=np.array([0.0, 1.0]), betas=np.array([0.0, 1.0]),
            multipliers=np.ones((2, 2)),
            slide_offsets=np.array([[0.0, -0.2], [0.0, -0.2]]),
            target_frequency=200.0,
        )
        assert table.slide_at(0.0, 0.5) == pytest.approx(-0.1)

    def test_it_is_a_surface_not_a_curve(self, compensation: ReedCompensation) -> None:
        """The two morphs interact, so the table must be indexed by both."""
        assert compensation.multipliers.shape == (
            len(compensation.alphas), len(compensation.betas)
        )


class TestCalibration:
    def test_the_uncompensated_morph_drags_the_pitch_badly(
        self, instrument: Trombolese
    ) -> None:
        voice = Voice(instrument, reed_parameters=PARAMETERS)
        pitches = [
            _sounding_frequency(
                voice,
                Controls(pressure=4200.0, lip_frequency=176.0, alpha=0.5, beta=beta),
            )
            for beta in (0.0, 1.0)
        ]
        drift = abs(1200.0 * np.log2(pitches[1] / pitches[0]))
        assert drift > 300.0

    def test_the_embouchure_is_wound_back_as_the_reed_stiffens(
        self, compensation: ReedCompensation
    ) -> None:
        """The double reed sits above the embouchure, so it must be lipped down."""
        middle = compensation.multipliers[len(compensation.alphas) // 2]
        assert middle[-1] < middle[0]

    def test_the_length_trim_goes_negative(
        self, compensation: ReedCompensation
    ) -> None:
        """An inward-striking valve sounds a regime flat; only shortening fixes it."""
        assert np.min(compensation.slide_offsets) < 0.0

    def test_it_holds_the_pitch_over_most_of_its_grid(
        self, instrument: Trombolese, compensation: ReedCompensation
    ) -> None:
        """Most points land, and the ones that miss, miss by an octave.

        The claim is deliberately statistical. The correction is a table over a
        control space whose regime map has genuine boundaries, and near one a
        note can settle an octave away from its neighbour. A full calibration
        clears its own grid within 26 cents; a coarse one like this fixture's
        leaves a point or two jumping. Asserting every point would be asserting
        something the method does not deliver.
        """
        target = compensation.target_frequency
        errors = []
        for alpha in compensation.alphas:
            for beta in compensation.betas:
                voice = Voice(
                    instrument, reed_parameters=PARAMETERS,
                    reed_compensation=compensation,
                )
                sounded = _sounding_frequency(
                    voice,
                    Controls(pressure=4200.0, lip_frequency=176.0,
                             alpha=float(alpha), beta=float(beta)),
                    seconds=0.6,
                )
                if np.isfinite(sounded):
                    errors.append(abs(1200.0 * np.log2(sounded / target)))

        assert len(errors) >= 10
        assert float(np.median(errors)) < 60.0
        within = sum(1 for e in errors if e < 120.0)
        assert within >= 0.7 * len(errors)

    def test_compensation_changes_what_the_voice_plays(
        self, instrument: Trombolese, compensation: ReedCompensation
    ) -> None:
        controls = Controls(pressure=4200.0, lip_frequency=176.0,
                            alpha=0.5, beta=1.0)
        plain = _sounding_frequency(
            Voice(instrument, reed_parameters=PARAMETERS), controls
        )
        corrected = _sounding_frequency(
            Voice(instrument, reed_parameters=PARAMETERS,
                  reed_compensation=compensation),
            controls,
        )
        target = compensation.target_frequency
        assert abs(1200.0 * np.log2(corrected / target)) < abs(
            1200.0 * np.log2(plain / target)
        )
