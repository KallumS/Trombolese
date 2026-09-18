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
PARAMETERS = ReedParameters()


@pytest.fixture(scope="module")
def instrument() -> Trombolese:
    return pitch_neutral(Trombolese(), regime=3, n_alpha=3)


@pytest.fixture(scope="module")
def compensation(instrument: Trombolese) -> ReedCompensation:
    # A small grid, but the production candidate resolution. Coarsening the
    # grid is cheap and harmless; coarsening the *candidates* is neither. At 9
    # candidates the scan steps by a factor of 1.34 between embouchures and
    # walks straight past the right one, which showed up as a fifth of the grid
    # landing over 120 cents out. At the default 19 the step is 1.14 and every
    # point lands. Measured: 10/15 at 9 candidates, 12/15 at 15, 15/15 at 19.
    return compensate_reed_morph(
        instrument, embouchure=176.0, alphas=(0.0, 0.5, 1.0), n_beta=5,
        reed_parameters=PARAMETERS,
    )


class TestLookup:
    @staticmethod
    def table(multipliers: np.ndarray) -> ReedCompensation:
        return ReedCompensation(
            alphas=np.array([0.0, 1.0]), betas=np.array([0.0, 1.0]),
            multipliers=multipliers, slide_offsets=np.zeros((2, 2)),
            target_frequency=200.0,
        )

    def test_multiplier_snaps_across_a_cliff(self) -> None:
        """Regime choice is discrete; blending two choices picks a third."""
        table = self.table(np.array([[1.0, 0.25], [1.0, 0.25]]))
        assert table.multiplier_at(0.0, 0.4) == pytest.approx(1.0)
        assert table.multiplier_at(0.0, 0.6) == pytest.approx(0.25)
        # Never a value that was not measured.
        for beta in np.linspace(0.0, 1.0, 21):
            assert table.multiplier_at(0.0, beta) in (1.0, 0.25)

    def test_multiplier_interpolates_within_a_plateau(self) -> None:
        """Snapping everywhere would step the pitch audibly under a sweep."""
        table = self.table(np.array([[1.00, 1.05], [1.00, 1.05]]))
        assert table.multiplier_at(0.0, 0.5) == pytest.approx(1.025)
        assert table.multiplier_at(0.0, 0.2) == pytest.approx(1.01)

    def test_the_two_regimes_of_the_lookup_do_not_overlap(self) -> None:
        """A plateau is a few percent wide; a regime step is tens of percent."""
        assert 1.05 < ReedCompensation.PLATEAU_TOLERANCE < 1.30

    def test_slide_trim_is_always_interpolated(self) -> None:
        """Within a regime, pitch goes smoothly with length, so this may blend."""
        table = ReedCompensation(
            alphas=np.array([0.0, 1.0]), betas=np.array([0.0, 1.0]),
            multipliers=np.ones((2, 2)),
            slide_offsets=np.array([[0.0, -0.2], [0.0, -0.2]]),
            target_frequency=200.0,
        )
        assert table.slide_at(0.0, 0.5) == pytest.approx(-0.1)

    def test_slide_trim_blends_even_across_a_multiplier_cliff(self) -> None:
        """The cliff rule belongs to regime choice, not to fine tuning."""
        table = ReedCompensation(
            alphas=np.array([0.0, 1.0]), betas=np.array([0.0, 1.0]),
            multipliers=np.array([[1.0, 0.25], [1.0, 0.25]]),
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
    def test_the_uncompensated_morph_still_drags_the_pitch(
        self, instrument: Trombolese
    ) -> None:
        """Smaller than it was, but far too much to leave uncorrected.

        Shrinking the valve's frequency ratio cut this from 833 cents to about
        217. That is the difference between a control that jumps regimes
        mid-sweep and one that merely plays out of tune, and it is why the
        compensation now has something tractable to correct.
        """
        voice = Voice(instrument, reed_parameters=PARAMETERS)
        pitches = [
            _sounding_frequency(
                voice,
                Controls(pressure=4200.0, lip_frequency=176.0, alpha=0.5, beta=beta),
            )
            for beta in (0.0, 1.0)
        ]
        drift = abs(1200.0 * np.log2(pitches[1] / pitches[0]))
        assert 120.0 < drift < 500.0

    def test_the_embouchure_correction_is_material(
        self, compensation: ReedCompensation
    ) -> None:
        """It must actually do something, but its direction is not fixed.

        Two effects pull opposite ways: the valve's resonance rises with beta,
        which sharpens the note, while an inward-striking valve sounds its
        regime flat. Which wins depends on the frequency ratio, so asserting a
        direction would be asserting a particular value of that parameter
        rather than anything about the correction.
        """
        middle = compensation.multipliers[len(compensation.alphas) // 2]
        assert abs(middle[-1] - middle[0]) > 0.02

    def test_the_length_trim_goes_negative(
        self, compensation: ReedCompensation
    ) -> None:
        """An inward-striking valve sounds a regime flat; only shortening fixes it."""
        assert np.min(compensation.slide_offsets) < 0.0

    def test_it_holds_the_pitch_across_its_grid(
        self, instrument: Trombolese, compensation: ReedCompensation
    ) -> None:
        """Nearly every point lands, and the ones that miss, miss by an octave.

        The claim stays statistical. The correction is a table over a control
        space whose regime map has genuine boundaries, and near one a note can
        settle an octave from its neighbour: the full calibration still leaves
        about one point in twenty over 50 cents. Asserting every point would be
        asserting something the method does not deliver.
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
        assert float(np.median(errors)) < 30.0
        within = sum(1 for e in errors if e < 120.0)
        assert within >= 0.9 * len(errors)

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


class TestTheFrequencyRatioKeepsTheMorphSmooth:
    """The ratio decides how far the valve wanders while the morph happens.

    A large one walks the valve's resonance straight across the bore's regime
    boundaries, and the note jumps an octave partway through the sweep. That is
    the root cause of the off-grid failures the compensation could not fix, and
    it is fixed here rather than in the table.
    """

    @staticmethod
    def largest_adjacent_jump(ratio: float, alpha: float = 0.0) -> float:
        parameters = ReedParameters(reed_frequency_ratio=ratio)
        instrument = pitch_neutral(Trombolese(), regime=3, n_alpha=3)
        voice = Voice(instrument, reed_parameters=parameters)
        sounded = [
            _sounding_frequency(
                voice,
                Controls(pressure=4200.0, lip_frequency=176.0,
                         alpha=alpha, beta=float(beta)),
            )
            for beta in np.linspace(0.0, 1.0, 9)
        ]
        jumps = [
            abs(1200.0 * np.log2(b / a))
            for a, b in zip(sounded, sounded[1:])
            if np.isfinite(a) and np.isfinite(b)
        ]
        return max(jumps) if jumps else float("inf")

    def test_the_default_ratio_keeps_the_sweep_continuous(self) -> None:
        assert self.largest_adjacent_jump(ReedParameters().reed_frequency_ratio) < 150.0

    def test_a_large_ratio_walks_the_valve_across_regimes(self) -> None:
        """Guards the choice: 2.0 was the previous default and jumps an octave."""
        assert self.largest_adjacent_jump(2.0) > 400.0

    def test_the_sweep_never_falls_silent(self) -> None:
        """Ratio 1.0 opens a dead zone through the middle of the morph."""
        parameters = ReedParameters()
        instrument = pitch_neutral(Trombolese(), regime=3, n_alpha=3)
        voice = Voice(instrument, reed_parameters=parameters)
        for beta in np.linspace(0.0, 1.0, 9):
            sounded = _sounding_frequency(
                voice,
                Controls(pressure=4200.0, lip_frequency=176.0,
                         alpha=0.0, beta=float(beta)),
            )
            assert np.isfinite(sounded), f"silent at beta={beta:.3f}"
