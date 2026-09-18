"""Checks on the register vent, and on what the conical end's problem really was."""

from __future__ import annotations

import numpy as np
import pytest

from trombolese import Trombolese, find_resonances, pitch_neutral
from trombolese.acoustics import shunt_matrix, tonehole_impedance
from trombolese.analysis import (
    evaluate_vent,
    find_vent_position,
    fit_register_vent,
    schedule_vent,
)
from trombolese.bore import RegisterVent
from trombolese.synth import Controls, Voice
from trombolese.waveguide import section_bore

# Coarser than the study grid: peaks are refined by interpolation, so this
# costs nothing in accuracy and a great deal less in vent-scan time.
FREQS = np.linspace(20.0, 900.0, 6_000)
CANDIDATES = np.linspace(0.10, 0.50, 11)
SAMPLE_RATE = 48_000.0


@pytest.fixture(scope="module")
def vented() -> Trombolese:
    base = pitch_neutral(Trombolese(), regime=3, n_alpha=3)
    return fit_register_vent(base, FREQS, regime=3, n_alpha=3)


class TestToneholeImpedance:
    def test_open_hole_is_an_inertance(self) -> None:
        """Impedance rising with frequency is the whole register-hole mechanism."""
        z = tonehole_impedance(np.array([100.0, 400.0]), 0.004, 0.006, opening=1.0)
        assert abs(z[1]) > abs(z[0])
        assert z[0].imag > 0.0

    def test_shut_hole_is_nearly_invisible(self) -> None:
        """A closed hole is a tiny cavity; it must not short the bore."""
        freqs = np.array([100.0])
        shut = tonehole_impedance(freqs, 0.004, 0.006, opening=0.0)
        open_ = tonehole_impedance(freqs, 0.004, 0.006, opening=1.0)
        assert abs(shut[0]) > 50.0 * abs(open_[0])

    def test_opening_lowers_the_impedance_monotonically(self) -> None:
        freqs = np.array([200.0])
        magnitudes = [
            abs(tonehole_impedance(freqs, 0.004, 0.006, opening=u)[0])
            for u in (0.1, 0.4, 0.7, 1.0)
        ]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_shunt_matrix_is_reciprocal(self) -> None:
        z = tonehole_impedance(FREQS[:50], 0.004, 0.006, opening=1.0)
        assert np.allclose(np.linalg.det(shunt_matrix(z)), 1.0, atol=1e-12)


class TestVentAcoustics:
    def test_a_shut_vent_changes_almost_nothing(self) -> None:
        from dataclasses import replace

        plain = pitch_neutral(Trombolese(), regime=3, n_alpha=3)
        with_vent = replace(plain, vent=RegisterVent(position=0.2))
        without = find_resonances(plain.response(FREQS, alpha=0.5), fmax=900.0,
                                  max_count=3)
        shut = find_resonances(
            with_vent.response(FREQS, alpha=0.5, vent_opening=0.0),
            fmax=900.0, max_count=3,
        )
        assert np.allclose(
            1200.0 * np.log2(shut.freqs / without.freqs), 0.0, atol=10.0
        )

    def test_an_open_vent_removes_the_low_register(self, vented: Trombolese) -> None:
        for alpha in (0.0, 0.5, 1.0):
            shut = find_resonances(
                vented.response(FREQS, alpha=alpha, vent_opening=0.0),
                fmax=900.0, max_count=1,
            )
            opened = find_resonances(
                vented.response(FREQS, alpha=alpha, vent_opening=1.0),
                fmax=900.0, max_count=1,
            )
            assert opened.freqs[0] > 2.0 * shut.freqs[0]

    def test_the_target_regime_survives(self, vented: Trombolese) -> None:
        """A vent that promotes by moving the note has not promoted the note."""
        for alpha in (0.25, 0.5, 0.75, 1.0):
            placement = evaluate_vent(vented, FREQS, alpha, regime=3)
            assert placement.is_usable
            assert abs(placement.detune_cents) < 60.0


class TestVentPlacement:
    def test_the_node_moves_with_the_morph(self) -> None:
        """A hole drilled for the cylinder is in the wrong place for the cone."""
        schedule = schedule_vent(
            Trombolese(vent=RegisterVent()), FREQS, regime=3, n_alpha=3
        )
        assert schedule.positions[-1] > schedule.positions[0]

    def test_the_schedule_does_not_lurch_between_nodes(self) -> None:
        """Later nodes also promote the target; jumping between them is no good."""
        schedule = schedule_vent(
            Trombolese(vent=RegisterVent()), FREQS, regime=3, n_alpha=5
        )
        assert np.all(np.diff(schedule.positions) >= -1e-9)
        assert np.all(np.abs(np.diff(schedule.positions)) < 0.15)

    def test_placement_stays_in_the_bore(self) -> None:
        schedule = schedule_vent(
            Trombolese(vent=RegisterVent()), FREQS, regime=3, n_alpha=3
        )
        assert np.all(schedule.positions > 0.0)
        assert np.all(schedule.positions < 1.0)

    def test_detuning_is_a_constraint_not_a_preference(self) -> None:
        """Destroying the target leaves nothing below it, which must not win."""
        placement = find_vent_position(
            Trombolese(vent=RegisterVent()), FREQS, alpha=1.0, regime=3,
            candidates=CANDIDATES,
        )
        assert abs(placement.detune_cents) <= 60.0


class TestVentInTheWaveguide:
    def test_shut_vent_is_not_placed(self, vented: Trombolese) -> None:
        bore = section_bore(vented, alpha=0.5, vent_opening=0.0)
        assert bore.vent_index is None

    def test_open_vent_sits_inside_the_ladder(self, vented: Trombolese) -> None:
        bore = section_bore(vented, alpha=0.5, vent_opening=1.0)
        assert bore.vent_index is not None
        assert 0 < bore.vent_index < bore.n_sections - 1

    def test_opening_the_vent_lowers_its_corner(self, vented: Trombolese) -> None:
        """A partly open vent shunts less, so its high-pass corner drops."""
        half = section_bore(vented, alpha=0.5, vent_opening=0.5)
        full = section_bore(vented, alpha=0.5, vent_opening=1.0)
        assert full.vent_cutoff > half.vent_cutoff > 0.0

    def test_the_vent_raises_the_sounding_floor(self, vented: Trombolese) -> None:
        """What a register key is for: the low notes stop being available."""
        voice = Voice(vented)
        sounded = {}
        for opening in (0.0, 1.0):
            voice.reset()
            output = voice.render(
                int(0.3 * SAMPLE_RATE),
                Controls(pressure=4200.0, lip_frequency=90.0, alpha=0.5,
                         vent=opening),
            )[-8192:]
            spectrum = np.abs(np.fft.rfft(output * np.hanning(len(output))))
            freqs = np.fft.rfftfreq(len(output), 1.0 / SAMPLE_RATE)
            sounded[opening] = freqs[np.argmax(spectrum)]
        assert sounded[1.0] > 1.25 * sounded[0.0]


class TestWhatTheConicalProblemActuallyWas:
    """The conical end was not unruly because it was conical."""

    def test_compensating_the_pedal_crowds_the_regimes(self) -> None:
        """Holding regime 1 needs a 4.5 m instrument, and length packs regimes in.

        This is what made the conical end hard to control: an instrument tuned
        to hold its pedal note is half as long again, and its regimes sit about
        two thirds as far apart, so the excitation has far less to choose
        between. Compensating for the regime actually played fixes it, and the
        register vent -- a real mechanism, tested above -- turned out not to be
        the answer to this particular question.
        """
        pedal = pitch_neutral(Trombolese(), regime=1, n_alpha=3)
        playing = pitch_neutral(Trombolese(), regime=3, n_alpha=3)

        assert pedal.total_length(1.0) > 1.4 * playing.total_length(1.0)

        def mean_spacing(instrument: Trombolese) -> float:
            found = find_resonances(
                instrument.response(FREQS, alpha=1.0), fmax=900.0, max_count=8
            )
            return float(np.mean(np.diff(found.freqs)))

        assert mean_spacing(playing) > 1.4 * mean_spacing(pedal)

    def test_the_conical_end_is_controllable_when_tuned_for_it(self) -> None:
        instrument = pitch_neutral(Trombolese(), regime=3, n_alpha=3)
        voice = Voice(instrument)
        sounded = []
        for embouchure in (90.0, 130.0, 170.0, 210.0):
            voice.reset()
            output = voice.render(
                int(0.3 * SAMPLE_RATE),
                Controls(pressure=4200.0, lip_frequency=embouchure,
                         alpha=1.0, beta=1.0),
            )[-8192:]
            spectrum = np.abs(np.fft.rfft(output * np.hanning(len(output))))
            freqs = np.fft.rfftfreq(len(output), 1.0 / SAMPLE_RATE)
            sounded.append(freqs[np.argmax(spectrum)])
        assert all(b > a for a, b in zip(sounded, sounded[1:]))
