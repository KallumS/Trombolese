"""Checks on the morphable excitation and the playable voice."""

from __future__ import annotations

import numpy as np
import pytest

from trombolese import Trombolese, pitch_neutral
from trombolese.constants import AIR_20C
from trombolese.reed import Reed, ReedParameters
from trombolese.synth import Controls, Voice

SAMPLE_RATE = 48_000.0


def dominant_frequency(signal: np.ndarray, sample_rate: float = SAMPLE_RATE) -> float:
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed))
    return float(np.fft.rfftfreq(len(signal), 1.0 / sample_rate)[np.argmax(spectrum)])


class TestReed:
    def test_striking_sign_flips_across_the_morph(self) -> None:
        """Lips blow open, a double reed blows closed. That sign is the morph."""
        reed = Reed(SAMPLE_RATE)
        reed.set_morph(0.0)
        assert reed.striking == pytest.approx(1.0)
        reed.set_morph(1.0)
        assert reed.striking == pytest.approx(-1.0)
        reed.set_morph(0.5)
        assert reed.striking == pytest.approx(0.0)

    def test_embouchure_controls_the_valve_at_both_ends(self) -> None:
        """Pinning the reed end to a fixed frequency would cost all pitch control."""
        reed = Reed(SAMPLE_RATE)
        for beta in (0.0, 0.5, 1.0):
            reed.set_morph(beta, frequency=100.0)
            low = reed._frequency
            reed.set_morph(beta, frequency=200.0)
            assert reed._frequency > low

    def test_a_closed_valve_passes_no_flow(self) -> None:
        reed = Reed(SAMPLE_RATE)
        reed.set_morph(0.0, 110.0)
        reed.opening = 0.0
        injected = reed.step(5000.0, 0.0, 1.0e6)
        assert injected == pytest.approx(0.0)

    def test_flow_solves_the_coupled_equation(self) -> None:
        """The returned wave must satisfy Bernoulli against the bore's impedance.

        This is the check that the quadratic is solved rather than guessed: the
        flow implied by the injected wave has to be the flow that the pressure
        difference across the valve actually produces.
        """
        reed = Reed(SAMPLE_RATE)
        reed.set_morph(0.0, 110.0)
        reed.opening = 3.0e-4
        impedance = 8.4e5
        returning = 120.0
        mouth = 4000.0

        injected = reed.step(mouth, returning, impedance)
        flow = (injected - returning) / impedance
        difference = mouth - 2.0 * returning - impedance * flow
        area = reed._width * 3.0e-4
        expected = area * np.sqrt(2.0 * abs(difference) / AIR_20C.density)
        assert flow == pytest.approx(np.copysign(expected, difference), rel=1e-6)

    def test_flow_reverses_with_the_pressure(self) -> None:
        reed = Reed(SAMPLE_RATE)
        reed.set_morph(0.0, 110.0)
        reed.opening = 3.0e-4
        forward = reed.step(4000.0, 0.0, 8.4e5)
        reed.opening = 3.0e-4
        backward = reed.step(-4000.0, 0.0, 8.4e5)
        assert forward > 0.0 > backward

    def test_the_valve_cannot_pass_through_shut(self) -> None:
        reed = Reed(SAMPLE_RATE)
        reed.set_morph(1.0, 110.0)
        for _ in range(4000):
            reed.step(30_000.0, 0.0, 8.4e5)
            assert reed.opening >= 0.0

    def test_stays_finite_under_absurd_pressure(self) -> None:
        reed = Reed(SAMPLE_RATE)
        reed.set_morph(0.3, 110.0)
        for _ in range(8000):
            injected = reed.step(1.0e6, 0.0, 8.4e5)
            assert np.isfinite(injected)


class TestVoice:
    @pytest.fixture(scope="class")
    @classmethod
    def instrument(cls) -> Trombolese:
        return pitch_neutral(Trombolese(), regime=3, n_alpha=3)

    def test_silent_without_breath(self, instrument: Trombolese) -> None:
        voice = Voice(instrument)
        output = voice.render(4000, Controls(pressure=0.0, lip_frequency=182.0))
        assert np.max(np.abs(output)) < 1e-9

    def test_self_oscillates(self, instrument: Trombolese) -> None:
        """The defining property: it must sustain, not merely ring down."""
        voice = Voice(instrument)
        output = voice.render(
            int(0.5 * SAMPLE_RATE),
            Controls(pressure=4200.0, lip_frequency=176.0, alpha=0.0),
        )
        first = np.sqrt(np.mean(output[12_000:16_000] ** 2))
        last = np.sqrt(np.mean(output[-4000:] ** 2))
        assert last > 1e-3
        assert last > 0.25 * first

    def test_output_is_finite_across_the_control_space(
        self, instrument: Trombolese
    ) -> None:
        voice = Voice(instrument)
        for alpha in (0.0, 1.0):
            for beta in (0.0, 1.0):
                voice.reset()
                output = voice.render(
                    6000,
                    Controls(pressure=5000.0, lip_frequency=176.0,
                             alpha=alpha, beta=beta),
                )
                assert np.all(np.isfinite(output))

    def test_louder_with_more_breath(self, instrument: Trombolese) -> None:
        voice = Voice(instrument)
        levels = []
        for pressure in (2500.0, 4000.0, 5500.0):
            voice.reset()
            output = voice.render(
                int(0.35 * SAMPLE_RATE),
                Controls(pressure=pressure, lip_frequency=176.0, alpha=0.0),
            )
            levels.append(np.sqrt(np.mean(output[-8000:] ** 2)))
        assert levels[0] < levels[1] < levels[2]

    def test_brighter_with_more_breath(self, instrument: Trombolese) -> None:
        """Brass timbre opens up with dynamic; a linear resonator would not."""
        voice = Voice(instrument)
        centroids = []
        for pressure in (2600.0, 5200.0):
            voice.reset()
            output = voice.render(
                int(0.35 * SAMPLE_RATE),
                Controls(pressure=pressure, lip_frequency=176.0, alpha=0.0),
            )[-8192:]
            spectrum = np.abs(np.fft.rfft(output * np.hanning(len(output))))
            freqs = np.fft.rfftfreq(len(output), 1.0 / SAMPLE_RATE)
            centroids.append(np.sum(freqs * spectrum) / np.sum(spectrum))
        assert centroids[1] > centroids[0]

    def test_embouchure_selects_the_regime(self, instrument: Trombolese) -> None:
        """Lipping up must move the note up, as on any brass instrument."""
        voice = Voice(instrument)
        sounded = []
        for lip in (110.0, 182.0, 253.0):
            voice.reset()
            output = voice.render(
                int(0.35 * SAMPLE_RATE),
                Controls(pressure=4200.0, lip_frequency=lip, alpha=0.0),
            )
            sounded.append(dominant_frequency(output[-8192:]))
        assert sounded[0] < sounded[1] < sounded[2]

    def test_the_slide_lowers_the_pitch(self, instrument: Trombolese) -> None:
        """Within one partial, extending the slide flattens the note."""
        voice = Voice(instrument)
        pitches = []
        for slide in (0.0, 0.15, 0.30):
            voice.reset()
            output = voice.render(
                int(0.35 * SAMPLE_RATE),
                Controls(pressure=4200.0, lip_frequency=176.0, slide=slide),
            )
            pitches.append(dominant_frequency(output[-8192:]))
        assert pitches[0] > pitches[1] > pitches[2]

    def test_too_much_slide_cracks_to_the_next_partial(
        self, instrument: Trombolese
    ) -> None:
        """Held embouchure, slide run out: the note breaks upward.

        Not a defect. A trombonist holding one embouchure while extending the
        slide eventually finds the partial has walked away beneath them and the
        note cracks up to the one above. The model does the same thing, and the
        player compensates the same way.
        """
        voice = Voice(instrument)
        pitches = []
        for slide in (0.30, 0.45):
            voice.reset()
            output = voice.render(
                int(0.35 * SAMPLE_RATE),
                Controls(pressure=4200.0, lip_frequency=176.0, slide=slide),
            )
            pitches.append(dominant_frequency(output[-8192:]))
        assert pitches[1] > pitches[0] * 1.1

    def test_the_bore_morphs_without_breaking_the_note(
        self, instrument: Trombolese
    ) -> None:
        """The headline gesture: sweep the bore under a sustained note."""
        voice = Voice(instrument)
        n = int(1.2 * SAMPLE_RATE)

        def automation(position: float) -> Controls:
            return Controls(
                pressure=4200.0, lip_frequency=176.0,
                alpha=float(np.clip((position - 0.15) / 0.7, 0.0, 1.0)),
            )

        output = voice.render(n, automation=automation)
        assert np.all(np.isfinite(output))
        # It must still be sounding at the conical end, not choked off.
        assert np.sqrt(np.mean(output[-6000:] ** 2)) > 1e-3
