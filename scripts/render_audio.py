#!/usr/bin/env python3
"""Render demonstrations of the Trombolese to ``audio/``.

Each file isolates one thing the instrument does. The one that matters is
``bore-glissando``: a single sustained note whose bore morphs from cylinder to
cone underneath it, at constant pitch.

Usage::

    python scripts/render_audio.py [--sample-rate 48000] [--out audio]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trombolese import Trombolese, pitch_neutral  # noqa: E402
from trombolese.synth import Controls, Voice  # noqa: E402
from trombolese.waveguide import tune_to_waveguide  # noqa: E402

SAMPLE_RATE = 48_000

#: Which regime the pitch compensation holds still. Only one can be held --
#: the morph exists to change the ratios between them -- so it should be the
#: one actually being played. Holding the third puts the instrument in a normal
#: register and costs only 7% of bore length, where holding the fundamental
#: would demand 81% and leave every playing register sliding anyway.
PLAYING_REGIME = 3


def envelope(n: int, attack: float = 0.06, release: float = 0.12) -> np.ndarray:
    """A breath envelope: onset, sustain, release."""
    t = np.linspace(0.0, 1.0, n)
    rise = np.clip(t / attack, 0.0, 1.0)
    fall = np.clip((1.0 - t) / release, 0.0, 1.0)
    return rise * fall


def normalise(signal: np.ndarray, peak: float = 0.7) -> np.ndarray:
    largest = np.max(np.abs(signal))
    if largest < 1e-12:
        return signal
    return signal * (peak / largest)


def render(voice: Voice, seconds: float, automation) -> np.ndarray:
    voice.reset()
    return voice.render(int(seconds * voice.sample_rate), automation=automation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "audio")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("tuning the instrument against the waveguide...")
    instrument = tune_to_waveguide(
        pitch_neutral(Trombolese(), regime=PLAYING_REGIME, n_alpha=9),
        regime=PLAYING_REGIME, n_alpha=9, sample_rate=args.sample_rate,
    )
    held = instrument.pitch_compensation.target_f1
    print(f"  holding regime {PLAYING_REGIME} at {held:.2f} Hz across the morph")
    voice = Voice(instrument, sample_rate=float(args.sample_rate))

    takes: dict[str, np.ndarray] = {}

    # 1. The gesture: one held note, the bore morphing underneath it.
    seconds = 8.0
    n = int(seconds * args.sample_rate)
    breath = envelope(n, attack=0.05, release=0.10)

    def bore_glissando(position: float) -> Controls:
        index = min(int(position * (n - 1)), n - 1)
        return Controls(
            pressure=4200.0 * breath[index],
            lip_frequency=held * 0.97,
            alpha=float(np.clip((position - 0.12) / 0.76, 0.0, 1.0)),
            beta=0.0,
        )

    print("rendering bore-glissando (the headline gesture)...")
    takes["bore-glissando"] = render(voice, seconds, bore_glissando)

    # 2. Lipping up through the regimes at the cylindrical end.
    print("rendering brass-regimes...")
    steps = [72.0, 110.0, 150.0, 190.0, 253.0]  # lipping up the regimes
    pieces = []
    for lip in steps:
        m = int(1.1 * args.sample_rate)
        env = envelope(m, attack=0.08, release=0.15)

        def one(position: float, lip=lip, env=env, m=m) -> Controls:
            i = min(int(position * (m - 1)), m - 1)
            return Controls(pressure=4200.0 * env[i], lip_frequency=lip, alpha=0.0)

        pieces.append(render(voice, 1.1, one))
    takes["brass-regimes"] = np.concatenate(pieces)

    # 3. A crescendo, to hear the spectrum open up with breath.
    print("rendering crescendo...")
    seconds = 5.0
    n = int(seconds * args.sample_rate)

    def crescendo(position: float) -> Controls:
        shape = np.clip(position / 0.85, 0.0, 1.0)
        release = np.clip((1.0 - position) / 0.1, 0.0, 1.0)
        return Controls(
            pressure=(1600.0 + 3400.0 * shape) * release,
            lip_frequency=held * 0.97, alpha=0.2,
        )

    takes["crescendo"] = render(voice, seconds, crescendo)

    # 4. The reed morph, lips toward double reed, at a fixed bore.
    print("rendering reed-morph...")
    seconds = 6.0
    n = int(seconds * args.sample_rate)
    breath = envelope(n, attack=0.05, release=0.12)

    def reed_morph(position: float) -> Controls:
        i = min(int(position * (n - 1)), n - 1)
        return Controls(
            pressure=4000.0 * breath[i], lip_frequency=held * 0.97,
            alpha=0.5, beta=float(np.clip((position - 0.1) / 0.8, 0.0, 1.0)),
        )

    takes["reed-morph"] = render(voice, seconds, reed_morph)

    # 5. The slide, for reference: ordinary continuous pitch.
    print("rendering slide...")
    seconds = 4.0
    n = int(seconds * args.sample_rate)
    breath = envelope(n, attack=0.05, release=0.12)

    def slide(position: float) -> Controls:
        i = min(int(position * (n - 1)), n - 1)
        return Controls(
            pressure=4200.0 * breath[i], lip_frequency=150.0, alpha=0.0,
            slide=0.55 * float(np.clip((position - 0.1) / 0.8, 0.0, 1.0)),
        )

    takes["slide"] = render(voice, seconds, slide)

    for name, signal in takes.items():
        path = args.out / f"{name}.wav"
        sf.write(path, normalise(signal), args.sample_rate, subtype="PCM_16")
        print(f"wrote {path}  ({len(signal) / args.sample_rate:.1f} s)")


if __name__ == "__main__":
    main()
