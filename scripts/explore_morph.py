#!/usr/bin/env python3
"""Sweep the bore morph and report what it does to the playing regimes.

Prints a table of resonance frequencies and intervals at each morph position,
and writes the four study figures to ``out/`` in both colour modes.

Usage::

    python scripts/explore_morph.py [--no-figures] [--slide METRES]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trombolese import (  # noqa: E402
    Trombolese,
    find_resonances,
    harmonic_fit,
    pitch_neutral,
    scan_morph,
)
from trombolese.plots import (  # noqa: E402
    SWEEP_ALPHAS,
    THEMES,
    plot_bore_profiles,
    plot_compensation,
    plot_impedance,
    plot_overblow,
    plot_resonance_ratios,
)

#: Analysis band. 20k points is not a resolution compromise: peaks are refined
#: by parabolic interpolation, so resonance frequencies agree with an 80k grid
#: to 0.002 cents, at an eighth of the cost.
FREQS = np.linspace(20.0, 900.0, 20_000)


def report(instrument: Trombolese, slide: float) -> None:
    """Print the resonance table across the morph."""
    compensation = instrument.pitch_compensation
    mode = (
        f"pitch-neutral, fundamental held at {compensation.target_f1:.2f} Hz"
        if compensation is not None
        else "uncompensated -- the morph moves the pitch"
    )
    print(f"Trombolese bore study -- {mode}")
    print(f"slide extension {slide * 100:.0f} cm, "
          f"bell radius {instrument.bell_radius * 1000:.0f} mm, "
          f"air {instrument.air.temperature:.0f} C "
          f"(c = {instrument.air.speed_of_sound:.1f} m/s)")
    print(f"total length {instrument.total_length(0.0, slide):.3f} m at alpha 0, "
          f"{instrument.total_length(1.0, slide):.3f} m at alpha 1")
    print()
    header = (f"{'alpha':>6} {'throat':>8} {'trunc':>6} {'bore L':>8} {'f1':>8} "
              f"{'f2/f1':>7} {'regime ratios (f_n / f_1)':<34} {'harmonicity':>12}")
    print(header)
    print("-" * len(header))

    for alpha in np.linspace(0.0, 1.0, 11):
        response = instrument.response(FREQS, alpha=float(alpha), slide=slide)
        found = find_resonances(response, fmax=900.0, max_count=6)
        fit = harmonic_fit(found.freqs)
        ratios = "  ".join(f"{r:5.2f}" for r in found.ratios)
        flag = "" if fit.converged else "*"
        print(f"{alpha:6.2f} {instrument.throat_radius(alpha) * 1000:7.2f}mm "
              f"{instrument.truncation_ratio(alpha):6.3f} "
              f"{instrument.bore_length_at(alpha):7.3f}m {found.freqs[0]:8.2f} "
              f"{found.overblow_ratio:7.3f} {ratios:<34} "
              f"{fit.rms_cents:9.0f}c{flag}")

    print()
    print("trunc is the cone's truncation ratio -- apex distance over total distance")
    print("to the bell. 0 would be a complete cone; above about 0.2 the bore is too")
    print("stubby to produce a harmonic series.")
    print("harmonicity is the RMS deviation of the regimes from the best-fit")
    print("harmonic series; * marks a fit that did not meet the 60-cent tolerance.")


def draw(instrument: Trombolese, slide: float, out_dir: Path) -> None:
    """Render every figure in every colour mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Both morph figures read from one scan; it is the expensive step.
    scan = scan_morph(instrument, FREQS, n_partials=4, slide=slide)

    for theme in THEMES.values():
        figures = {
            "bore-profiles": plot_bore_profiles(
                instrument, SWEEP_ALPHAS, theme, slide),
            "impedance": plot_impedance(
                instrument, FREQS, SWEEP_ALPHAS, theme, slide),
            "overblow": plot_overblow(scan, theme),
            "regimes": plot_resonance_ratios(scan, theme),
        }
        if instrument.pitch_compensation is not None:
            figures["compensation"] = plot_compensation(
                instrument.pitch_compensation, theme)
        for name, fig in figures.items():
            path = out_dir / f"{name}-{theme.name}.png"
            fig.savefig(path, facecolor=fig.get_facecolor())
            print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true",
                        help="print the table only")
    parser.add_argument("--uncompensated", action="store_true",
                        help="skip pitch compensation, letting the morph move "
                             "the pitch")
    parser.add_argument("--slide", type=float, default=0.0,
                        help="slide extension in metres (default 0)")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "out",
                        help="directory for figures")
    args = parser.parse_args()

    instrument = Trombolese()
    if not args.uncompensated:
        instrument = pitch_neutral(instrument, slide=args.slide)
    report(instrument, args.slide)
    if not args.no_figures:
        print()
        draw(instrument, args.slide, args.out)


if __name__ == "__main__":
    main()
