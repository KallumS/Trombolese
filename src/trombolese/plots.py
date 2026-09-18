"""Figures for the bore study.

Four figures, each answering one question:

1. :func:`plot_bore_profiles` -- what the morph does to the geometry.
2. :func:`plot_impedance` -- what it does to the resonance structure.
3. :func:`plot_overblow` -- how the overblow interval travels from a twelfth
   to an octave.
4. :func:`plot_resonance_ratios` -- how the whole set of regimes migrates from
   the odd-harmonic family to the complete-harmonic one.

Colour follows the data's job. ``alpha`` (and partial number, in figure 4) is a
continuous ordered magnitude, not an identity, so every multi-line figure uses
a single-hue ordinal ramp rather than a categorical palette -- the reader is
meant to see *an ordering*, not five unrelated things. The five blue steps are
the validated ordinal ramp for each surface, which is why the sweeps are drawn
at five values of ``alpha``. Both light and dark variants are selected for
their own surface rather than being an automatic inversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.figure import Figure

from .analysis import MorphScan
from .bore import Trombolese

__all__ = ["Theme", "LIGHT", "DARK", "THEMES", "SWEEP_ALPHAS",
           "plot_bore_profiles", "plot_impedance", "plot_overblow",
           "plot_resonance_ratios", "plot_compensation"]


@dataclass(frozen=True)
class Theme:
    """Surface, ink and ramp for one colour mode."""

    name: str
    surface: str
    text_primary: str
    text_secondary: str
    grid: str
    accent: str
    ramp: tuple[str, ...]

    def apply(self, fig: Figure, ax) -> None:
        fig.patch.set_facecolor(self.surface)
        ax.set_facecolor(self.surface)
        ax.tick_params(colors=self.text_secondary, labelsize=9, length=3, width=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(self.grid)
            ax.spines[side].set_linewidth(0.8)
        ax.grid(True, color=self.grid, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(self.text_secondary)
        ax.yaxis.label.set_color(self.text_secondary)
        ax.title.set_color(self.text_primary)


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    grid="#e6e5e1",
    accent="#e34948",
    ramp=("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    grid="#383835",
    accent="#e66767",
    ramp=("#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"),
)

THEMES = {LIGHT.name: LIGHT, DARK.name: DARK}

#: The morph positions the ordinal ramp is validated for.
SWEEP_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

_LINEWIDTH = 1.8


def _ramp_steps(theme: Theme, count: int) -> list[str]:
    """``count`` steps spread evenly across the theme's ordinal ramp.

    Taking the first N steps of a five-step ramp would crowd the series into
    one end of the lightness range; spreading them keeps both extremes and the
    validated lightness gaps, which only widen under subsampling.
    """
    if count <= 1:
        return [theme.ramp[-2]]
    if count >= len(theme.ramp):
        return list(theme.ramp)
    positions = np.linspace(0, len(theme.ramp) - 1, count)
    return [theme.ramp[int(round(p))] for p in positions]


def _new_axes(theme: Theme, size: tuple[float, float]):
    fig = Figure(figsize=size, dpi=144)
    ax = fig.add_subplot(111)
    theme.apply(fig, ax)
    return fig, ax


def _legend(ax, theme: Theme, title: str | None, loc: str = "best") -> None:
    legend = ax.legend(
        title=title,
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        labelcolor=theme.text_secondary,
        loc=loc,
    )
    if title is not None:
        legend.get_title().set_color(theme.text_secondary)


def _alpha_label(alpha: float) -> str:
    if alpha == 0.0:
        return "0.00  (cylindrical)"
    if alpha == 1.0:
        return "1.00  (conical)"
    return f"{alpha:.2f}"


def plot_bore_profiles(
    instrument: Trombolese,
    alphas: tuple[float, ...] = SWEEP_ALPHAS,
    theme: Theme = LIGHT,
    slide: float = 0.0,
) -> Figure:
    """Bore radius against distance from the lips, in two panels.

    The bell reaches 108 mm while the part the morph actually moves spans
    2-15 mm, so a single pair of axes renders the interesting region as a flat
    line. The upper panel is the whole instrument in silhouette, mirrored about
    its axis so it reads as a bore; the lower panel drops the mirror and clips
    to the bore itself, where the taper is legible.
    """
    fig = Figure(figsize=(9.0, 5.6), dpi=144)
    silhouette, taper = fig.subplots(2, 1, height_ratios=(1.0, 1.15))
    for ax in (silhouette, taper):
        theme.apply(fig, ax)

    # Under pitch compensation the bore is a different length at every morph
    # position, so the enlarged panel has to span the longest of them.
    bore_end = max(instrument.bore_length_at(a) for a in alphas) + slide
    if instrument.include_mouthpiece:
        bore_end += instrument.cup_length + instrument.cup_throat_length

    for colour, alpha in zip(_ramp_steps(theme, len(alphas)), alphas):
        x, radius = instrument.profile(alpha, slide)
        label = _alpha_label(alpha)
        silhouette.plot(x * 100.0, radius * 1000.0, color=colour,
                        linewidth=_LINEWIDTH, label=label)
        silhouette.plot(x * 100.0, -radius * 1000.0, color=colour,
                        linewidth=_LINEWIDTH)
        taper.plot(x * 100.0, radius * 1000.0, color=colour,
                   linewidth=_LINEWIDTH, label=label)

    silhouette.axhline(0.0, color=theme.grid, linewidth=0.8)
    silhouette.set_ylabel("radius (mm)")
    silhouette.set_title("Bore profile across the morph", fontsize=11,
                         loc="left", pad=12,
                  color=theme.text_primary)

    taper.set_xlim(0.0, bore_end * 100.0)
    taper.set_ylim(0.0, instrument.cone_bell_entry_radius * 1000.0 * 1.35)
    taper.set_xlabel("distance from lips (cm)")
    taper.set_ylabel("radius (mm)")
    taper.set_title("the morphing bore, enlarged", fontsize=9.5, loc="left",
                    pad=6, color=theme.text_secondary)
    _legend(taper, theme, "morph  $\\alpha$", loc="upper left")

    fig.tight_layout()
    return fig


def plot_impedance(
    instrument: Trombolese,
    freqs: np.ndarray,
    alphas: tuple[float, ...] = SWEEP_ALPHAS,
    theme: Theme = LIGHT,
    slide: float = 0.0,
) -> Figure:
    """Input-impedance magnitude against frequency, one curve per morph position.

    The peaks are the playing regimes. Reading left to right across the ramp,
    the peaks both move apart at the bottom and rearrange: the cylindrical
    curve's regimes sit at odd multiples of the first, the conical one's at
    every multiple.
    """
    fig, ax = _new_axes(theme, (9.0, 4.6))

    for colour, response in zip(_ramp_steps(theme, len(alphas)),
                                instrument.sweep(freqs, alphas, slide)):
        ax.plot(response.freqs, response.magnitude_db, color=colour,
                linewidth=_LINEWIDTH, label=_alpha_label(response.alpha))

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|Z$_{in}$| (dB re 1 acoustic ohm)")
    ax.set_title("Input impedance seen by the reed", fontsize=11, loc="left", pad=12,
                  color=theme.text_primary)
    ax.set_xlim(float(freqs[0]), float(freqs[-1]))
    _legend(ax, theme, "morph  $\\alpha$")
    fig.tight_layout()
    return fig


def plot_overblow(scan: MorphScan, theme: Theme = LIGHT) -> Figure:
    """The second regime as a multiple of the first, against morph position.

    One series, so no legend: the title names it. The two reference intervals
    are drawn as annotated rules rather than as competing series, since they
    are constants of the physics and not measurements.
    """
    fig, ax = _new_axes(theme, (7.0, 4.2))

    # The curve runs from top-left to bottom-right, so each reference label
    # goes to the end where the curve is furthest away.
    references = (
        (3.0, "twelfth  (cylinder)", 0.985, "right"),
        (2.0, "octave  (cone)", 0.015, "left"),
    )
    for level, label, x, align in references:
        ax.axhline(level, color=theme.grid, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate(label, xy=(x, level), xytext=(0, 6),
                    textcoords="offset points", color=theme.text_secondary,
                    fontsize=9, ha=align)

    ax.plot(scan.alphas, scan.overblow_ratios, color=theme.ramp[-2], linewidth=2.2)
    ax.set_xlabel("morph  $\\alpha$      (0 = cylindrical,  1 = conical)")
    ax.set_ylabel("$f_2 / f_1$")
    ax.set_title("Overblow interval across the morph", fontsize=11, loc="left", pad=12,
                  color=theme.text_primary)
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    return fig


def plot_resonance_ratios(scan: MorphScan, theme: Theme = LIGHT) -> Figure:
    """Each regime as a multiple of the first, against morph position.

    This is the clearest single picture of what the instrument does. At the
    left the curves sit near 1, 3, 5, 7 -- the odd-harmonic family of a closed
    cylinder. At the right they have migrated toward 1, 2, 3, 4 -- the complete
    series of a cone. Every point in between is playable.
    """
    fig, ax = _new_axes(theme, (7.0, 4.8))

    n_series = scan.ratios.shape[1]
    colours = _ramp_steps(theme, n_series)
    for index in range(n_series):
        series = scan.ratios[:, index]
        ax.plot(scan.alphas, series, color=colours[index],
                linewidth=_LINEWIDTH, label=f"regime {index + 1}")
        # Four or fewer series are direct-labelled as well as listed, so
        # identity never rests on colour alone.
        if n_series <= 4 and np.isfinite(series[-1]):
            ax.annotate(f"{index + 1}", xy=(scan.alphas[-1], series[-1]),
                        xytext=(6, -3), textcoords="offset points",
                        color=theme.text_secondary, fontsize=9)

    ax.set_xlabel("morph  $\\alpha$      (0 = cylindrical,  1 = conical)")
    ax.set_ylabel("$f_n / f_1$")
    ax.set_title("Migration of the playing regimes", fontsize=11, loc="left", pad=12,
                  color=theme.text_primary)
    ax.set_xlim(0.0, 1.04)
    _legend(ax, theme, None)
    fig.tight_layout()
    return fig


def plot_compensation(compensation, theme: Theme = LIGHT) -> Figure:
    """Bore length required to hold the fundamental, against morph position.

    One series, so no legend. The instrument has to grow by most of a factor of
    two to keep its pitch as the bore turns conical, because a cone sounds
    ``c / 2L`` where a cylinder of the same length sounds ``c / 4L``.
    """
    fig, ax = _new_axes(theme, (7.0, 4.2))

    ax.plot(compensation.alphas, compensation.lengths, color=theme.ramp[-2],
            linewidth=2.2)
    ax.annotate(
        f"x{compensation.length_ratio:.2f} longer at the conical end",
        xy=(0.985, compensation.lengths[-1]), xytext=(0, -16),
        textcoords="offset points", color=theme.text_secondary,
        fontsize=9, ha="right",
    )

    ax.set_xlabel("morph  $\\alpha$      (0 = cylindrical,  1 = conical)")
    ax.set_ylabel("bore length (m)")
    ax.set_title(
        f"Bore length holding the fundamental at {compensation.target_f1:.1f} Hz",
        fontsize=11, loc="left", pad=12, color=theme.text_primary,
    )
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    return fig
