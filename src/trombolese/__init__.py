"""Offline acoustic model of the Trombolese bore.

Stage 1 of the project: work out what the instrument's geometry *does*,
in NumPy, before any of it is committed to realtime DSP. The deliverable is
an answer to "what does this bore resonate at, and how does the cylinder-to-cone
morph move those resonances", accurate enough to design a waveguide against.

Typical use::

    import numpy as np
    from trombolese import Trombolese, find_resonances

    instrument = Trombolese()
    freqs = np.linspace(20.0, 900.0, 80_000)
    regimes = find_resonances(instrument.response(freqs, alpha=0.5))
    print(regimes.freqs, regimes.overblow_ratio)
"""

from __future__ import annotations

from .acoustics import (
    cone_matrix,
    cylinder_matrix,
    input_impedance,
    radiation_impedance,
    wavenumber,
)
from .analysis import (
    HarmonicFit,
    MorphScan,
    Resonances,
    compensate_pitch,
    find_resonances,
    harmonic_fit,
    pitch_neutral,
    scan_morph,
)
from .bore import BoreResponse, PitchCompensation, Segment, Trombolese
from .constants import AIR_20C, Air

__all__ = [
    "Air",
    "AIR_20C",
    "Trombolese",
    "BoreResponse",
    "Segment",
    "Resonances",
    "find_resonances",
    "HarmonicFit",
    "harmonic_fit",
    "MorphScan",
    "scan_morph",
    "PitchCompensation",
    "compensate_pitch",
    "pitch_neutral",
    "wavenumber",
    "radiation_impedance",
    "cylinder_matrix",
    "cone_matrix",
    "input_impedance",
]

__version__ = "0.1.0"
