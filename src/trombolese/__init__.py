"""Offline acoustic model of the Trombolese bore.

Two layers, and the first is the authority for the second.

**Stage 1** works out what the geometry *does*, in the frequency domain:
:mod:`~trombolese.acoustics` and :mod:`~trombolese.bore` model the instrument
as a chain of transfer matrices, and :mod:`~trombolese.analysis` reads the
playing regimes off the result. Nothing here makes a sound; it establishes what
the instrument is.

**Stage 2** makes it sound. :mod:`~trombolese.waveguide` turns the same
geometry into a scattering ladder in the time domain,
:mod:`~trombolese.reed` supplies an excitation that morphs from brass lips to
a double reed, and :mod:`~trombolese.synth` makes the pair playable. The
waveguide is checked against stage 1's resonances, which is what keeps the
sounding instrument honest to the designed one.

Design study::

    import numpy as np
    from trombolese import Trombolese, find_resonances, pitch_neutral

    instrument = pitch_neutral(Trombolese(), regime=3)
    regimes = find_resonances(
        instrument.response(np.linspace(20.0, 900.0, 20_000), alpha=0.5)
    )
    print(regimes.freqs, regimes.overblow_ratio)

Playing it::

    from trombolese import Controls, Voice, tune_to_waveguide

    voice = Voice(tune_to_waveguide(instrument, regime=3))
    audio = voice.render(48_000, Controls(pressure=4200.0, lip_frequency=176.0))
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
    VentPlacement,
    compensate_pitch,
    evaluate_vent,
    find_resonances,
    find_vent_position,
    fit_register_vent,
    harmonic_fit,
    pitch_neutral,
    scan_morph,
    schedule_vent,
)
from .bore import (
    BoreResponse,
    PitchCompensation,
    RegisterVent,
    Segment,
    Trombolese,
    VentSchedule,
)
from .constants import AIR_20C, Air
from .reed import Reed, ReedParameters
from .synth import Controls, ReedCompensation, Voice, compensate_reed_morph
from .waveguide import (
    SectionedBore,
    Waveguide,
    measure_input_impedance,
    regime_of,
    section_bore,
    tune_to_waveguide,
)

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
    # register vent
    "RegisterVent",
    "VentSchedule",
    "VentPlacement",
    "evaluate_vent",
    "find_vent_position",
    "schedule_vent",
    "fit_register_vent",
    "wavenumber",
    "radiation_impedance",
    "cylinder_matrix",
    "cone_matrix",
    "input_impedance",
    # stage 2: the sounding instrument
    "SectionedBore",
    "Waveguide",
    "section_bore",
    "measure_input_impedance",
    "regime_of",
    "tune_to_waveguide",
    "Reed",
    "ReedParameters",
    "Voice",
    "Controls",
    "ReedCompensation",
    "compensate_reed_morph",
]

__version__ = "0.1.0"
