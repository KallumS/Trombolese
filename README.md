# Trombolese

A physical-modelling wind instrument that crosses a trombone with an oboe, with
one twist neither parent has: the bore itself is a continuous control. A single
parameter morphs it from a cylinder to a cone while the instrument is playing,
which changes not just its timbre but *which pitches it can sound*.

This repository is **stage 1**: an offline acoustic model, in NumPy, that works
out what the geometry actually does before any of it is committed to realtime
DSP. It answers one question — where does this bore resonate, and how does the
morph move those resonances — accurately enough to design a waveguide against.

## Why a bore morph is the interesting parameter

A cylindrical tube driven at a closed end resonates on **odd harmonics** and
overblows at the **twelfth**. A cone resonates on the **complete harmonic
series** and overblows at the **octave**. Those are different instrument
families, and no physical wind instrument can travel between them: the bore is
brass, and brass does not move.

Here it does, continuously, as a playable parameter.

![Migration of the playing regimes](out/regimes-light.png)

At `alpha = 0` the resonances sit at 1, 3, 5, 7 times the fundamental — the
odd-harmonic signature of a closed cylinder. At `alpha = 1` they have migrated
to 1, 2, 3, 4. Every position in between is playable, and the transition is
smooth and monotonic rather than a jump between two states.

The overblow interval — what you get when you tighten your lip rather than move
the slide — travels from a twelfth to an octave:

![Overblow interval across the morph](out/overblow-light.png)

## Results

At 20 °C, with the slide closed:

| alpha | throat | truncation | f₁ (Hz) | f₂/f₁ | regime ratios |
|------:|-------:|-----------:|--------:|------:|---------------|
| 0.00 | 6.95 mm | — | 35.9 | 3.044 | 1, 3.04, 5.08, 7.05 |
| 0.25 | 5.71 mm | 0.637 | 43.0 | 2.588 | 1, 2.59, 4.24, 5.82 |
| 0.50 | 4.47 mm | 0.408 | 50.2 | 2.290 | 1, 2.29, 3.66, 4.94 |
| 0.75 | 3.24 mm | 0.249 | 57.4 | 2.108 | 1, 2.11, 3.26, 4.30 |
| 1.00 | 2.00 mm | 0.133 | 63.9 | 2.023 | 1, 2.02, 3.03, 3.87 |

The fundamental rises by a little over an octave across the sweep, so the morph
is not pitch-neutral — it is a timbral *and* a registral control. Whether to
compensate for that in the instrument's tuning is a playability decision for
stage 2, not an acoustics one.

## Two things the geometry got wrong on the first pass

Both were found by sweeping the model and noticing the resonances did not move
the way the textbook says they should. They are design constraints on the
instrument, not incidental bugs, and they are the main reason this stage exists
at all — both would have been far more expensive to discover in DSP.

**1. The throat cannot carry a long cylindrical section.** The first version
kept a trombone's 0.55 m cylindrical slide, at the throat radius, ahead of the
taper. A truncated cone only behaves like a complete one if whatever replaces
its missing apex has roughly the volume of that apex; 0.55 m of narrow tube has
several times too much, and acts as a transmission line rather than a
compliance besides. The result was an instrument whose upper resonances ignored
the morph completely — only the fundamental moved. The slide is now part of the
taper itself.

**2. A cone cannot reach the bell at a trombone's bore radius.** Opening from
2 mm to 6.95 mm over 2.15 m truncates the cone at 29% of its apex distance, far
too stubby for a harmonic series: the regimes come out at 1, 2.21, 3.51 instead
of 1, 2, 3. Reaching 15 mm instead puts the truncation at 13% and the regimes
at 1, 2.05, 3.14. **The conical limit of this instrument is therefore
necessarily wider-bored than its cylindrical limit** — nearer a euphonium than
a trombone. The bell's *mouth* stays fixed at 108 mm so the radiating aperture,
and hence the radiation impedance, is constant across the morph; only the
bell's entry follows the bore.

![Bore profile across the morph](out/bore-profiles-light.png)

## Running it

```bash
pip install -e ".[dev]"
python scripts/explore_morph.py          # table + figures into out/
python scripts/explore_morph.py --slide 0.3 --no-figures
pytest
```

As a library:

```python
import numpy as np
from trombolese import Trombolese, find_resonances, scan_morph

instrument = Trombolese()
freqs = np.linspace(20.0, 900.0, 20_000)

regimes = find_resonances(instrument.response(freqs, alpha=0.5), fmax=900.0)
print(regimes.freqs, regimes.overblow_ratio)

scan = scan_morph(instrument, freqs)      # the whole sweep at once
```

Every geometric parameter is a field on `Trombolese`, so alternative
instruments are a constructor call:

```python
Trombolese(bore_length=1.2, cone_bell_entry_radius=0.022, bell_radius=0.06)
```

## How it works

The bore is a chain of cylindrical and conical frusta. Each gets a 2×2 transfer
matrix relating pressure and volume flow at its input to those at its output;
chaining is a matrix product, and the input impedance follows from the
radiation impedance at the mouth. Everything is vectorised over frequency.

Peaks of `|Z_in|` are the playing regimes: both a lip reed and a double reed
are pressure-controlled valves behaving close to a flow source, so they
cooperate with *maxima* of input impedance.

Two implementation notes worth keeping:

- The conical matrix is **built from the spherical-wave solutions rather than
  transcribed**. Collecting the two independent solutions into a basis matrix
  `V(x)` gives the segment matrix as `V(x1) V(x2)⁻¹` directly, which is harder
  to get wrong than a memorised closed form, handles contracting cones with no
  special case, and is checked against the cylindrical limit in the tests.
  Referencing the exponentials to the segment's own start, rather than to the
  apex, is what stops a gently tapered segment — whose apex may be kilometres
  away — from overflowing.
- Resonance peaks are refined by **parabolic interpolation**, which matters more
  than grid density: a 10,000-point grid then agrees with an 80,000-point one to
  0.002 cents, at an eighth of the cost.

## What this model does not include

These bound how far the numbers should be trusted. None of them move the
resonance frequencies much, which is what this stage is for, but several matter
a great deal for stage 2.

- **No excitation.** There is no reed, no lip, no breath. This is the resonator
  alone — it says where the instrument *wants* to oscillate, not what comes out.
  The lip-reed-to-double-reed morph is stage 2's problem.
- **Plane and spherical waves only.** No higher-order transverse modes. Safe
  below roughly 14 kHz for the bore, but it degrades inside the bell flare.
- **A crude mouthpiece.** Two cylinders — a cup and a throat — collapsing toward
  a reed staple as the bore turns conical. A real mouthpiece is what pulls a
  brass instrument's stretched modes into an almost-harmonic series, so leaving
  it out would misrepresent the trombone limit; two cylinders get the broad
  effect and not the detail.
- **An approximate radiation impedance.** Exact in the low- and high-frequency
  limits, an engineering fit between them. A waveguide implementation will want
  a proper reflection-function fit.
- **No nonlinear propagation.** At trombone dynamics, nonlinear steepening is a
  large part of the brassy timbre. It barely moves the resonance frequencies,
  so it is deferred — but it is not optional if you want this to *sound* like a
  trombone at forte.
- **No tone holes and no wall vibration.**

## Where this goes next

The planned stack is a Faust DSP core shipped as a **CLAP** plugin.

This model hands stage 2 three things: the bore geometry as a function of
`alpha`, the resonance structure to validate a waveguide against, and the
truncation-ratio constraint that fixes how wide the conical limit has to be.

The next step is a digital waveguide whose delay-line lengths and scattering
junctions reproduce these impedance curves, then a morphable excitation —
lip reed at `alpha = 0`, double reed at `alpha = 1` — driving it. `alpha`,
breath pressure and slide position all want to be per-note continuous controls,
so MPE support belongs in the voice architecture from the start rather than
being retrofitted.

## Layout

```
src/trombolese/
    constants.py   air properties against temperature
    acoustics.py   losses, radiation, transfer matrices
    bore.py        the instrument's geometry and its input impedance
    analysis.py    resonance extraction, harmonic fitting, morph sweeps
    plots.py       the four study figures, light and dark
scripts/
    explore_morph.py
tests/
```

Figures are generated in both light and dark variants; the README shows the
light ones.
