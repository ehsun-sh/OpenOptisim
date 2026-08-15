# OpenOptiSim

**An open-source, modular simulator for optical communication links and photonic systems.**

[![CI](https://github.com/ehsun-sh/OpenOptisim/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsun-sh/OpenOptisim/actions/workflows/ci.yml)
![status](https://img.shields.io/badge/status-pre--alpha-orange)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)

---

> ### ⚠️ Project status: pre-alpha, Phase 0
>
> A complete 10 Gb/s OOK link runs end to end and produces numbers that match theory:
> PRBS → NRZ → CW laser → MZM → fiber (loss + dispersion) → PIN → filter → eye/Q/BER.
> Every physics block is validated against a closed-form result in CI.
>
> Projects save to versioned JSON and sweeps are first-class, so a curve is one call rather than
> a hand-written loop that mutates the graph.
>
> **Not implemented yet:** SSFM/nonlinearity, PMD, amplifiers, equalisers, coherent detection,
> WDM crosstalk, and the GUI. See the [roadmap](#roadmap).
>
> This is not yet a useful simulator. It is a foundation with the expensive decisions made and
> tested. Criticism of those decisions is worth more right now than any feature —
> **[the architecture document](docs/ARCHITECTURE.md)** is where they are argued out.

---

## Try it

```bash
pip install -e ".[dev]" && pytest
```

```python
from oosim import SimulationContext, Graph
from oosim.components import CWLaser, Combiner, Fiber, PowerMeter

ctx = SimulationContext(bit_rate=10e9, samples_per_symbol=16, sequence_length=64)
g = Graph(ctx)

laser = g.add(CWLaser(power=0.0, wavelength=1550.0))  # 0 dBm
fiber = g.add(Fiber(length=80.0, attenuation=0.2))  # 80 km, 0.2 dB/km
meter = g.add(PowerMeter())
g.chain(laser, fiber, meter)

print(g.run()[meter])  # PowerReading(-16.000 dBm; 1550.00nm=-16.000dBm)
```

Two carriers stay two independently sampled bands, which is the point of the signal model:

```python
g = Graph(ctx)
ch1 = g.add(CWLaser(wavelength=1550.0, label="ch1"))
ch2 = g.add(CWLaser(wavelength=1551.0, label="ch2"))
mux = g.add(Combiner(2))
fiber = g.add(Fiber(length=80.0, attenuation=0.2))
meter = g.add(PowerMeter())

g.connect(ch1, mux["in0"])
g.connect(ch2, mux["in1"])
g.chain(mux, fiber, meter)

print(g.run()[meter])
# PowerReading(-12.990 dBm; 1551.00nm=-16.000dBm, 1550.00nm=-16.000dBm)
```

Each band carries its own centre frequency, so channel spacing never enters the sample rate.
Put those two lasers 6 THz apart instead of 125 GHz and nothing about the run changes — which is
exactly what a single-carrier signal model cannot do.

## Results

`python examples/ook_link.py` builds a 10 Gb/s OOK link and characterises it. Abridged output:

```
Receiver sensitivity (back to back)        Dispersion-limited reach (0 dBm launch)
  launch      Q    BER (from Q)  counted     distance     Q    BER (from Q)
  -22 dBm   1.59     5.61e-02    460/8184        0 km   94.75    0.00e+00
  -20 dBm   2.51     6.09e-03     49/8184       40 km    9.26    1.03e-20
  -19 dBm   3.15     8.17e-04      7/8184       60 km    6.48    4.63e-11
  -16 dBm   6.25     2.11e-10   none counted    80 km    3.76    8.58e-05
  -14 dBm   9.83     3.99e-23   none counted   120 km    0.63    2.65e-01
```

Two things are worth reading off that. **Sensitivity is −16 dBm** for a Q of 6 — the right figure
for a PIN into a plain 50 Ω load. **Reach is ~62 km**, and it is set by dispersion, not by loss:
with dispersion switched off the same 60 km span gives Q = 15.4 instead of 6.5. That is the
textbook result for uncompensated 10 G NRZ on standard fiber.

The two columns are also a cross-check on each other. 120 km of 0.2 dB/km is 24 dB, and launching
0 dBm through it gives the same Q as launching −24 dBm back to back. Modulator, fiber, detector,
filter and analyzer all have to agree for that to hold; it is
[a test](tests/test_ber.py), not a coincidence.

Both curves come from `sweep()`, and the same script writes the schematic to
[`examples/ook_link.oosim`](examples/ook_link.oosim) — versioned JSON, diffable, runnable headless.

```python
result = sweep(graph, {("laser", "power"): [-24.0, -21.0, -18.0]}, runs=8)
q = result.metric(analyzer, lambda m: m.q_factor)     # shape (points, runs)
```

Repeats matter more than they look. At −20 dBm, eight runs of the same link give error counts of
37 to 58 — a 50% spread on the thing being measured, while Q itself is stable to ±1%. A single
BER at a marginal operating point is one sample, not an answer.

## What this is

A block-diagram simulator for optical systems: drop components on a canvas, wire a link, run it,
get an eye diagram and a BER number — with correct optical noise accounting. And the same model,
scriptable from Python.

Think **OptiSystem's workflow, VPIphotonics' ambition, open source, and verifiable.**

```
  PRBS ──▶ NRZ ──▶ CW Laser ──▶ MZM ──▶ Fiber ──▶ PIN ──▶ BER / Eye
                                 ▲
                            (V_π, ER, IL)
```

## Why

There is a real gap in open-source photonics tooling, but it is **not** where it first appears
to be.

The physics kernels already exist. SSFM, coherent DSP, and S-matrix circuit solvers are all
available in open source — as libraries, in fragments, each with its own incompatible data model.
What does not exist is a single coherent tool that puts them behind a usable interface with a
sound signal representation underneath.

So the gap is **integration and user experience**, not numerics. That shapes the whole strategy:
where a mature open-source kernel exists, OpenOptiSim wraps or depends on it rather than
rewriting it. The value added here is the data model, the execution engine, the component
library, and the UI.

| Existing tool | What it does | Relationship |
| :--- | :--- | :--- |
| [OptiCommPy](https://github.com/edsonportosilva/OptiCommPy) | Python: SSFM, coherent DSP, BER | Reference & cross-validation target |
| [GNPy](https://github.com/Telecominfraproject/oopt-gnpy) | Optical network planning / OSNR budgets | Complementary — network layer, not waveform layer |
| [QAMPy](https://github.com/ChalmersPhotonicsLab/QAMpy) | Coherent DSP algorithms | Reference for Phase 3 |
| [SAX](https://github.com/gdsfactory/sax) | S-matrix photonic circuit solver | **This is Phase 4** — integrate, don't reimplement |
| [gdsfactory](https://github.com/gdsfactory/gdsfactory) | Photonic layout & PDK ecosystem | PDK path for Phase 4 |
| [Meep](https://github.com/NanoComp/meep) | FDTD / full-wave EM | Feeds component models *in*; not a competitor |
| [GNU Radio](https://www.gnuradio.org/) | Block-based SDR | Architectural reference for dataflow scheduling |

## Design decisions

These are the choices that matter, stated up front so they can be argued with. Full reasoning is
in the [architecture document](docs/ARCHITECTURE.md).

| Decision | Rationale |
| :--- | :--- |
| **Engine before GUI** | A wrong engine forces a total rewrite; a wrong GUI does not. |
| **Multi-band optical signal** from day one | A single scalar carrier frequency cannot express a 40-channel DWDM system without a physically impossible sample rate. Discovering that in Phase 3 means rewriting the core. |
| **Noise carried in spectral bins**, separate from sampled fields | ASE spans the amplifier bandwidth; the signal does not. Sampling both together makes realistic runs impossible. |
| **Block-mode execution** (whole time window per call) | Every component becomes a pure function of its inputs. Vectorizes naturally; not a streaming scheduler. |
| **Python-first**, behind a narrow kernel boundary | ~90–95% of SSFM runtime is inside FFT — library code in any language. GPU via CuPy is nearly free. Contributors who write fiber models write Python. Native kernels stay an option, not a prerequisite. |
| **Typed ports** (Optical / Electrical / Binary / Symbol / Metric) | An MZM has an electrical input. Invalid wiring is rejected at edit time, not at run time. |
| **Immutable signals** | A WDM link is hundreds of MB. Value-copying between blocks makes the tool unusable regardless of language. |
| **Python-package plugins**, parameters declared once | Requiring contributors to match a C++ ABI suppresses exactly the contributions the plugin system exists to attract. |
| **Every physics block validated against a closed-form result** | Comparison against commercial tools needs a licence and is not reproducible in CI. A simulator nobody can verify has no scientific value. |
| **No FFTW** | FFTW is GPL-2.0-or-later. Linking it would make the project GPL. pocketfft (BSD) is used instead. |

## Architecture

```text
                    ┌──────────────────────────────────────┐
                    │        Visual Designer (Web UI)      │
                    │   graph editor · plots (WebGL)       │
                    └───────────────────┬──────────────────┘
                                        │  project JSON + WebSocket
                                        │  (data reduced engine-side)
                    ┌───────────────────▼──────────────────┐
                    │            Session Server            │
                    └───────────────────┬──────────────────┘
 ┌──────────────────────────────────────▼───────────────────────────────────────┐
 │                             Public Python API                                 │
 │        oosim.Graph · Component · run() · sweep()                              │
 └──────────────┬─────────────────────────────────────┬─────────────────────────┘
 ┌──────────────▼───────────────┐    ┌────────────────▼─────────────────┐
 │      Component Library       │◄──►│         Execution Engine         │
 │  plugins · registry · schema │    │  scheduler · sweeps · run graph  │
 └──────────────┬───────────────┘    └────────────────┬─────────────────┘
 ┌──────────────▼─────────────────────────────────────▼─────────────────────────┐
 │                     Core Data Model + Numerical Kernels                       │
 │       SimulationContext · Signals · FFT / SSFM / filters / noise              │
 │              back-ends: NumPy → CuPy → (optional) native                      │
 └──────────────────────────────────────────────────────────────────────────────┘
```

The GUI is a client of the public Python API with no privileged access. **If a feature is not
reachable from Python, it does not exist.**

## The core data model

The part most worth reviewing — see [`src/oosim/signals.py`](src/oosim/signals.py). An optical
signal is not one array of numbers:

```python
@dataclass(frozen=True)
class Band:
    """One sampled band: complex envelope in two orthogonal polarizations (Jones vector)."""

    Ex: np.ndarray  # complex64, shape (N,), read-only
    Ey: np.ndarray
    f0: float  # band centre frequency [Hz]
    fs: float  # band sample rate [Hz]


@dataclass(frozen=True)
class NoiseBin:
    """Spectrally-resolved noise, carried separately from the sampled bands."""

    f_start: float
    f_end: float
    psd_x: float  # [W/Hz] per polarization
    psd_y: float


@dataclass(frozen=True)
class OpticalSignal:
    bands: tuple[Band, ...]
    noise: tuple[NoiseBin, ...]
```

Fields are `sqrt(W)`, so instantaneous power is `|Ex|**2 + |Ey|**2`. Arrays are read-only, which
is what lets metadata-only blocks share buffers instead of copying a span at a time.

Global run parameters (bit rate, oversampling, sequence length, RNG seed) live in a shared
`SimulationContext`, not in individual signals — so blocks cannot silently disagree about the
time window, and results are reproducible.

## Roadmap

| Phase | Scope | Estimate¹ |
| :--- | :--- | :--- |
| **0 — Foundations** ✅ | Signal model, context, port types, component base, registry, scheduler, `.oosim` project format, sweeps, CI | ~1 month |
| **1 — MVP: linear link** *(essentially done)* | ✅ PRBS → NRZ → laser → MZM → fiber (α + CD) → PIN → filter → eye/Q/BER, validated end to end. **Python only, no GUI.** | ~2–3 months |
| **1.5 — Nonlinear & amplified** | Adaptive-step SSFM, Kerr, PMD, EDFA (gain/NF/saturation/ASE), APD | ~2 months |
| **2 — GUI & DSP** | Graph editor, plots, pulse shaping, FIR, equalizers (LMS/CMA), OSA, constellation, sweeps | ~3–4 months |
| **3 — Coherent & WDM** | IQ mod, M-QAM, LO, 90° hybrid, balanced detection, coherent DSP, DWDM + crosstalk, 400G/800G references, CuPy back-end | ~6 months |
| **4 — PIC** | Waveguides, ring resonators, MMI, MZI via integration with an existing S-matrix solver; PDK import | — |

¹ One developer, part-time. Estimates, not commitments.

Phase 1 is deliberately smaller than a first instinct suggests: SSFM, PMD, Kerr, APD and the GUI
are all pushed out of it. Shipping a *validated* linear link quickly matters more than breadth.

## Validation

Every physics block ships with a test against a closed-form result, run in CI
([`tests/test_physics.py`](tests/test_physics.py)):

| Case | Expected | |
| :--- | :--- | :-- |
| Attenuation | `P_out = P_in · 10^(-αL/10)` | ✅ |
| Cascaded spans | Loss is additive in dB | ✅ |
| Source power | Independent of the simulated time window | ✅ |
| Phase noise | Broadens the line, conserves average power | ✅ |
| Multi-carrier | Channels stay separate bands; spacing does not drive `Fs` | ✅ |
| Gaussian pulse, CD only | `T₁/T₀ = √(1 + (z/L_D)²)`, `L_D = T₀²/\|β₂\|` | ✅ |
| Chirped Gaussian | `T₁/T₀ = √((1 + Cβ₂z/T₀²)² + (β₂z/T₀²)²)` — pins the sign of β₂ | ✅ |
| Dispersion compensation | `+D` then `−D` restores the input sample-for-sample | ✅ |
| GVD | Energy conserved (Parseval); β₂ = −Dλ²/2πc per band | ✅ |
| PRBS | Period `2ⁿ−1`; `2ⁿ⁻¹` marks; every n-bit window appears once | ✅ |
| Ideal push-pull MZM | `P_out/P_in = cos²(πV / 2V_π)`; null depth equals the declared ER | ✅ |
| PIN detector | `I = R·P`; shot `σ² = 2qIB`; thermal `σ² = 4kTB/R_L` | ✅ |
| Receiver filter | 3 dB at `B`; noise bandwidth `B·√(π/4ln2)`; zero group delay | ✅ |
| **BER** | `½·erfc(Q/√2)` matched against **directly counted errors**, 10⁻⁴–10⁻¹ | ✅ |
| Link consistency | `L` km of span ≡ launching `α·L` dB lower, end to end | ✅ |
| Lossless SSFM | Energy conserved with nonlinearity | ⬜ |
| Fundamental soliton (N=1) | Envelope magnitude invariant along propagation | ⬜ |
| EDFA | `P_ASE = 2·n_sp·hν·(G−1)·B_o` | ⬜ |

Component models are derived from published literature and standards (Agrawal, *Nonlinear Fiber
Optics*; ITU-T G.652 / G.694.1; relevant IEEE 802.3 clauses), cited in each component's
docstring — never from inspection of commercial tools.

## Contributing

The core is small enough that changing it is still cheap, which makes right now the most useful
time to push back on it. Most valuable first:

* **Review the signal model and scheduler** — [`src/oosim/signals.py`](src/oosim/signals.py),
  [`src/oosim/graph.py`](src/oosim/graph.py), and §3–§4 of the
  [architecture document](docs/ARCHITECTURE.md). If something there is wrong, it is far cheaper
  to fix now than after fifty components depend on it.
* **Tell us if this duplicates existing work.** If a project already does this well, that is worth
  knowing before several months go into it.
* **Describe your use case.** Which components, which measurements, what you currently use and
  what frustrates you about it.
* **Add a component.** A component is a Python class with declared parameters and typed ports —
  see [`src/oosim/components/`](src/oosim/components/) for the pattern. Every physics block needs
  a test against a closed-form result; a component without one will not be merged.

Open an issue for any of the above.

```bash
pip install -e ".[dev]" && ruff check . && ruff format --check . && mypy && pytest
```

## License

[Apache-2.0](LICENSE) — permissive enough for industrial adoption, with an explicit patent grant.

Dependency licences are checked before adoption, not after. The concrete case already identified:
FFTW is GPL-2.0-or-later, so it (and `pyFFTW`) cannot be linked without making the whole project
GPL — pocketfft/`scipy.fft` (BSD) is used instead.

---

**[→ Full architecture & roadmap](docs/ARCHITECTURE.md)**
