# OpenOptiSim

**An open-source, modular simulator for optical communication links and photonic systems.**

![status](https://img.shields.io/badge/status-design%20phase-orange)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)

---

> ### ⚠️ Project status: design phase — no code yet
>
> This repository currently contains **the architecture and roadmap only**. There is no working
> simulator here. It is published early so that the design can be reviewed and criticised before
> implementation starts — which is the cheapest possible time to find out that something is wrong.
>
> If you are here to evaluate the idea, the document to read is
> **[the architecture & roadmap](docs/ARCHITECTURE.md)**.
> Feedback on §3 (data model) and §4 (execution engine) is worth more than feedback on anything else.

---

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

The part most worth reviewing. An optical signal is not one array of numbers:

```python
@dataclass
class Band:
    """One sampled band: complex envelope in two orthogonal polarizations (Jones vector)."""
    Ex: np.ndarray   # complex64, shape (N,)
    Ey: np.ndarray
    f0: float        # band center frequency [Hz]
    fs: float        # band sample rate [Hz]

@dataclass
class NoiseBin:
    """Spectrally-resolved noise, carried separately from the sampled bands."""
    f_start: float; f_end: float
    psd_x: float;   psd_y: float      # [W/Hz] per polarization

@dataclass
class OpticalSignal:
    bands: list[Band]
    noise: list[NoiseBin]
```

Global run parameters (bit rate, oversampling, sequence length, RNG seed) live in a shared
`SimulationContext`, not in individual signals — so blocks cannot silently disagree about the
time window, and results are reproducible.

## Roadmap

| Phase | Scope | Estimate¹ |
| :--- | :--- | :--- |
| **0 — Foundations** | Signal model, context, port types, component base, scheduler, project format, CI | ~1 month |
| **1 — MVP: linear link** | PRBS → NRZ → CW laser → MZM → fiber (α + CD) → PIN → eye/BER. **Python only, no GUI.** Full analytical validation suite. | ~2–3 months |
| **1.5 — Nonlinear & amplified** | Adaptive-step SSFM, Kerr, PMD, EDFA (gain/NF/saturation/ASE), APD | ~2 months |
| **2 — GUI & DSP** | Graph editor, plots, pulse shaping, FIR, equalizers (LMS/CMA), OSA, constellation, sweeps | ~3–4 months |
| **3 — Coherent & WDM** | IQ mod, M-QAM, LO, 90° hybrid, balanced detection, coherent DSP, DWDM + crosstalk, 400G/800G references, CuPy back-end | ~6 months |
| **4 — PIC** | Waveguides, ring resonators, MMI, MZI via integration with an existing S-matrix solver; PDK import | — |

¹ One developer, part-time. Estimates, not commitments.

Phase 1 is deliberately smaller than a first instinct suggests: SSFM, PMD, Kerr, APD and the GUI
are all pushed out of it. Shipping a *validated* linear link quickly matters more than breadth.

## Validation

Every physics block ships with a test against a closed-form result, run in CI:

| Case | Expected |
| :--- | :--- |
| Lossless, dispersionless, linear fiber | Output bit-identical to input |
| Attenuation only | `P_out = P_in · exp(-αL)` |
| Gaussian pulse, CD only | `T(z) = T₀·√(1 + (z/L_D)²)`, `L_D = T₀²/\|β₂\|` |
| Lossless SSFM | Energy conserved (Parseval) |
| Fundamental soliton (N=1) | Envelope magnitude invariant along propagation |
| Ideal push-pull MZM | `P_out/P_in = cos²(πV / 2V_π)` |
| PIN detector | `I = R·P`; shot `σ² = 2qIB`; thermal `σ² = 4kTB/R_L` |
| Ideal OOK, Gaussian noise | `BER = ½·erfc(Q/√2)` |
| EDFA | `P_ASE = 2·n_sp·hν·(G−1)·B_o` |

Component models are derived from published literature and standards (Agrawal, *Nonlinear Fiber
Optics*; ITU-T G.652 / G.694.1; relevant IEEE 802.3 clauses), cited in each component's
docstring — never from inspection of commercial tools.

## Contributing

Not open for code contributions yet — there is no code. What is genuinely useful right now:

* **Review the [architecture document](docs/ARCHITECTURE.md)**, especially the
  signal data model (§3) and the execution engine (§4). If something there is wrong, now is when
  it is cheap to fix.
* **Tell us if this duplicates existing work.** If a project already does this well, that is worth
  knowing before several months go into it.
* **Describe your use case.** Which components, which measurements, what you currently use and
  what frustrates you about it.

Open an issue for any of the above.

## License

[Apache-2.0](LICENSE) — permissive enough for industrial adoption, with an explicit patent grant.

Dependency licences are checked before adoption, not after. The concrete case already identified:
FFTW is GPL-2.0-or-later, so it (and `pyFFTW`) cannot be linked without making the whole project
GPL — pocketfft/`scipy.fft` (BSD) is used instead.

---

**[→ Full architecture & roadmap](docs/ARCHITECTURE.md)**
