# OpenOptiSim — Architecture & Roadmap

> **Vision:** An open-source, modular platform for system-level simulation of optical
> communication links and photonic circuits — the tool that fills the gap between
> low-level EM solvers (Meep) and closed commercial system simulators (OptiSystem, VPIphotonics).

**Status:** design document, pre-implementation.
**Revision:** 2 — restructured, corrected, and expanded from the initial concept draft.

---

## 0. Objectives Summary

**Motivation.** Build a comprehensive, modular, open-source tool for designing and simulating
optical and photonic systems, usable by universities, photonics startups, and small companies.

**Development path.**

* **Priority 1** — a working simulator covering what OptiSystem does at the fundamental level:
  unidirectional optical links, lasers, modulators, fibers, photodetectors, BER/eye analysis.
* **Priority 2** — grow progressively toward VPIphotonics-class capability: coherent systems,
  WDM, and photonic integrated circuits.

**Architectural principles.**

1. **Modular, not monolithic** — decoupled layers with explicit contracts between them.
2. **Engine-first** — the computational core and its scripting API are built and validated
   *before* any GUI work. A wrong engine forces a total rewrite; a wrong GUI does not.
3. **Incremental** — every phase ends in something runnable and validated, not in scaffolding.
4. **Physics you can check** — every block ships with a test against a closed-form analytical
   result. A simulator nobody can verify has no scientific value.

---

## 1. Positioning & Prior Art

The honest competitive picture matters, because it determines what we should build and what
we should reuse.

### 1.1 What already exists

| Project | Domain | Relevance |
| :--- | :--- | :--- |
| **OptiCommPy** | Python; SSFM, coherent DSP, constellation/BER | Covers much of Phases 1–3 numerically. Reference and possible dependency. |
| **GNPy** | Optical transport network planning (TIP) | Network/OSNR-budget layer, not waveform-level. Complementary. |
| **QAMPy** | Coherent DSP (equalization, carrier recovery) | Reference for Phase 3 DSP algorithms. |
| **SAX** | JAX-based S-matrix photonic circuit solver | **This is Phase 4.** Integrate, don't reimplement. |
| **Photontorch**, **Simphony** | Photonic circuit simulation | Same space as SAX; alternative back-ends. |
| **gdsfactory** | Photonic layout + PDK ecosystem | The layout/PDK side of Phase 4. |
| **Meep**, **Tidy3D** | FDTD / full-wave EM | Component-level physics. Feeds models *into* us; not a competitor. |
| **GNU Radio** | Block-based SDR | Excellent architectural reference for dataflow scheduling. No optical physics. |

### 1.2 The actual gap

**The gap is not physics kernels — it is integration and user experience.**

SSFM, coherent DSP, and S-matrix solvers all exist in open source, as libraries, in fragments,
each with its own data model. What does *not* exist is a single tool where an engineer or a
student can drop blocks on a canvas, wire a link, press run, and get an eye diagram and a BER
number with correct optical noise accounting — and then drop into Python for the same model.

This reframes the strategy: our value is the **coherent data model, the execution engine, the
component library, and the UI**. Where a mature open-source kernel exists, we wrap or depend on
it rather than rewriting it.

---

## 2. System Architecture

```text
                    ┌──────────────────────────────────────┐
                    │        Visual Designer (Web UI)      │
                    │   graph editor · parameter forms ·   │
                    │        plots (WebGL/Canvas)          │
                    └───────────────────┬──────────────────┘
                                        │  project JSON + REST / WebSocket
                                        │  (compact, pre-reduced result data)
                    ┌───────────────────▼──────────────────┐
                    │            Session Server            │
                    │   run lifecycle · progress · cache   │
                    └───────────────────┬──────────────────┘
                                        │  in-process
 ┌──────────────────────────────────────▼───────────────────────────────────────┐
 │                             Public Python API                                 │
 │        oosim.Graph · Component · run() · sweep() · analysis helpers           │
 │                    (this is the product for scripting users)                  │
 └──────────────┬─────────────────────────────────────┬─────────────────────────┘
                │                                     │
 ┌──────────────▼───────────────┐    ┌────────────────▼─────────────────┐
 │      Component Library       │◄──►│         Execution Engine         │
 │  plugins · registry · schema │    │  scheduler · sweeps · run graph  │
 └──────────────┬───────────────┘    └────────────────┬─────────────────┘
                │                                     │
 ┌──────────────▼─────────────────────────────────────▼─────────────────────────┐
 │                     Core Data Model + Numerical Kernels                       │
 │   SimulationContext · Signal types · FFT / SSFM / filters / noise             │
 │              back-ends: NumPy → CuPy → (optional) native C++                  │
 └──────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Note on the layering

Analysis and visualization are **not** a layer in this stack. They are *consumers* of results —
some run in the engine (eye-histogram binning, spectrum estimation, BER counting), some run in
the browser (rendering). Placing them as a layer between the GUI and the component library, as
the initial draft did, describes no real dependency and would produce a confused codebase.

### 2.2 Layer responsibilities

1. **Core data model + kernels** — the signal representation, the simulation context, and the
   heavy numerics. Everything else depends on this and it depends on nothing.
2. **Execution engine** — turns a graph of components into an ordered sequence of block
   executions, manages parameter sweeps and multi-run statistics.
3. **Component library** — the physics. Each component is a plugin registered into the engine.
4. **Public Python API** — the stable, documented surface. The GUI is a client of it, with no
   privileged access. If a feature is not reachable from Python, it does not exist.
5. **Session server** — a thin process that owns runs, streams progress, and, critically,
   **reduces result data before it reaches the UI**.
6. **Visual designer** — graph editing and plotting only. No physics, no numerics.

---

## 3. Core Data Model

This section is the most important part of the document. Every one of the four phases below
depends on getting these three types right, and every one of them is expensive to change later.

### 3.1 SimulationContext

Global parameters belong to the run, not to individual signals. Every block reads them; no
block may contradict them.

```python
@dataclass(frozen=True)
class SimulationContext:
    bit_rate: float           # symbol/bit rate [Bd or b/s]
    samples_per_symbol: int   # oversampling factor
    sequence_length: int      # number of symbols in the time window
    seed: int                 # master RNG seed — reproducibility is mandatory

    @property
    def sample_rate(self) -> float:      # Fs
        return self.bit_rate * self.samples_per_symbol

    @property
    def num_samples(self) -> int:        # N
        return self.sequence_length * self.samples_per_symbol

    @property
    def time_window(self) -> float:      # T
        return self.num_samples / self.sample_rate
```

Every sampled signal in a run shares `N`, `Fs`, and the same time origin. This is what makes
blocks composable at all, and it is what the initial draft's per-signal `sampling_rate` /
`num_samples` fields quietly allowed to drift apart.

**Reproducibility rule:** all stochastic blocks (laser phase noise, ASE, shot noise, thermal
noise) draw from independent generators derived deterministically from `seed` and the block's
identity — never from a global RNG. Otherwise adding a block changes the noise in every other
block, and regression tests become impossible.

### 3.2 Signal types and port typing

A link carries more than optical fields. Ports are typed, and the editor refuses invalid
connections at edit time rather than failing at run time.

| Type | Contents | Example producer → consumer |
| :--- | :--- | :--- |
| `Binary` | bit array | PRBS → NRZ driver |
| `Symbol` | complex symbol array + constellation | QAM mapper → pulse shaper |
| `Electrical` | real (or complex I/Q) waveform | Driver → MZM RF port |
| `Optical` | `OpticalSignal` (below) | Laser → MZM → fiber |
| `Metric` | scalar/array results | BER analyzer → report |

An MZM therefore has **two** inputs (one `Optical`, one `Electrical`) and one `Optical` output —
which the original single-`optical_out` manifest example had no way to express.

### 3.3 OpticalSignal — multi-band, with separate noise bins

This is the single most consequential correction to the original design.

A single scalar `center_frequency` can represent exactly one carrier. A 40-channel DWDM system
spanning 4 THz cannot be represented that way without sampling the entire band — an
`Fs` of ~8 THz, which is computationally impossible. Discovering this in Phase 3 means
rewriting the core type and every block that touches it.

```python
@dataclass
class Band:
    """One sampled band: complex envelope in two orthogonal polarizations (Jones vector)."""
    Ex: np.ndarray            # complex64/128, shape (N,)
    Ey: np.ndarray            # complex64/128, shape (N,)
    f0: float                 # band center frequency [Hz]
    fs: float                 # band sample rate [Hz] — usually context.sample_rate

@dataclass
class NoiseBin:
    """Spectrally-resolved noise carried separately from the sampled bands.

    Keeping ASE out of the sampled representation is what allows a realistic Fs:
    the noise floor spans the whole amplifier bandwidth, the signal does not.
    """
    f_start: float            # [Hz]
    f_end: float              # [Hz]
    psd_x: float              # PSD per polarization [W/Hz]
    psd_y: float

@dataclass
class OpticalSignal:
    bands: list[Band]
    noise: list[NoiseBin]
```

Rules the engine enforces:

* Bands are disjoint in frequency. A block that would cause spectral overlap (e.g. a broadening
  nonlinearity, or a MUX with insufficient spacing) either merges the bands onto a common grid
  or raises a diagnostic — it never silently aliases.
* Noise bins convert to sampled noise **only** where a detector or a nonlinear block requires it
  (signal–noise beating), and only over the bandwidth actually being sampled.
* A block that cannot handle multiple bands declares so and the engine reports a clear error,
  rather than processing `bands[0]` and discarding the rest.

### 3.4 Ownership and copy semantics

A 100 km WDM link with 2²⁰ samples per band and several bands is hundreds of megabytes.
Naive value-copying between blocks makes the simulator unusable regardless of language.

**Rule: signals are immutable.** A block receives read-only inputs and returns new outputs.
Arrays are marked non-writeable; blocks that only change metadata (e.g. an ideal attenuator
applied lazily, a frequency shift) share the underlying buffers. Only blocks that genuinely
transform samples allocate.

### 3.5 Numeric precision

Default to **complex64** for sampled fields. Double precision doubles memory traffic and FFT
cost for accuracy that system-level optical simulation rarely needs. Precision is a
`SimulationContext` option so that accuracy studies can switch to complex128; the validation
suite runs in both.

---

## 4. Execution Engine

Absent from the original draft, and the heart of any dataflow simulator.

### 4.1 Execution model

**Block-mode, whole-window-at-once.** Each component is invoked once per run and processes the
entire time window in one call. This matches how OptiSystem behaves, keeps every block a pure
function of its inputs, and makes vectorized NumPy/GPU implementations natural. It is not a
streaming/sample-by-sample scheduler like GNU Radio, and it should not try to be.

### 4.2 Scheduling

* Build the directed graph, validate port types and connectivity.
* **Acyclic case:** topological sort, execute in order. This covers Phases 1–3.
* **Cyclic case:** a cycle is an error *unless* it is broken by an explicit loop-control
  component (recirculating loop, optical feedback) that declares an iteration count. This keeps
  the semantics well-defined instead of leaving loop behavior implicit.
* Intermediate signals are released as soon as their last consumer has run, so peak memory is
  the graph cut width, not the whole graph.

### 4.3 Sweeps and multi-run statistics

First-class from the start, because BER curves are the primary deliverable of this kind of tool:

* **Parameter sweep** — any component parameter may be swept; runs are independent and
  parallelize across processes trivially.
* **Monte-Carlo / repeated runs** — for BER counting at low error rates, with per-run seed
  derivation so results are reproducible and aggregable.
* Results are addressed by `(sweep_point, run_index, block, port)` and stored in a result store
  the UI and Python API both read.

---

## 5. Component Model & Plugin Architecture

### 5.1 Why not `.cpp` + `.json`

The original proposal — a C++ source file plus a JSON manifest per component — has two problems
that would suppress exactly the community contribution it was designed to attract:

1. A contributor must have a C++ toolchain with an ABI matching our build. A researcher with a
   custom fiber model will not do this.
2. Parameters are declared twice (JSON *and* C++). They will drift.

### 5.2 The model

A component is a Python class. Its parameter schema is declared **once**, in code, and the JSON
manifest consumed by the GUI is *generated* from it.

```python
@component(
    name="CW Laser",
    category="Optical Sources",
    version="1.0.0",
)
class CWLaser(Component):
    power      = Param(10.0,   unit="dBm", doc="Average output power")
    wavelength = Param(1550.0, unit="nm",  min=1200, max=1700)
    linewidth  = Param(100.0,  unit="kHz", min=0)

    outputs = {"out": PortType.OPTICAL}

    def run(self, ctx: SimulationContext) -> dict:
        ...
```

* **Units are part of the type**, validated and converted centrally. Unit confusion (dBm vs W,
  nm vs THz) is the most common source of wrong results in this domain.
* Third-party components install as normal Python packages and register through entry points.
  No build step, no ABI.
* A component that needs native speed calls into the kernel layer; it does not itself have to
  be native.

---

## 6. Implementation Language & Performance

### 6.1 Decision: Python-first, with a kernel boundary

The original plan specified a C++20 core with pybind11 bindings. Reconsidered:

* In SSFM — the heaviest workload in the whole tool — roughly 90–95% of runtime is inside FFT.
  That is library code in either language. The remaining element-wise operations vectorize
  well in NumPy.
* GPU acceleration is nearly free from Python via CuPy (drop-in `cupy.fft`), whereas from C++
  it is a substantial cuFFT integration effort.
* The contributor pool for optical-physics models writes Python, not C++20 with pybind11.
* A C++ core plus pybind11 plus Electron is three toolchains for a small team. Realistically it
  is the decision most likely to stall the project in Phase 1.

**Therefore:** implement in Python with NumPy/SciPy, behind a narrow, explicit kernel interface
(FFT, SSFM step, filtering, noise generation). Back-ends are pluggable: NumPy today, CuPy for
GPU, and a native C++/Rust module later if profiling shows a specific kernel needs it. The
architecture keeps that door open; it does not walk through it on day one.

### 6.2 FFT library and licensing — a real constraint

**FFTW is GPL-2.0-or-later** (commercial licenses are sold separately). Linking it makes the
whole project GPL. If the goal is broad industrial adoption under a permissive license, FFTW is
not available to us — and this applies equally to `pyFFTW`.

Use **pocketfft** (BSD-3, and already what `numpy.fft` uses) or `scipy.fft`. Performance is
adequate; for the cases where it is not, the kernel boundary lets a user opt into an
FFTW/MKL back-end in their own GPL-compatible deployment.

### 6.3 SSFM must use adaptive step size

Fixed-step SSFM produces results that look plausible and are wrong. Step size is selected from
a nonlinear-phase criterion (bound the maximum nonlinear phase rotation per step, e.g. a few
milliradians) or a local-error method with a target tolerance. Step count is reported with every
result so that accuracy is auditable.

---

## 7. Validation & Testing Strategy

**Every physics block ships with a test against a closed-form result.** Comparing against
OptiSystem is useful as a one-time sanity check, but it cannot be the basis of automated
testing: it requires a commercial license and is not reproducible in CI.

Golden analytical cases:

| Case | Expected |
| :--- | :--- |
| Lossless, dispersionless, linear fiber | Output is bit-identical to input |
| Attenuation only | `P_out = P_in · exp(-αL)` |
| Gaussian pulse, CD only | `T(z) = T₀·√(1 + (z/L_D)²)`, `L_D = T₀²/\|β₂\|` |
| Lossless SSFM | Energy conserved (Parseval) to tolerance |
| Fundamental soliton (N=1) | Envelope magnitude invariant along propagation |
| Ideal push-pull MZM | `P_out/P_in = cos²(πV / 2V_π)`; extinction ratio matches spec |
| PIN detector | `I = R·P`; shot noise `σ² = 2qIB`; thermal `σ² = 4kTB/R_L` |
| Ideal OOK, Gaussian noise | `BER = ½·erfc(Q/√2)` |
| EDFA | `P_ASE = 2·n_sp·hν·(G−1)·B_o`; OSNR degradation matches noise figure |

Plus: end-to-end regression tests with pinned seeds, and cross-checks against OptiCommPy for
scenarios where an analytical result does not exist.

CI runs the full suite on every commit. Numerical results are pinned with explicit tolerances,
never with exact float equality.

---

## 8. Project File Format

The saved schematic is a versioned, human-readable, git-diffable JSON document (`.oosim`):

```json
{
  "schema_version": 1,
  "context": { "bit_rate": 10e9, "samples_per_symbol": 16,
               "sequence_length": 1024, "seed": 42 },
  "nodes": [
    { "id": "laser1", "type": "oosim.sources.CWLaser",
      "params": { "power": 10.0, "wavelength": 1550.0 },
      "ui": { "x": 100, "y": 200 } }
  ],
  "edges": [
    { "from": ["laser1", "out"], "to": ["mzm1", "optical_in"] }
  ]
}
```

Design rules: `schema_version` from the first commit; UI-only data (positions) segregated from
semantic data so that a diff shows physics changes clearly; a project file must be runnable
headless with no GUI installed.

---

## 9. GUI Strategy

* **Web-first.** The editor is a browser application talking to the session server. An optional
  desktop wrapper (Tauri preferred over Electron for size and memory) is packaging, not
  architecture. A browser-accessible tool removes the largest barrier to trying it.
* **Graph editor:** use an existing canvas library (React Flow / rete.js). The node-graph editor
  is a solved problem and not where this project's value lies.
* **Plot data is reduced in the engine, not in the browser.** This is the key point the original
  draft missed. An eye diagram is a 2D histogram — the engine bins millions of samples and sends
  a small array; the browser never receives raw sample buffers. Same for spectra (send the
  computed PSD on a display grid) and BER curves. With this rule, ordinary Canvas/WebGL2
  rendering is more than sufficient.
* Avoid commercial charting dependencies (e.g. SciChart) — incompatible with an open-source
  project's distribution model. `uPlot`, `regl`, or plain WebGL2 are sufficient once data is
  pre-reduced.

---

## 10. Roadmap

Timelines assume one developer working part-time. They are estimates, not commitments.

### Phase 0 — Foundations (~1 month)

Repository, license, CI, packaging. `SimulationContext`, signal types, port typing, component
base class and registry, scheduler, result store, project file format, test harness.

*Exit criterion:* a two-block graph runs end-to-end from Python and from a `.oosim` file.
No physics yet — and that is the point.

### Phase 1 — MVP: linear optical link (~2–3 months)

**Deliberately smaller than the original Phase 1.** SSFM, PMD, Kerr, APD and a GUI have all been
moved out; delivering a validated linear link quickly matters more than breadth.

* PRBS generator, NRZ driver
* CW laser (power, wavelength, linewidth/phase noise)
* Mach-Zehnder modulator (V_π, extinction ratio, insertion loss, bandwidth)
* Fiber — linear model: attenuation + chromatic dispersion
* PIN photodiode with shot and thermal noise
* Power meter, optical spectrum estimate, eye diagram, Q-factor and BER estimate
* Python API only. No GUI.
* Full analytical validation suite from §7 passing in CI.

*Exit criterion:* a Jupyter notebook reproduces a textbook dispersion-limited-reach curve.

### Phase 1.5 — Nonlinear fiber & amplification (~2 months)

SSFM with adaptive step size, Kerr nonlinearity, PMD, EDFA with gain/NF/saturation/ASE (first
real exercise of the `NoiseBin` model), APD. Soliton and energy-conservation tests.

### Phase 2 — GUI and DSP (~3–4 months)

Graph editor, parameter forms, run control and progress, result plots with engine-side
reduction. DSP blocks: pulse shaping (raised cosine, Gaussian), FIR filtering, resampling,
equalizers (LMS/CMA). Optical spectrum analyzer, eye analyzer, constellation display.
Parameter sweep UI.

### Phase 3 — Coherent & WDM (~6 months)

First real exercise of the multi-band signal model. IQ modulator, M-QAM mapping, local
oscillator, 90° optical hybrid, balanced photodetectors, coherent DSP chain (CD compensation,
CMA, carrier phase recovery). DWDM MUX/DEMUX with crosstalk. 400G/800G reference designs.
CuPy back-end for SSFM.

### Phase 4 — Photonic integrated circuits

**Integrate rather than reimplement.** A bidirectional S-matrix solver is a large project on its
own and mature open-source implementations exist (SAX, Simphony). The work here is a PIC
subsystem block that delegates to such a solver, exposing waveguides, ring resonators, MMI
couplers, Y-junctions and MZIs, plus PDK import via the gdsfactory ecosystem — while the
top-level link simulation remains dataflow.

---

## 11. Licensing & Legal

* **License:** Apache-2.0 recommended for the core — permissive enough for industrial adoption,
  with an explicit patent grant. GPL would guarantee contributions back but sharply limits
  commercial use, which works against the stated goal of serving startups and small companies.
* **Dependency licenses must be checked before adoption,** not after. FFTW (GPL) is the concrete
  case already identified; see §6.2.
* **Do not derive component models, parameter sets, or file formats from OptiSystem or
  VPIphotonics by inspection.** Model from published literature and standards
  (Agrawal, *Nonlinear Fiber Optics*; ITU-T G.652, G.694.1; relevant IEEE 802.3 clauses) and
  cite the source in each component's docstring. This protects the project and makes the models
  reviewable.
* **Naming:** "OpenOptiSim" sits close to *OptiSystem* / Optiwave for a project that explicitly
  cites it as a benchmark. Worth choosing a more distinct name before the first public release,
  while renaming is still cheap.

---

## 12. Immediate Action Items

1. Create the repository: license, `CONTRIBUTING.md`, CI (lint, type-check, pytest), packaging.
2. Implement `SimulationContext`, `Band` / `NoiseBin` / `OpticalSignal`, and the port type system,
   with unit tests — before any component exists.
3. Implement the component base class, parameter/unit system, registry, and JSON manifest
   generation.
4. Implement the scheduler (topological sort, type validation, memory release) and prove it on a
   trivial pass-through graph.
5. Define and freeze `.oosim` schema v1; round-trip test.
6. Build the Phase 1 chain: PRBS → NRZ → CW Laser → MZM → linear fiber → PIN → BER.
7. Stand up the validation suite from §7 in CI, with every tolerance justified in a comment.
8. Write the first example notebook — for a scientific open-source tool, the examples are the
   product.
