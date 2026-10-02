# SHBT-R Quantum Computer: Digital-Twin Simulation Engine and Bare-Metal Microkernel

This repository (`shbt-qc`) contains the engineering physics digital-twin simulation engine, freestanding bare-metal microkernel runtime (`shbt-os`), formal verification suite, EDA exporter toolchain, and CLI orchestrator for the Static Holographic Boundary Theory (SHBT-R) quantum computer architecture.

The platform delivers multi-physics co-simulation across thermal, optoelectronic, and topological QEC domains while embedding the C-runtime microkernel for deterministic hardware-in-the-loop (HIL) execution.

For the companion boundary CFT and precision cosmology theory simulator, see [shbt-precision](https://github.com/sys1own/shbt-precision.git).

---

## 1. System Overview and Core Physics Targets

The SHBT-R architecture models a 312-channel synthetic frequency photonic quantum processor organized into 52 hexameric clusters. The physical and mathematical parameters governing the substrate include:

- **Algebraic Gauge Sector**: Formulated under Wess-Zumino-Witten (WZW) affine levels (k<sub>l</sub>, k<sub>q</sub>, K) = (26, 8, 312) corresponding to SU(2)<sub>26</sub>, SU(3)<sub>8</sub>, and SO(10)<sub>312</sub> with central charges c<sub>vis</sub> = 1325/154, c<sub>parent</sub> = 351/8, and c<sub>dark</sub><sup>comp</sup> = 1197103/362670.
- **Photonic Core Substrate**: 312 active InP/InGaAsP microcavity channels (plus 24 redundant spares, 336 total fabricated) driven by a 72 GHz dynamical Casimir effect (DCE) pump and seeded with 36 GHz two-mode squeezed vacuum states (r=1.52).
- **Boundary Isometry Limit**: Formally bounded code projection operator norm defect ‖W<sup>†</sup> W - P<sub>code</sub>‖<sub>op</sub> ≤ 3.430 × 10<sup>-3</sup>.
- **Cryogenic and Acoustic Interface**: T<sub>ambient</sub> = 4.2 K liquid Helium-4 bath interfacing a single-crystal sapphire substrate (V<sub>substrate</sub> ≥ 1911 cm <sup>3</sup>, Z<sub>sapphire</sub> = 44.178 MRayl) through a nanoporous silica aerogel quarter-wave matching layer (d<sub>m</sub> = 6.395 nm, Z<sub>m</sub> = 1.1512 MRayl) with Kapitza thermal boundary resistance coefficient α<sub>K</sub> = 142.0 W /( m <sup>2</sup> K <sup>4</sup>).
- **Zero-Heap Memory Arena**: 2112-byte contiguous `UnifiedStinespringFrame` pre-allocated in SRAM, partitioned into an active visible register capacity (η<sub>A</sub> = 10/33, 640 bytes) and a dark ledger capacity (η<sub>D</sub> = 23/33, 1472 bytes containing 124 8-byte `ShbtBraidDescriptor` structs).
- **Hardware Register Contract**: Normative `SHBT-MMIO-1` interface at base address `0x70000000` spanning 56 bytes (`0x00`-`0x34`).

---
## System Topology
```
 ╭────────────────────────────────────────────────────────────────────────────────────╮
 │        SHBT-R 312-CHANNEL PHOTONIC QUANTUM PROCESSOR & RUNTIME TOPOLOGY            │
 ╰────────────────────────────────────────────────────────────────────────────────────╯

 ┌── [ 1. PARAMETRIC SQUEEZING & PHOTONIC PROCESSOR CORE ] ───────────────────────────┐
 │                                                                                    │
 │  ╭────────────────────────────────────╮    36 GHz TMSV     ╭─────────────────────╮ │
 │  │ 72 GHz Dynamical Casimir Pump      │── Squeezed Vacuum─►│ 52 Hexamer Arrays   │ │
 │  │ • Non-degenerate Josephson drive   │   (r = 1.52 Seed)  │ • 312 Active InP Ch │ │
 │  │ • Sub-SQL quadrature displacement  │                    │ • 24 Spare Channels │ │
 │  ╰────────────────────────────────────╯                    │ • 336 Total Fab PIC │ │
 │                                                            ╰─────────┬───────────╯ │
 │   Boundary Isometry Defect Limit: ‖W†W − P_code‖_op ≤ 3.430×10⁻³     │ Optical     │
 │   Canonical WZW Branch: (k_l, k_q, K) = (26, 8, 312) | c_vis=1325/154│ Modes       │
 └──────────────────────────────────────────────────────────────────────┼─────────────┘
                                                                        ▼
 ┌── [ 2. CRYOGENIC & PHONONIC SUBSTRATE INTERFACE ] ─────────────────────────────────┐
 │                                                                                    │
 │  ╭──────────────────────────────────────────────────────────────────────────────╮  │
 │  │ 4.2 K Liquid Helium-4 Thermal Bath (T_ambient = 4.200 K)                     │  │
 │  ╰──────────────────────────────────────┬───────────────────────────────────────╯  │
 │                                         │ Kapitza Boundary: α_K = 142.0 W/(m²·K⁴)  │
 │                                         ▼                                          │
 │  ╭──────────────────────────────────────────────────────────────────────────────╮  │
 │  │ Nanoporous Silica Aerogel Quarter-Wave Matching Layer                        │  │
 │  │ • Thickness: d_m = 6.395 nm | Acoustic Impedance: Z_m = 1.1512 MRayl         │  │
 │  ╰──────────────────────────────────────┬───────────────────────────────────────╯  │
 │                                         │ Acoustic Reflection Suppressed           │
 │                                         ▼                                          │
 │  ╭──────────────────────────────────────────────────────────────────────────────╮  │
 │  │ Single-Crystal Sapphire Acoustic Reservoir Substrate                         │  │
 │  │ • Acoustic Impedance: Z_sapphire = 44.178 MRayl | Volume V_sub ≥ 1911 cm³    │  │
 │  │ • Acoustic Transients Damped: τ_decay ≤ 110.00 ns (Brittle σ_max < 350 MPa)  │  │
 │  ╰──────────────────────────────────────────────────────────────────────────────╯  │
 └────────────────────────────────────────────────────────────────────────────────────┘

 ┌── [ 3. 2,112-BYTE ZERO-HEAP STINESPRING ARENA (.stinespring_frame) ] ──────────────┐
 │                                                                                    │
 │  0x000                                0x280                                 0x840  │
 │  ╭──────────────────────────────────────┬───────────────────────────────────────╮  │
 │  │ ACTIVE VISIBLE REGISTER (η_A = 10/33)│ TOPOLOGICAL DARK LEDGER (η_D = 23/33) │  │
 │  │ • Capacity: 640 Bytes (10 Cachelines)│ • Capacity: 1,472 Bytes (23 Cachelines│  │
 │  │ • 160 × binary32 Active Channel Amps │ • 124 × 8-Byte Fibonacci Descriptors  │  │
 │  │ • Trace Preserving: Δ_norm < 10⁻¹²⁰  │ • 480 Bytes SECDED & Syndrome Metadata│  │
 │  ╰──────────────────────────────────────┴───────────────────────────────────────╯  │
 │   Strict 64-Byte Cacheline Alignment | NOLOAD Working Memory Section in SRAM       │
 └───────────────────────────────────────────────────┬────────────────────────────────┘
                                                     │ Memory-Mapped Telemetry
                                                     ▼
 ┌── [ 4. FREESTANDING C11 shbt-os RUNTIME (SHBT-MMIO-1 @ 0x70000000) ] ──────────────┐
 │                                                                                    │
 │  0x00: CR0_CTRL       0x08: DET_MU_HI     0x18: RF_PHASE_V    0x24: ECC_SYNDROME   │
 │  0x04: SR0_STAT       0x0C: DET_MU_LO     0x1C: SHUNT_TRIG    0x2C: REMAP_SRC/DST  │
 │  ────────────────────────────────────────────────────────────────────────────────  │
 │  • Hardware Interface: 56-Byte Register Standard Spanning Offsets 0x00 to 0x34     │
 │  • Deterministic Quench Recovery: t_rec ≤ 120.00 ns (Phase Loop Lock in 9.24 ns)   │
 │  • SECDED Hamming(72,64) ECC: Single-bit correct, double-bit detect on registers   │
 │  • AVX-512 Real-Time Remapping: Vectorized O(1) Givens rotation spare replacement  │
 ╰────────────────────────────────────────────────────────────────────────────────────╯
```

## 2. Repository Layout

The repository is structured to maintain modular separation across multi-physics simulation, CAD tolerance engines, EDA exporters, bare-metal runtime compilation, formal verification, and automated integration testing:

```text
shbt-qc/
├── cli/
│   └── main.py                     # Unifying shbt-tool CLI orchestrator
├── eda/
│   └── exporters/
│       ├── pdk_hexamer_exporter.py # Photonic gdsfactory GDSII mask layout generator
│       └── export_interposer_rf.py # 12-layer Rogers RO4350B interposer Touchstone S2P generator
├── formal/
│   └── formal_verification.py     # Z3 theorem prover logic bounds and safety invariant proofs
├── kernel/
│   ├── include/
│   │   └── shbt_hardware.h        # Normative SHBT-MMIO-1 struct and _Static_assert offsets
│   ├── src/
│   │   ├── shbt_core_runtime.c    # Freestanding C11 microkernel (SECDED ECC, recovery, AVX-512 remap)
│   │   └── shbt_user_mmio.c       # User-space MMIO relocation shim
│   └── linker.ld                  # GNU linker script for freestanding targets (.stinespring_frame)
├── simulator/
│   ├── multi_physics/              # 3D thermal FEA, Kapitza resistance, and vibration models
│   │   └── thermal_fea_sp_sim.py
│   ├── cad_yield/                  # Monte Carlo SPC yield engine & DCE squeezing conversion
│   │   └── spc_photonic_cad.py
│   └── hil_qec/                    # HIL MMIO emulator, MPS QEC decoder, and ctypes bridge
│       ├── hil_mps_qec.py
│       ├── kernel_bridge.py
│       └── test_closed_loop_quench.py
├── tests/
│   ├── reference_test.c           # C reference test suite
│   └── run_all_tests.py           # Master end-to-end integration test runner
└── docs/
    └── paper/
        ├── main.tex               # Self-contained master paper TeX
        └── qc.pdf                 # Compiled manuscript specification

```

---

## 3. Multiphysics Engine & EDA Toolchain

1. **3D Cryogenic Thermal Diffusion (`simulator/multi_physics/`)**: Solves non-linear heat transport at 4.2 K with temperature-dependent specific heat C<sub>v</sub>(T) = γ<sub>1</sub> T + β<sub>3</sub> T<sup>3</sup> and Kapitza interface flux q<sub>K</sub> = α<sub>K</sub> T<sup>3</sup> (T<sub>SOI</sub> - T<sub>InP</sub>).


2. **Photonic CAD & Squeezing Engine (`simulator/cad_yield/`)**: Evaluates SPC tolerance over 52 hexamers (CD 250.12 ± 0.42 nm, sidewall 89.88 ± 0.04<sup>∘</sup>) and models the 36 GHz to 193.41 THz electro-optic conversion matrix.


3. **Photonic GDSII Exporter (`eda/exporters/pdk_hexamer_exporter.py`)**: Uses `gdsfactory` to output wafer-ready GDSII layouts for C<sub>6</sub>-symmetric ring hexamers, directional couplers, and thermo-optic phase shifters.


4. **Interposer RF Exporter (`eda/exporters/export_interposer_rf.py`)**: Exports 2-port Touchstone S-parameter files (S<sub>11</sub>, S<sub>21</sub>) up to 40 GHz for the 12-layer Rogers RO4350B interposer (Z<sub>0</sub> = 50.12 ± 0.80 Ω, FEXT ≤ -70.0 dB).



---

## 4. Bare-Metal Microkernel (`shbt-os`) Execution Model

The `shbt-os` microkernel (`kernel/src/shbt_core_runtime.c`) executes in freestanding C11 with zero heap allocations, zero libc dependencies, and strong ordering over MMIO registers:

* **SECDED Hamming(72,64) ECC**: Computes check bits across 64-bit payloads with single-bit correction and double-bit error detection.


* **Deterministic Quench Recovery (`shbt_recover`)**: Executes the 4-step post-quench recovery sequence (inspect fault arrow assert RF blanking arrow flush/correct ECC arrow request PLL lock) in ≤ 120.00 ns execution budget.


* **AVX-512 Givens Remapping (`shbt_remap`)**: Vectorized O(1) column rotation remapping for dynamic channel replacement.


* **Linker Memory Arena (`kernel/linker.ld`)**: Allocates the 2112-byte `.stinespring_frame` arena on a strict 64-byte boundary with `NOLOAD` semantics.



---

## 5. System CLI Orchestrator (`shbt-tool`)

The unifying CLI interface in `cli/main.py` provides command-line control over all project subsystems:

```bash
# Compile the freestanding C microkernel into shbt_reference.so
python3 cli/main.py build-kernel

# Execute multi-physics FEA and MPS QEC co-simulation
python3 cli/main.py sim

# Export GDSII masks and Touchstone S2P interposer files
python3 cli/main.py export-eda

# Run Z3 formal verification proofs and Pytest closed-loop testbench
python3 cli/main.py verify

```

---

## 6. Build, Verification, and Testing

### 6.1 Prerequisites

* Python 3.10+ with `numpy`, `scipy`, `pytest`, `gdsfactory`, and `z3-solver`.


* GCC ≥ 10 or Clang ≥ 11 with AVX-512 support (`-mavx512f`).



### 6.2 Master Test Suite

To run the master test runner across all 7 verification sub-suites:

```bash
python3 tests/run_all_tests.py

```

### 6.3 Manual Module Verification

```bash
# 1. Test freestanding C microkernel reference driver
gcc -O3 -std=c11 -I kernel/include tests/reference_test.c -o ref_test && ./ref_test

# 2. Run Z3 formal verification proofs
python3 formal/formal_verification.py

# 3. Run closed-loop quench recovery Pytest suite
pytest simulator/hil_qec/test_closed_loop_quench.py

```

## 7. SHBT Ecosystem Code Repository Crosswalk

The Static Holographic Boundary Theory (SHBT) program is organized as a
federated, nine-repository engineering ecosystem. `shbt-qc` supplies the
bare-metal C11 `shbt-os` microkernel runtime, the 2,112-byte
`.stinespring_frame` SRAM layout, SECDED Hamming(72,64) ECC scrubbing,
and the InP photonic PDK rules that define `shbt-warp`'s 128-byte
dual-cacheline MMIO contract at `0x70000000`.

                                  ╭──────────────────────────────────────────╮
                                  │             [shbt-precision]             │
                                  │      Computational Math & Cosmology      │
                                  │     (512-bit MPFR / WZW Characters)      │
                                  ╰────────────────────┬─────────────────────╯
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     ▼                                 ▼                                 ▼
       ╭───────────────────────────╮     ╭───────────────────────────╮     ╭───────────────────────────╮
       │       [shbt-power]        │     │         [shbt-cf]         │     │         [shbt-qc]         │
       │  Commercial Fusion Grid   │     │  1,800-Module LANR Array  │     │ Bare-Metal Microkernel &  │
       │   (8,750 MW p-11B Twin)   │     │    & Thermal-Hydraulics   │     │   Photonic Quantum Bus    │
       ╰─────────────┬─────────────╯     ╰─────────────┬─────────────╯     ╰─────────────┬─────────────╯
                     │                                 │                                 │
                     └────────────────────────┬────────┴─────────────────────────────────┘
                                              ▼
       ╭───────────────────────────────────────────────────────────────────────────────────────────╮
       │                                SPECIALIZED VEHICLE TWINS                                  │
       │                                                                                           │
       │  • shbt-ghost : Reactionless Propulsion & Local Gravity Wells (3+1 CCZ4 / PCSS Crowbars)  │
       │  • shbt-recon : Macroscopic State Translocation Gateway (Stinespring V_macro / 504 Gbps)  │
       │  • shbt-sglt  : Synthetic Gravitational Lensing Telescope (SE-L2 Swarm / TMSV Metrology)  │
       │  • shbt-warp  : Holographic Warp Metric & 3+1D Flight Twin (ADM α=1.0 / 500 TJ Graser)    │
       ╰──────────────────────────────────────────┬────────────────────────────────────────────────╯
                                                  │
                                                  ▼
       ╭───────────────────────────────────────────────────────────────────────────────────────────╮
       │                                       shbt-exotic                                         │
       │                MULTI-PROTOCOL SPACETIME ENGINEERING CO-SIMULATION BENCH                   │
       │                                                                                           │
       │  • Cross-Protocol Field Coupling (Warp + Stasis + Translocation + Wells + Comms)          │
       │  • Global Energy Condition & Ford-Roman Quantum Inequality (QI) Dark-Ledger Auditing      │
       │  • Dynamic 5-Stage Multi-Technology Flight Director & Relativistic PDE Mesh Solvers       │
       ╰───────────────────────────────────────────────────────────────────────────────────────────╯

#### Standardized 9-Pillar Ecosystem Crosswalk Table

| Repository | Domain Role & Platform Scope | Shared Invariants & Interface Contracts |
| :--- | :--- | :--- |
| [`shbt-precision`](https://github.com/sys1own/shbt-precision) | Computational Math & Cosmological Foundation Core | 512-bit MPFR numerics, canonical WZW (26, 8, 312), Δ<sub>fr</sub> ≡ 0, Landauer debt P<sub>debt</sub> = 906.00 kW. |
| [`shbt-power`](https://github.com/sys1own/shbt-power) | Commercial p-¹¹B Aneutronic Fusion Power Plant Twin | 8,750 MW fusion / 7,832.903 MW net export, 70-gate audit, closed-loop thermal ledger, 128-byte SHBT-MMIO-POWER. |
| [`shbt-cf`](https://github.com/sys1own/shbt-cf) | LANR Cold Fusion Reactor Workbench & Thermal-Hydraulics | 1,800-module LANR starter grid (999.054 kW net DC), dual-stage CoSb<sub>3</sub>/ZrNiSn TEG, Kapitza resistance ΔT<sub>K</sub> = 3.546 K. |
| [`shbt-qc`](https://github.com/sys1own/shbt-qc) | Photonic Quantum Computer Twin & C11 Microkernel | Bare-metal C11 shbt-os microkernel, base 56-byte SHBT-MMIO-1 at 0x70000000, SECDED Hamming(72,64) ECC, AVX-512 interlocks. |
| [`shbt-ghost`](https://github.com/sys1own/shbt-ghost) | Ghost Seed Reactionless Propulsion & Metric Stabilization | Sub-2.5 ns PCSS crowbars, 94.20% SiC inductive recovery, 3+1 CCZ4/ADM stabilization (β<sup>i</sup> → 0, \|det(g)+1\| ≤ 10<sup>-12</sup>). |
| [`shbt-recon`](https://github.com/sys1own/shbt-recon) | Macroscopic State Translocation & Gateway Twin | Macroscopic Stinespring dilation (V<sub>unified</sub><sup>macro</sup>), dark ledger η<sub>D</sub> = 23/33, 128-byte C-ABI DMA streaming, 78-gate audit. |
| [`shbt-sglt`](https://github.com/sys1own/shbt-sglt) | Synthetic Gravitational Lensing Telescope (SE-L2) Stack | 2PN relativistic beam optics, TMSV heterodyne metrology (r = 2.50, 21.715 dB), 5th-order minimum-jerk flight profiles. |
| [`shbt-exotic`](https://github.com/sys1own/shbt-exotic) | Multi-Protocol Spacetime Engineering Co-Simulation | Cross-protocol metric coupling (all 6 phenomena), Ford-Roman QI dark-ledger auditing, Heegaard-Floer boundary relabeling. |
| [`shbt-warp`](https://github.com/sys1own/shbt-warp) | Holographic Warp Drive Digital Twin & 3+1D ADM Engine | Alcubierre metric foliation (α = 1.0, γ<sub>ij</sub> = δ<sub>ij</sub>), 500 TJ ¹⁷⁸ᵐ²Hf graser battery (109 TW burst), 128-gate audit, 8 Z3 proofs. |
