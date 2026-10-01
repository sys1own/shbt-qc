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

```
                       Nine-Repository SHBT Ecosystem
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
   Foundations            Physics & Energy          Specialized Vehicle Twins
        |                    Authorities                     |
 [shbt-precision]          [shbt-cf]                 [shbt-ghost]
 [shbt-qc]  (this repo)    [shbt-power]              [shbt-recon]
        |                  [shbt-exotic]             [shbt-sglt]
        |                                            [shbt-warp]
```

| Repository | Domain Role | Integration into `shbt-qc` |
| :--- | :--- | :--- |
| [`sys1own/shbt-precision`](https://github.com/sys1own/shbt-precision) | Arbitrary-precision numerics core | 512-bit MPFR framework, canonical WZW affine branch (26, 8, 312) arithmetic, zero-allocation audit primitives |
| [`sys1own/shbt-cf`](https://github.com/sys1own/shbt-cf) | Cold-fusion reactor & HIL workbench | LANR starter-grid specification, dual-stage TEG enthalpy recovery, 3D two-phase helium thermal-hydraulics |
| [`sys1own/shbt-power`](https://github.com/sys1own/shbt-power) | Fusion plant digital twin | Closed-loop ledger methodology and the 70-gate verification standard |
| [`sys1own/shbt-ghost`](https://github.com/sys1own/shbt-ghost) | Fast interlocks & metric control | CCZ4 stabilization, PCSS optical crowbars, SiC inductive recovery shunts |
| [`sys1own/shbt-exotic`](https://github.com/sys1own/shbt-exotic) | Boundary CFT & spacetime engineering | Boundary state-vector formulations, Heegaard-Floer relabeling, dark-ledger partitioning (η<sub>A</sub> = 10/33, η<sub>D</sub> = 23/33) |
| [`sys1own/shbt-recon`](https://github.com/sys1own/shbt-recon) | Macroscopic states & telemetry | V<sub>unified</sub><sup>macro</sup> tracking, MWPM TQEC decoder, 128-byte dual-cacheline C-ABI, POSIX SPSC rings |
| [`sys1own/shbt-sglt`](https://github.com/sys1own/shbt-sglt) | Relativistic optics & cryogenics | TMSV heterodyne metrology, 2PN beam optics, minimum-jerk kinematics |
| [`sys1own/shbt-warp`](https://github.com/sys1own/shbt-warp) | Holographic warp drive & spacetime engine | Consumes `shbt-qc`'s bare-metal C11 `shbt-os` runtime, 2,112-byte `.stinespring_frame` SRAM layout, SECDED Hamming(72,64) ECC scrubbing, and InP photonic PDK rules for its 128-byte MMIO contract at `0x70000000` |
| **`sys1own/shbt-qc`** (this repo) | Bare-metal runtime & HIL microkernel | Freestanding C11 `shbt-os` execution model, `SHBT-MMIO-1` register map, AVX-512 interlocks |
