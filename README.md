# SHBT-R Quantum Computer: Digital-Twin Simulation Engine and Bare-Metal Microkernel

This repository (`shbt-qc`) contains the engineering physics digital-twin simulation engine, freestanding bare-metal microkernel runtime (`shbt-os`), formal verification suite, EDA exporter toolchain, and CLI orchestrator for the Static Holographic Boundary Theory (SHBT-R) quantum computer architecture.

The platform delivers multi-physics co-simulation across thermal, optoelectronic, and topological QEC domains while embedding the C-runtime microkernel for deterministic hardware-in-the-loop (HIL) execution.

For the companion boundary CFT and precision cosmology theory simulator, see [shbt-precision](https://github.com/sys1own/shbt-precision.git).

---

## 1. System Overview and Core Physics Targets

The SHBT-R architecture models a 312-channel synthetic frequency photonic quantum processor organized into 52 hexameric clusters. The physical and mathematical parameters governing the substrate include:

- **Algebraic Gauge Sector**: Formulated under Wess-Zumino-Witten (WZW) affine levels $(k_l, k_q, K) = (26, 8, 312)$ corresponding to $SU(2)_{26}$, $SU(3)_8$, and $SO(10)_{312}$ with central charges $c_{\text{vis}} = 1325/154$, $c_{\text{parent}} = 351/8$, and $c_{\text{dark}}^{\text{comp}} = 1197103/362670$.
- **Photonic Core Substrate**: 312 active InP/InGaAsP microcavity channels (plus 24 redundant spares, 336 total fabricated) driven by a 72 GHz dynamical Casimir effect (DCE) pump and seeded with 36 GHz two-mode squeezed vacuum states ($r=1.52$).
- **Boundary Isometry Limit**: Formally bounded code projection operator norm defect $\Vert{}W^\dagger W - P_{\text{code}}\Vert{}_{\text{op}} \le 3.430 \times 10^{-3}$.
- **Cryogenic and Acoustic Interface**: $T_{\text{ambient}} = 4.2\text{ K}$ liquid Helium-4 bath interfacing a single-crystal sapphire substrate ($V_{\text{substrate}} \ge 1911\text{ cm}^3$, $Z_{\text{sapphire}} = 44.178\text{ MRayl}$) through a nanoporous silica aerogel quarter-wave matching layer ($d_m = 6.395\text{ nm}$, $Z_m = 1.1512\text{ MRayl}$) with Kapitza thermal boundary resistance coefficient $\alpha_K = 142.0\text{ W}/(\text{m}^2\text{K}^4)$.
- **Zero-Heap Memory Arena**: 2112-byte contiguous `UnifiedStinespringFrame` pre-allocated in SRAM, partitioned into an active visible register capacity ($\eta_A = 10/33$, 640 bytes) and a dark ledger capacity ($\eta_D = 23/33$, 1472 bytes containing 124 8-byte `ShbtBraidDescriptor` structs).
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

1. **3D Cryogenic Thermal Diffusion (`simulator/multi_physics/`)**: Solves non-linear heat transport at $4.2\text{ K}$ with temperature-dependent specific heat $C_v(T) = \gamma_1 T + \beta_3 T^3$ and Kapitza interface flux $q_K = \alpha_K T^3 (T_{\text{SOI}} - T_{\text{InP}})$.


2. **Photonic CAD & Squeezing Engine (`simulator/cad_yield/`)**: Evaluates SPC tolerance over 52 hexamers (CD $250.12 \pm 0.42\text{ nm}$, sidewall $89.88 \pm 0.04^\circ$) and models the $36\text{ GHz}$ to $193.41\text{ THz}$ electro-optic conversion matrix.


3. **Photonic GDSII Exporter (`eda/exporters/pdk_hexamer_exporter.py`)**: Uses `gdsfactory` to output wafer-ready GDSII layouts for $C_6$-symmetric ring hexamers, directional couplers, and thermo-optic phase shifters.


4. **Interposer RF Exporter (`eda/exporters/export_interposer_rf.py`)**: Exports 2-port Touchstone S-parameter files ($S_{11}, S_{21}$) up to 40 GHz for the 12-layer Rogers RO4350B interposer ($Z_0 = 50.12 \pm 0.80\ \Omega$, FEXT $\le -70.0\text{ dB}$).



---

## 4. Bare-Metal Microkernel (`shbt-os`) Execution Model

The `shbt-os` microkernel (`kernel/src/shbt_core_runtime.c`) executes in freestanding C11 with zero heap allocations, zero libc dependencies, and strong ordering over MMIO registers:

* **SECDED Hamming(72,64) ECC**: Computes check bits across 64-bit payloads with single-bit correction and double-bit error detection.


* **Deterministic Quench Recovery (`shbt_recover`)**: Executes the 4-step post-quench recovery sequence (inspect fault $\rightarrow$ assert RF blanking $\rightarrow$ flush/correct ECC $\rightarrow$ request PLL lock) in $\le 120.00\text{ ns}$ execution budget.


* **AVX-512 Givens Remapping (`shbt_remap`)**: Vectorized $O(1)$ column rotation remapping for dynamic channel replacement.


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


* GCC $\ge 10$ or Clang $\ge 11$ with AVX-512 support (`-mavx512f`).



### 6.2 Master Test Suite

To run the master test runner across all 7 verification sub-suites:

```bash
python3 tests/run_all_tests.py

```

### 6.3 Manual Module Verification

```bash
# 1. Test freestanding C microkernel reference driver
gcc -O3 -std=c11 -I kernel/include tests/reference_test.c kernel/src/shbt_user_mmio.c -o ref_test && ./ref_test

# 2. Run Z3 formal verification proofs
python3 formal/formal_verification.py

# 3. Run closed-loop quench recovery Pytest suite
pytest simulator/hil_qec/test_closed_loop_quench.py

```
