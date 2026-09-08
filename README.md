# SHBT-R Quantum Computer: Digital-Twin Simulation Engine and Bare-Metal Microkernel

This repository contains the engineering physics digital-twin simulation engine, bare-metal microkernel runtime (`shbt-os`), formal verification suite, and hardware synthesis toolchain for the Static Holographic Boundary Theory (SHBT-R) quantum computer architecture.

The platform delivers multi-physics co-simulation across thermal, optoelectronic, and topological QEC domains while embedding the C-runtime microkernel for deterministic hardware-in-the-loop (HIL) execution.

---

## 1. System Overview and Core Physics Targets

The SHBT-R architecture models a 312-channel synthetic frequency photonic quantum processor organized into 52 hexameric clusters. The physical and mathematical parameters governing the substrate include:

- **Algebraic Gauge Sector**: Formulated under the Wess-Zumino-Witten (WZW) affine levels $(k_l, k_q, K) = (26, 8, 312)$ corresponding to $SU(2)_{26}$, $SU(3)_8$, and $SO(10)_{312}$.
- **Photonic Core Substrate**: 312 active InP/InGaAsP microcavity channels driven by a 72 GHz dynamical Casimir effect (DCE) pump and seeded with 36 GHz two-mode squeezed vacuum states.
- **Cryogenic and Acoustic Interface**: $T_{\text{ambient}} = 4.2\text{ K}$ liquid Helium-4 bath interfacing a single-crystal sapphire substrate ($V_{\text{substrate}} \ge 1911\text{ cm}^3$, $Z_{\text{sapphire}} = 44.178\text{ MRayl}$) through a nanoporous silica aerogel quarter-wave matching layer ($d_m = 6.395\text{ nm}$, $Z_m = 1.1512\text{ MRayl}$).
- **Superconducting Readout Bus**: 32-channel Nb/NbN coplanar waveguide loops operating with characteristic impedance $Z_0 = 50.0\ \Omega$ below critical temperature $T_c = 16.2\text{ K}$.
- **Zero-Heap Memory Arena**: 2112-byte contiguous `UnifiedStinespringFrame` pre-allocated in SRAM, partitioned into an active visible register capacity ($\eta_A = 10/33$, 640 bytes) and a dark ledger capacity ($\eta_D = 23/33$, 1472 bytes).

---

## 2. Repository Layout

The repository is structured to maintain modular separation across multi-physics simulation, CAD tolerance engines, bare-metal runtime compilation, formal verification, and automated integration testing:

```text
shbt-qc/
├── simulator/
│   ├── multi_physics/      # 3D thermal FEA, Kapitza resistance, and vibration jitter models
│   │   └── thermal_fea_sp_sim.py
│   ├── cad_yield/          # Monte Carlo SPC yield engine & microcavity coupling generator
│   │   └── spc_photonic_cad.py
│   └── hil_qec/            # MMIO bus emulator & Matrix Product State (MPS) QEC decoder
│       └── hil_mps_qec.py
├── kernel/
│   ├── include/            # Hardware abstraction C headers and register definitions
│   │   ├── shbt_hardware.h
│   │   └── shbt_hal.h
│   └── src/                # Cryogenic SRAM bare-metal C microkernel sources
│       └── shbt_core_runtime.c
├── formal/                 # Z3 theorem prover logic bounds and TLA+ specifications
│   └── formal_verification.py
├── scripts/                # Toolchain drivers, packaging, and OS exporter
│   └── export_os.py
└── tests/                  # Integrated test runner for end-to-end verification
    └── run_all_tests.py
```

---

## 3. Multiphysics Co-Simulation Engine

The simulation engine models three core physical domains operating in parallel:

### 3.1 3D Cryogenic Thermal Diffusion (`simulator/multi_physics/`)
Thermal diffusion across the 28nm FD-SOI and InP substrate grid ($20 \times 20 \times 10$, $\Delta x = 1\,\mu\text{m}$, $\Delta t = 10\text{ ps}$) at $4.2\text{ K}$ accounts for low-temperature boundary scattering and Kapitza interface resistance $R_K$:

$$\rho C_p(T) \frac{\partial T}{\partial t} = \nabla \cdot \left( k(T) \nabla T \right) + Q_{\text{quench}}(r, t)$$

Material properties obey cubic low-temperature transport laws:
- $c_{p,\text{InP}} = 2.1 \times 10^{-3} T^3 + 10^{-6}\text{ J/(kg K)}$
- $\kappa_{\text{InP}} = 8.5 \times 10^{-4} T^3 + 10^{-5}\text{ W/(m K)}$

### 3.2 Photonic CAD & SPC Yield Engine (`simulator/cad_yield/`)
Evaluates statistical process control (SPC) variation across 312 nanophotonic microcavities governed by inter-cavity evanescent coupling $J_{mn}$ and Kerr non-linearity $\chi^{(3)}$:

$$\frac{d A_m}{d t} = \left( i \Delta \omega_m - \frac{\gamma_m}{2} \right) A_m + i \sum_{n} J_{mn} A_n + i \gamma_{\text{Kerr}} |A_m|^2 A_m + \sqrt{\gamma_{\text{in}}} A_{\text{pump}}$$

### 3.3 Hardware-in-the-Loop QEC Decoder (`simulator/hil_qec/`)
Intercepts MMIO register operations at base address `0x70000000` to mirror real-time thermal transients and optical phase drifts into an inline Matrix Product State (MPS) tensor network decoder operating at bond dimension $\chi = 256$.

---

## 4. Bare-Metal Microkernel (`shbt-os`) Execution Model

The `shbt-os` runtime manages hardware state transitions without POSIX kernel overhead, heap allocations, or virtual memory translation delays.

- **Memory Layout**: Executed entirely inside the 64-byte aligned 2112-byte `UnifiedStinespringFrame` static SRAM arena.
- **GoI Wavefront Scheduler**: Evaluates continuous graph token reductions in $O(1)$ time via matrix wavefront inversion:
  $$\text{EX}(M, U) = (I - U \cdot M)^{-1} \cdot U$$
- **Trace-Free Constraint**: Enforces $\text{Tr}(\hat{O}_{\text{excitation}}) = 0$ across active transformations, eliminating local electric Weyl curvature ($E_{\mu\nu} = 0$) and zeroing framing defects ($\Delta_{\text{fr}} = 0$).
- **ECC Memory Scrubbing**: SECDED Hamming(72,64) hardware scrubbing engine running across memory frames to guarantee effective soft error rate $\text{SER}_{\text{effective}} \le 7.12 \times 10^{-42}\text{ errors/bit-hr}$.

---

## 5. SIMD Telemetry & Emergency Safety Interlock

A deterministic 4-instruction AVX-512 SIMD telemetry loop audits sensor registers in 1.14 ns cycles to enforce eigenvector rigidity detuning floor $\delta_\Phi < 10^{-12}$:

```assembly
vmovaps  zmm0, [rdi]        ; Load 16 sensor lanes (64-byte aligned)
vcmpps   k1, zmm0, zmm1, 23 ; Compare sensor readings against threshold
vmovmskps eax, k1           ; Extract comparison mask
test     eax, eax           ; Test for threshold breach
jnz      .TRIGGER_SHUNT     ; Detuning breach: trigger emergency bias shunt
```

If detuning breaches $\delta_\Phi \ge 10^{-12}$, an electro-optic bias shunt drops DAC control power to zero within 2.5 ns, preventing microcavity quenches and preserving lattice boundary invariants.

---

## 6. Build, Verification, and Execution

### 6.1 Prerequisites
- Python 3.10+ with `numpy`, `scipy`, and `z3-solver`.
- C compiler with AVX-512 support (`gcc` $\ge 10$ or `clang` $\ge 11$).

### 6.2 Running the Verification Suite
To execute the complete end-to-end multi-physics and bare-metal runtime verification suite:

```bash
python tests/run_all_tests.py
```

To run specific subsystem modules individually:

```bash
# 1. Multi-physics thermal and FEA simulation
python simulator/multi_physics/thermal_fea_sp_sim.py

# 2. Photonic CAD and SPC yield engine
python simulator/cad_yield/spc_photonic_cad.py

# 3. Hardware-in-the-loop MPS QEC syndrome decoder
python simulator/hil_qec/hil_mps_qec.py

# 4. Formal Z3 verification checks
python formal/formal_verification.py
```

### 6.3 Compiling and Testing the C Reference Kernel
To compile and execute the reference microkernel runtime test driver:

```bash
gcc -O3 -mavx512f -mavx512bw -I kernel/include tests/reference_test.c -o reference_test
./reference_test
```
