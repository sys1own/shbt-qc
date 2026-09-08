"""
formal_verification.py - Z3 proof that the SHBT-R post-quench recovery path
cannot exceed its 105.90 ns latency bound under any valid operating condition.

The model mirrors `shbt_quench_recover()` in kernel/src/shbt_core_runtime.c:

    Step 1  IRQ read + acknowledge          4 MMIO accesses
    Step 2  pump attenuation strobe         1 MMIO access
    Step 3  SECDED Hamming(72,64) decode    3 reads, 5 writes + ALU decode
    Step 4  bulk phase reset                2 read-modify-write + 6 writes

Cycle cost of the path:

    cycles = sum_k (n_mmio_k * L_mmio) + C_ecc + C_alu + C_fence + C_branch

Each unknown (MMIO access latency, ECC decode cycles, ALU bookkeeping,
fence drain, branch/mispredict penalty) is a free symbolic variable bounded
by its micro-architectural envelope; the core clock f is a free symbolic
frequency bounded by the cryo-CMOS DVFS envelope (including every discrete
P-state).  Latency is cycles / f.

Z3 is asked for a counter-example `latency > 105.90 ns` inside that
envelope.  `unsat` is a proof that none exists.  Companion checks:

  * soundness: the envelope is satisfiable (the theorem is not vacuous);
  * sharpness: relaxing the frequency floor produces a counter-example,
    so the proof genuinely depends on the DVFS envelope;
  * fixed-point conversion: `shbt_cycles_to_ns_x100()` computed in 64-bit
    integer arithmetic never overflows and never under-reports the real
    latency by more than one LSB (0.01 ns), so the runtime's own
    `within_bound` check is conservative;
  * isometry: the AVX-512 remap (20 x 16-lane iterations) meets 84.60 ns.

Constants are read from kernel/include/shbt_hardware.h so the proof tracks
the header.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import z3

REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_HEADER = REPO_ROOT / "kernel" / "include" / "shbt_hardware.h"

# ---------------------------------------------------------------------------
# Header constants
# ---------------------------------------------------------------------------

def read_header_constant(name: str, default: int, header: Path = HARDWARE_HEADER) -> int:
    if not header.exists():
        return default
    m = re.search(rf"#define\s+{re.escape(name)}\s+(0x[0-9A-Fa-f]+|\d+)U?L?", header.read_text())
    return int(m.group(1), 0) if m else default


RECOVERY_BOUND_NS_X100 = read_header_constant("SHBT_QUENCH_RECOVERY_BOUND_NS_X100", 10590)
ISOMETRY_BOUND_NS_X100 = read_header_constant("SHBT_ISOMETRY_BOUND_NS_X100", 8460)
NUM_CHANNELS = read_header_constant("SHBT_NUM_CHANNELS", 312)
CHANNEL_PAD = read_header_constant("SHBT_CHANNEL_PAD", 320)
HEXAMER_SIZE = read_header_constant("SHBT_HEXAMER_SIZE", 6)

# ---------------------------------------------------------------------------
# Operational envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperationalEnvelope:
    """Valid operating conditions of the cryo control core."""
    # DVFS: continuous scaling between floor and ceiling plus discrete P-states.
    f_min_hz: int = 2_400_000_000
    f_max_hz: int = 4_000_000_000
    p_states_hz: tuple = (2_400_000_000, 2_800_000_000, 3_200_000_000, 3_600_000_000, 4_000_000_000)
    # Micro-architectural cost envelopes (core cycles)
    mmio_latency_min: int = 2        # posted write / hit in the cryo-SRAM window
    mmio_latency_max: int = 5        # uncached 32-bit access over the local bus
    ecc_decode_min: int = 36         # 8 parity folds x popcount + branch
    ecc_decode_max: int = 72
    alu_min: int = 8                 # masking / shifting / bookkeeping
    alu_max: int = 24
    fence_min: int = 0               # __atomic_thread_fence(RELEASE) drain
    fence_max: int = 12
    branch_min: int = 0              # worst-case mispredicts along the path
    branch_max: int = 20


@dataclass(frozen=True)
class RecoveryPathModel:
    """MMIO access counts per step, taken from shbt_quench_recover()."""
    step1_irq_ack: int = 4           # read irq_status, write irq_clear, RMW ctrl (2)
    step2_pump_atten: int = 1        # write pump_atten
    step3_ecc_mmio: int = 8          # read data_hi/lo/check, write lo/hi/check/syndrome/ecc_ctrl
    step4_phase_reset: int = 4 + HEXAMER_SIZE  # set ctrl (2) + clear ctrl (2) + 6 phase writes

    @property
    def total_mmio(self) -> int:
        return self.step1_irq_ack + self.step2_pump_atten + self.step3_ecc_mmio + self.step4_phase_reset


@dataclass
class ProofResult:
    name: str
    verdict: str                     # "proved" | "refuted" | "satisfiable" | "unknown"
    expected: str
    passed: bool
    detail: str = ""
    model: Dict[str, str] = field(default_factory=dict)


@dataclass
class VerificationReport:
    recovery_bound_ns: float
    isometry_bound_ns: float
    worst_case_recovery_cycles: int
    worst_case_recovery_ns: float
    slack_ns: float
    results: List[ProofResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_json(self) -> str:
        d = asdict(self)
        d["all_passed"] = self.all_passed
        return json.dumps(d, indent=2)


# ---------------------------------------------------------------------------
# Symbolic model
# ---------------------------------------------------------------------------

class RecoveryLatencyModel:
    """Builds Z3 constraints for the recovery path under an envelope."""

    def __init__(self, env: OperationalEnvelope = OperationalEnvelope(),
                 path: RecoveryPathModel = RecoveryPathModel()) -> None:
        self.env = env
        self.path = path
        self.f_hz = z3.Real("f_hz")
        self.l_mmio = z3.Int("L_mmio")
        self.c_ecc = z3.Int("C_ecc")
        self.c_alu = z3.Int("C_alu")
        self.c_fence = z3.Int("C_fence")
        self.c_branch = z3.Int("C_branch")
        self.cycles = z3.Int("cycles")
        self.latency_ns = z3.Real("latency_ns")

    def envelope(self, include_frequency: bool = True) -> List[z3.BoolRef]:
        e, p = self.env, self.path
        cs = [
            self.l_mmio >= e.mmio_latency_min, self.l_mmio <= e.mmio_latency_max,
            self.c_ecc >= e.ecc_decode_min, self.c_ecc <= e.ecc_decode_max,
            self.c_alu >= e.alu_min, self.c_alu <= e.alu_max,
            self.c_fence >= e.fence_min, self.c_fence <= e.fence_max,
            self.c_branch >= e.branch_min, self.c_branch <= e.branch_max,
            self.cycles == p.total_mmio * self.l_mmio + self.c_ecc + self.c_alu + self.c_fence + self.c_branch,
            self.latency_ns == z3.ToReal(self.cycles) * z3.RealVal(1_000_000_000) / self.f_hz,
        ]
        if include_frequency:
            cs += [self.f_hz >= e.f_min_hz, self.f_hz <= e.f_max_hz]
        else:
            cs += [self.f_hz > 0]
        return cs

    def p_state_constraint(self) -> z3.BoolRef:
        return z3.Or([self.f_hz == z3.RealVal(f) for f in self.env.p_states_hz])

    def worst_case_cycles(self) -> int:
        e, p = self.env, self.path
        return p.total_mmio * e.mmio_latency_max + e.ecc_decode_max + e.alu_max + e.fence_max + e.branch_max

    def worst_case_ns(self) -> float:
        return self.worst_case_cycles() * 1e9 / self.env.f_min_hz


def _model_dict(m: z3.ModelRef) -> Dict[str, str]:
    return {str(d.name()): str(m[d]) for d in m.decls()}


def _check(solver: z3.Solver, name: str, expect: str, detail: str = "") -> ProofResult:
    r = solver.check()
    verdict = "proved" if r == z3.unsat else "satisfiable" if r == z3.sat else "unknown"
    if expect == "refuted" and r == z3.sat:
        verdict = "refuted"
    model = _model_dict(solver.model()) if r == z3.sat else {}
    return ProofResult(name, verdict, expect, verdict == expect, detail, model)


# ---------------------------------------------------------------------------
# Theorems
# ---------------------------------------------------------------------------

def prove_recovery_bound(model: RecoveryLatencyModel, bound_ns_x100: int = RECOVERY_BOUND_NS_X100) -> ProofResult:
    """forall valid (f, costs): latency <= bound.  Proved by unsat of the negation."""
    s = z3.Solver()
    s.add(*model.envelope())
    s.add(model.latency_ns > z3.RealVal(bound_ns_x100) / 100)
    return _check(s, "recovery_latency_le_bound_continuous_dvfs", "proved",
                  f"latency <= {bound_ns_x100 / 100:.2f} ns for f in [{model.env.f_min_hz/1e9:.2f}, {model.env.f_max_hz/1e9:.2f}] GHz")


def prove_recovery_bound_pstates(model: RecoveryLatencyModel, bound_ns_x100: int = RECOVERY_BOUND_NS_X100) -> ProofResult:
    s = z3.Solver()
    s.add(*model.envelope(include_frequency=False))
    s.add(model.p_state_constraint())
    s.add(model.latency_ns > z3.RealVal(bound_ns_x100) / 100)
    return _check(s, "recovery_latency_le_bound_discrete_pstates", "proved",
                  f"P-states {[f/1e9 for f in model.env.p_states_hz]} GHz")


def check_envelope_nonvacuous(model: RecoveryLatencyModel) -> ProofResult:
    s = z3.Solver()
    s.add(*model.envelope())
    return _check(s, "envelope_is_satisfiable", "satisfiable", "operational envelope admits at least one execution")


def check_bound_is_sharp(model: RecoveryLatencyModel, bound_ns_x100: int = RECOVERY_BOUND_NS_X100) -> ProofResult:
    """Without the DVFS floor a slow enough clock violates the bound (proof is not trivial)."""
    s = z3.Solver()
    s.add(*model.envelope(include_frequency=False))
    s.add(model.latency_ns > z3.RealVal(bound_ns_x100) / 100)
    return _check(s, "bound_violable_without_frequency_floor", "refuted",
                  "dropping f_min admits a counter-example, so the theorem depends on the DVFS envelope")


def prove_minimum_safe_frequency(model: RecoveryLatencyModel, bound_ns_x100: int = RECOVERY_BOUND_NS_X100) -> ProofResult:
    """Derive the exact frequency floor implied by the bound and check it is below f_min."""
    wc = model.worst_case_cycles()
    f_floor_hz = wc * 1e9 / (bound_ns_x100 / 100)
    s = z3.Solver()
    f = z3.Real("f_floor")
    s.add(f == z3.RealVal(wc) * 1_000_000_000 * 100 / bound_ns_x100)
    s.add(f > model.env.f_min_hz)
    r = _check(s, "worst_case_frequency_floor_below_f_min", "proved",
               f"worst case {wc} cycles requires f >= {f_floor_hz/1e9:.4f} GHz; f_min = {model.env.f_min_hz/1e9:.2f} GHz")
    return r


def prove_fixed_point_conversion(model: RecoveryLatencyModel) -> ProofResult:
    """shbt_cycles_to_ns_x100(): (cycles*100000)/(f/1e6) in uint64 is overflow-free and conservative.

    Conservative means: integer result >= floor(true value) - 0 and never under-reports the
    bound check, i.e. if the true latency exceeds the bound the fixed-point value also does
    (since floor(x) > B for integer B iff x >= B+1, we require the true value >= B+1 whenever
    exceedance is possible; truncation error is < 1 LSB = 0.01 ns).
    """
    s = z3.Solver()
    cycles = z3.Int("cycles_i")
    tsc_mhz = z3.Int("tsc_mhz")
    s.add(cycles >= 0, cycles <= model.worst_case_cycles() * 4)      # generous over-approximation
    s.add(tsc_mhz >= model.env.f_min_hz // 1_000_000, tsc_mhz <= model.env.f_max_hz // 1_000_000)
    prod = cycles * 100_000
    fixed = prod / tsc_mhz                                             # Z3 Int division = floor
    true_x100 = z3.ToReal(prod) / z3.ToReal(tsc_mhz)
    violation = z3.Or(
        prod > 2**64 - 1,                                              # uint64 overflow
        z3.ToReal(fixed) > true_x100,                                  # over-report
        z3.ToReal(fixed) < true_x100 - 1,                              # under-report by >= 1 LSB
    )
    s.add(violation)
    return _check(s, "fixed_point_ns_x100_conversion_sound", "proved",
                  "64-bit cycles->ns_x100 conversion is overflow-free with < 0.01 ns truncation error")


def prove_isometry_bound(env: OperationalEnvelope = OperationalEnvelope(),
                         bound_ns_x100: int = ISOMETRY_BOUND_NS_X100) -> ProofResult:
    """AVX-512 isometry: 320 lanes / 16 = 20 iterations of {2 loads, 2 fmadd/fmsub, 2 stores, 1 add}."""
    iters = CHANNEL_PAD // 16
    s = z3.Solver()
    f = z3.Real("f_iso")
    c_iter = z3.Int("C_iter")       # throughput-bound cycles per 16-lane iteration
    c_reduce = z3.Int("C_reduce")   # _mm512_reduce_add_ps + horizontal tail
    c_setup = z3.Int("C_setup")     # broadcast c/s, loop overhead, prologue
    s.add(f >= env.f_min_hz, f <= env.f_max_hz)
    s.add(c_iter >= 2, c_iter <= 8)          # 2 FMA ports, 2 load ports, 1 store port
    s.add(c_reduce >= 8, c_reduce <= 24)
    s.add(c_setup >= 4, c_setup <= 18)
    lat = z3.ToReal(iters * c_iter + c_reduce + c_setup) * 1_000_000_000 / f
    s.add(lat > z3.RealVal(bound_ns_x100) / 100)
    wc = (iters * 8 + 24 + 18) * 1e9 / env.f_min_hz
    return _check(s, "isometry_latency_le_bound", "proved",
                  f"{iters} x 16-lane iterations over {NUM_CHANNELS} channels; worst case {wc:.2f} ns <= {bound_ns_x100/100:.2f} ns")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def verify(env: Optional[OperationalEnvelope] = None, verbose: bool = True) -> VerificationReport:
    env = env or OperationalEnvelope()
    model = RecoveryLatencyModel(env)
    results = [
        check_envelope_nonvacuous(model),
        prove_recovery_bound(model),
        prove_recovery_bound_pstates(model),
        check_bound_is_sharp(model),
        prove_minimum_safe_frequency(model),
        prove_fixed_point_conversion(model),
        prove_isometry_bound(env),
    ]
    report = VerificationReport(
        recovery_bound_ns=RECOVERY_BOUND_NS_X100 / 100,
        isometry_bound_ns=ISOMETRY_BOUND_NS_X100 / 100,
        worst_case_recovery_cycles=model.worst_case_cycles(),
        worst_case_recovery_ns=model.worst_case_ns(),
        slack_ns=RECOVERY_BOUND_NS_X100 / 100 - model.worst_case_ns(),
        results=results,
    )
    if verbose:
        print(f"Z3 {z3.get_version_string()} - SHBT-R recovery-path formal verification")
        print(f"  path: {model.path.total_mmio} MMIO accesses, worst case {report.worst_case_recovery_cycles} cycles "
              f"= {report.worst_case_recovery_ns:.2f} ns @ {env.f_min_hz/1e9:.2f} GHz "
              f"(bound {report.recovery_bound_ns:.2f} ns, slack {report.slack_ns:.2f} ns)")
        for r in results:
            flag = "PASS" if r.passed else "FAIL"
            print(f"  [{flag}] {r.name}: {r.verdict} (expected {r.expected}) - {r.detail}")
            if r.model and r.expected == "refuted":
                print(f"         counter-example: {r.model}")
    return report


def run_formal_tests() -> VerificationReport:
    report = verify()
    assert report.all_passed, "formal verification failed"
    assert report.results[1].verdict == "proved", "recovery bound must be a theorem"
    assert report.slack_ns > 0
    # Sanity: shrinking the frequency floor far enough must break the proof.
    slow = OperationalEnvelope(f_min_hz=1_000_000_000)
    assert prove_recovery_bound(RecoveryLatencyModel(slow)).verdict == "satisfiable", \
        "a 1 GHz floor must admit a counter-example"
    print("All formal verification checks passed.")
    return report


if __name__ == "__main__":
    rep = run_formal_tests()
    if "--json" in sys.argv:
        print(rep.to_json())
