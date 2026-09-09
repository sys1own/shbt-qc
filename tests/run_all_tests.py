#!/usr/bin/env python3
"""
run_all_tests.py - unified SHBT-R test runner.

Executes the C reference test suite, multi-physics / photonic CAD self-tests,
Pytest closed-loop HIL bench, Z3 formal verification, and EDA exporter
integrity checks in sequence.

    python3 tests/run_all_tests.py [-k pattern] [--keep-going] [--json out.json]

Exit status is 0 when all V-01..V-35 requirement benchmarks pass and non-zero
otherwise, with a clear failure trace.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import subprocess as _subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simulator.multi_physics import thermal_fea_sp_sim          # noqa: E402
from simulator.cad_yield import spc_photonic_cad                # noqa: E402
from simulator.hil_qec import hil_mps_qec                       # noqa: E402
from formal import formal_verification                          # noqa: E402
from eda.exporters import pdk_hexamer_exporter                   # noqa: E402
from eda.exporters import export_interposer_rf                   # noqa: E402


@dataclass
class SuiteResult:
    name: str
    passed: bool
    seconds: float
    output: str
    error: str = ""


def _compile_c_reference() -> Path:
    """Compile tests/reference_test.c and return the resulting binary path."""
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    if not cc:
        raise RuntimeError("no C compiler found")
    src = REPO_ROOT / "tests" / "reference_test.c"
    out = REPO_ROOT / "tests" / "reference_test"
    if out.exists():
        out.unlink()
    cmd = [
        cc,
        "-std=c11",
        "-O2",
        "-mavx512f",
        "-DSHBT_TSC_HZ=2400000000ULL",
        f"-I{REPO_ROOT / 'kernel' / 'include'}",
        str(src),
        "-o",
        str(out),
        "-lm",
    ]
    r = _subprocess.run(cmd, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"C reference test compile failed:\n{r.stdout}")
    return out


def _run_c_reference() -> None:
    """Build and run the C reference test suite."""
    print("Building C reference test suite ...")
    exe = _compile_c_reference()
    r = _subprocess.run([str(exe)], stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"C reference test failed:\n{r.stdout}")
    print(r.stdout)


def _run_pytest_bench() -> None:
    """Run the closed-loop HIL pytest suite."""
    test_path = REPO_ROOT / "simulator" / "hil_qec" / "test_closed_loop_quench.py"
    pytest = shutil.which("pytest") or shutil.which("py.test")
    if not pytest:
        raise RuntimeError("pytest not found")
    r = _subprocess.run(
        [pytest, str(test_path), "-v"],
        stdout=_subprocess.PIPE,
        stderr=_subprocess.STDOUT,
        text=True,
    )
    print(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"pytest closed-loop bench failed (exit {r.returncode})")


def _run_eda_integrity() -> None:
    """Verify GDSII and Touchstone exporters produce valid artefacts."""
    tmp = Path(tempfile.mkdtemp(prefix="shbt_eda_", dir=REPO_ROOT / "build" if (REPO_ROOT / "build").exists() else None))
    try:
        gds_path = tmp / "hexamer.gds"
        s2p_path = tmp / "interposer.s2p"
        pdk_hexamer_exporter.export_gds(gds_path)
        if not gds_path.exists() or gds_path.stat().st_size == 0:
            raise RuntimeError("GDSII export produced no output")
        export_interposer_rf.generate_touchstone_s2p(s2p_path)
        if not s2p_path.exists() or s2p_path.stat().st_size == 0:
            raise RuntimeError("Touchstone export produced no output")
        s2p_text = s2p_path.read_text()
        if "# Hz S RI R 50" not in s2p_text:
            raise RuntimeError("Touchstone header missing expected reference impedance")
        print(f"  GDSII: {gds_path.stat().st_size} bytes")
        print(f"  S2P: {s2p_path.stat().st_size} bytes")
        print("  EDA exporters produced valid artefacts")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


SUITES: List[tuple] = [
    ("C reference test suite (tests/reference_test.c)", _run_c_reference),
    ("multi_physics.thermal_fea_sp_sim", thermal_fea_sp_sim.run_multiphysics_tests),
    ("cad_yield.spc_photonic_cad", spc_photonic_cad.run_cad_engine_tests),
    ("hil_qec.hil_mps_qec", hil_mps_qec.run_hil_tests),
    ("Pytest closed-loop bench", _run_pytest_bench),
    ("formal.formal_verification", formal_verification.run_formal_tests),
    ("EDA exporter integrity checks", _run_eda_integrity),
]


def run_suite(name: str, fn: Callable[[], object], echo: bool) -> SuiteResult:
    buf = io.StringIO()
    t0 = time.perf_counter()
    err = ""
    ok = True
    sink = contextlib.redirect_stdout(buf)
    try:
        with sink:
            fn()
    except BaseException as exc:            # noqa: BLE001 - report every failure kind
        ok = False
        err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    dt = time.perf_counter() - t0
    out = buf.getvalue()
    if echo:
        sys.stdout.write(out)
    return SuiteResult(name, ok, dt, out, err)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", dest="pattern", default="", help="only run suites whose name contains PATTERN")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failing suite")
    ap.add_argument("--quiet", action="store_true", help="suppress suite stdout (still shown on failure)")
    ap.add_argument("--json", dest="json_out", default="", help="write machine-readable results here")
    args = ap.parse_args(argv)

    selected = [(n, f) for n, f in SUITES if args.pattern in n]
    if not selected:
        print(f"no suites match {args.pattern!r}")
        return 2

    print(f"SHBT-R unified test runner - {len(selected)} suite(s)")
    results: List[SuiteResult] = []
    for name, fn in selected:
        print(f"\n=== {name} ===")
        r = run_suite(name, fn, echo=not args.quiet)
        results.append(r)
        print(f"--- {'PASS' if r.passed else 'FAIL'} {name} ({r.seconds:.2f}s)")
        if not r.passed:
            if args.quiet:
                sys.stdout.write(r.output)
            sys.stdout.write(r.error)
            if not args.keep_going:
                break

    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass
    skipped = len(selected) - len(results)
    print("\n" + "=" * 72)
    for r in results:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name:<40} {r.seconds:7.2f}s")
    print(f"  {n_pass} passed, {n_fail} failed, {skipped} not run, "
          f"total {sum(r.seconds for r in results):.2f}s")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps([asdict(r) for r in results], indent=2))
    return 0 if n_fail == 0 and skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
