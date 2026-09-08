#!/usr/bin/env python3
"""
run_all_tests.py - unified SHBT-R test runner.

Imports every simulation / verification module and executes its self-test
entry point, then drives scripts/export_os.py end-to-end into a scratch
directory and validates the produced release archive.

    python3 tests/run_all_tests.py [-k pattern] [--keep-going] [--json out.json]

Exit status is non-zero if any suite fails.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
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
from scripts import export_os                                   # noqa: E402


@dataclass
class SuiteResult:
    name: str
    passed: bool
    seconds: float
    output: str
    error: str = ""


def _run_export_end_to_end() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="shbt_release_", dir=REPO_ROOT / "build" if (REPO_ROOT / "build").exists() else None))
    try:
        res = export_os.export(tmp, skip_formal=False, verbose=True)
        export_os.verify_release(res)
        assert res.zip_path.name == export_os.RELEASE_ZIP_NAME
        assert res.manifest.formal_all_passed, "release must carry a passing formal report"
        assert res.manifest.toolchain in ("clang", "x86_64-elf-gcc", "mock")
        assert res.manifest.elf["class"] == "ELF64" and res.manifest.elf["machine"] == 62
        assert len(res.zip_sha256) == 64
        assert (tmp / "SHA256SUMS").exists() and (tmp / f"{export_os.RELEASE_ZIP_NAME}.sha256").exists()
        # The generated HAL header must at least parse as C when a host compiler is present.
        if shutil.which("gcc") or shutil.which("cc"):
            import subprocess
            cc = shutil.which("gcc") or shutil.which("cc")
            probe = "#include \"shbt_hal.h\"\nint p(void){ return (int)shbt_hal_read_status(); }\n"
            r = subprocess.run([cc, "-std=c11", "-fsyntax-only", "-ffreestanding", "-Wall", "-Wextra", "-Werror",
                                "-I", str(export_os.KERNEL_INC), "-x", "c", "-"],
                               input=probe, capture_output=True, text=True)
            assert r.returncode == 0, f"generated shbt_hal.h failed to compile:\n{r.stderr}"
        print(f"  export_os: {res.manifest.toolchain} toolchain, zip sha256 {res.zip_sha256[:16]}..., "
              f"{len(res.manifest.files)} artefacts")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


SUITES: List[tuple] = [
    ("multi_physics.thermal_fea_sp_sim", thermal_fea_sp_sim.run_multiphysics_tests),
    ("cad_yield.spc_photonic_cad", spc_photonic_cad.run_cad_engine_tests),
    ("hil_qec.hil_mps_qec", hil_mps_qec.run_hil_tests),
    ("formal.formal_verification", formal_verification.run_formal_tests),
    ("scripts.export_os (end-to-end)", _run_export_end_to_end),
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
