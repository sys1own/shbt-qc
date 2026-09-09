#!/usr/bin/env python3
"""shbt-tool - unifying CLI orchestrator for the SHBT-R reference stack."""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import struct
import subprocess
import sys
from typing import List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simulator.multi_physics import thermal_fea_sp_sim  # noqa: E402
from simulator.hil_qec import hil_mps_qec  # noqa: E402
from eda.exporters import pdk_hexamer_exporter  # noqa: E402
from eda.exporters import export_interposer_rf  # noqa: E402
from formal import formal_verification  # noqa: E402


def _compiler() -> str:
    for cmd in ("gcc", "clang", "cc"):
        exe = shutil.which(cmd)
        if exe:
            return exe
    return ""


def _elf64_ok(path: pathlib.Path) -> tuple[bool, str]:
    data = path.read_bytes()
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return False, "missing ELF magic"
    ei_class = data[4]
    e_type, e_machine = struct.unpack_from("<HH", data, 16)
    if ei_class != 2:
        return False, f"not ELF64 (EI_CLASS={ei_class})"
    if e_type != 3:  # ET_DYN
        return False, f"e_type={e_type}, expected ET_DYN(3)"
    if e_machine != 62:  # EM_X86_64
        return False, f"e_machine={e_machine}, expected EM_X86_64(62)"
    return True, f"ELF64 ET_DYN EM_X86_64 {len(data)} bytes"


def cmd_build_kernel(args: argparse.Namespace) -> int:
    compiler = _compiler()
    if not compiler:
        print("ERROR: no C compiler found (gcc/clang/cc)", file=sys.stderr)
        return 1

    out = (REPO_ROOT / args.out).resolve()
    if out.is_dir():
        out = out / "shbt_reference.so"
    out.parent.mkdir(parents=True, exist_ok=True)

    tsc_hz = int(float(os.environ.get("SHBT_TSC_HZ", "2400000000")))
    src = REPO_ROOT / "kernel" / "src" / "shbt_user_mmio.c"
    cmd = [
        compiler,
        "-O3",
        "-std=c11",
        "-ffreestanding",
        "-nostdlib",
        "-fPIC",
        "-shared",
        "-mavx512f",
        f"-DSHBT_TSC_HZ={tsc_hz}ULL",
        f"-I{REPO_ROOT / 'kernel' / 'include'}",
        str(src),
        "-o",
        str(out),
    ]
    print("Building SHBT-R reference kernel ...")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    ok, msg = _elf64_ok(out)
    if not ok:
        print(f"ERROR: ELF validation failed: {msg}", file=sys.stderr)
        return 1
    print(f"  ELF OK: {msg}")
    readelf = shutil.which("readelf")
    if readelf:
        subprocess.run([readelf, "-h", str(out)], check=False)
    print(f"Built {out}")
    return 0


def cmd_sim(args: argparse.Namespace) -> int:
    print("=== SHBT-R multi-physics FEA self-test ===")
    thermal_fea_sp_sim.run_multiphysics_tests()
    print("\n=== SHBT-R MPS-QEC / HIL closed-loop self-test ===")
    hil_mps_qec.run_hil_tests()
    return 0


def cmd_export_eda(args: argparse.Namespace) -> int:
    print("=== GDSII hexamer mask export ===")
    pdk_hexamer_exporter.main()
    print("\n=== Touchstone S2P RF interposer export ===")
    export_interposer_rf.main()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    print("=== Z3 formal verification ===")
    report = formal_verification.run_formal_tests()
    print(f"  worst-case recovery: {report.worst_case_recovery_ns:.2f} ns")
    print("\n=== Pytest closed-loop bench ===")
    pytest = shutil.which("pytest") or shutil.which("py.test")
    if not pytest:
        print("ERROR: pytest not found", file=sys.stderr)
        return 1
    test_path = REPO_ROOT / "simulator" / "hil_qec" / "test_closed_loop_quench.py"
    r = subprocess.run([pytest, str(test_path), "-v"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(r.stdout)
    r.check_returncode()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="shbt-tool", description="SHBT-R CLI orchestrator")
    sub = ap.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-kernel", help="compile the freestanding microkernel into shbt_reference.so")
    p_build.add_argument("--out", default="shbt_reference.so", help="output path")
    p_build.set_defaults(func=cmd_build_kernel)

    p_sim = sub.add_parser("sim", help="run multi-physics FEA and MPS-QEC co-simulation")
    p_sim.set_defaults(func=cmd_sim)

    p_export = sub.add_parser("export-eda", help="export GDSII mask and Touchstone S2P")
    p_export.set_defaults(func=cmd_export_eda)

    p_verify = sub.add_parser("verify", help="run Z3 formal verification and pytest testbench")
    p_verify.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
