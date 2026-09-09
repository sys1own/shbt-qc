"""ctypes/FFI bridge to the SHBT-R freestanding microkernel shared library.

The bridge loads ``shbt_reference.so`` (built from ``kernel/src/shbt_core_runtime.c``)
and exposes the four public symbols:

* ``shbt_ecc_encode``  -> uint8
* ``shbt_ecc_decode``  -> SECDEDResult
* ``shbt_recover``     -> int32
* ``shbt_remap``       -> void (AVX-512 Givens rotation)

A user-space MMIO pointer relocation function ``shbt_set_mmio`` is also exported
by the HIL build of the shared library so that ``shbt_recover`` can be exercised
from host Python without mapping a fixed physical address.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Ctypes layout helpers
# -----------------------------------------------------------------------------
class _ShbtRegisters(ctypes.Structure):
    """ctypes mirror of the volatile ShbtRegisters struct in shbt_hardware.h."""

    _pack_ = 4
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("blank", ctypes.c_uint32),
        ("fifo_data", ctypes.c_uint32),
        ("pll_ctrl", ctypes.c_uint32),
        ("ecc_low", ctypes.c_uint32),
        ("ecc_high", ctypes.c_uint32),
        ("ecc_check", ctypes.c_uint32),
        ("ecc_commit", ctypes.c_uint32),
        ("fault_latch", ctypes.c_uint32),
        ("channel_select", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("phase_offset", ctypes.c_uint32),
        ("ecc_counts", ctypes.c_uint32),
        ("control", ctypes.c_uint32),
    ]


class _SECDEDResult(ctypes.Structure):
    """ctypes mirror of secded_result_t from shbt_core_runtime.c."""

    _fields_ = [
        ("corrected_data", ctypes.c_uint64),
        ("corrected_check", ctypes.c_uint8),
        ("single_bit_error", ctypes.c_uint8),
        ("double_bit_error", ctypes.c_uint8),
    ]


ShbtRegisters = _ShbtRegisters
SECDEDResult = _SECDEDResult


def _find_or_build_so() -> str:
    """Locate shbt_reference.so and rebuild a user-space copy if missing."""
    candidates = [
        _repo_root() / "shbt_reference.so",
        _repo_root() / "build" / "shbt_reference.so",
        Path.cwd() / "shbt_reference.so",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return _build_hil_so()


def _build_hil_so() -> str:
    """Compile the user-space relocatable .so used by this bridge."""
    root = _repo_root()
    out = root / "shbt_reference.so"
    tsc_hz = int(float(os.environ.get("SHBT_TSC_HZ", "2400000000")))
    cmd = [
        "gcc",
        "-O3",
        "-std=c11",
        "-ffreestanding",
        "-nostdlib",
        "-fPIC",
        "-shared",
        "-mavx512f",
        f"-DSHBT_TSC_HZ={tsc_hz}ULL",
        f"-I{root / 'kernel' / 'include'}",
        str(root / "kernel" / "src" / "shbt_user_mmio.c"),
        "-o",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return str(out)


class SHBTKernelBridge:
    """ctypes wrapper for the SHBT-R reference microkernel shared library."""

    def __init__(self, so_path: Optional[Union[str, Path]] = None) -> None:
        if so_path is None:
            so_path = _find_or_build_so()
        self._so_path = str(so_path)
        self._lib = ctypes.CDLL(self._so_path)
        self._bind_symbols()
        self._regs: Optional[ShbtRegisters] = None

    # -------------------------------------------------------------------------
    def _bind_symbols(self) -> None:
        lib = self._lib

        # uint8_t shbt_ecc_encode(uint64_t data)
        lib.shbt_ecc_encode.argtypes = [ctypes.c_uint64]
        lib.shbt_ecc_encode.restype = ctypes.c_uint8

        # secded_result_t shbt_ecc_decode(uint64_t data, uint8_t check_code)
        lib.shbt_ecc_decode.argtypes = [ctypes.c_uint64, ctypes.c_uint8]
        lib.shbt_ecc_decode.restype = SECDEDResult

        # int32_t shbt_recover(void)
        lib.shbt_recover.argtypes = []
        lib.shbt_recover.restype = ctypes.c_int32

        # void shbt_remap(double *col_a, double *col_b, double c, double s, size_t n)
        lib.shbt_remap.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_size_t,
        ]
        lib.shbt_remap.restype = None

        # Optional HIL relocation helpers (present in the user-space build)
        if hasattr(lib, "shbt_set_mmio"):
            lib.shbt_set_mmio.argtypes = [ctypes.POINTER(ShbtRegisters)]
            lib.shbt_set_mmio.restype = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    @property
    def so_path(self) -> str:
        return self._so_path

    def set_mmio(self, regs: ShbtRegisters) -> None:
        """Point the C microkernel's SHBT_MMIO pointer at ``regs``.

        This must be called before ``shbt_recover`` is exercised from
        user-space HIL simulation.
        """
        if hasattr(self._lib, "shbt_set_mmio"):
            self._regs = regs
            self._lib.shbt_set_mmio(ctypes.byref(regs))
        else:
            raise RuntimeError(
                "Loaded shbt_reference.so does not export shbt_set_mmio; "
                "rebuild with kernel/src/shbt_user_mmio.c"
            )

    @property
    def mmio(self) -> Optional[ShbtRegisters]:
        """The last ShbtRegisters instance passed to ``set_mmio``."""
        return self._regs

    def ecc_encode(self, data: int) -> int:
        return int(self._lib.shbt_ecc_encode(ctypes.c_uint64(data)))

    def ecc_decode(self, data: int, check_code: int) -> SECDEDResult:
        return self._lib.shbt_ecc_decode(ctypes.c_uint64(data), ctypes.c_uint8(check_code))

    def recover(self) -> int:
        """Run the microkernel post-quench recovery routine.

        Returns 0 on success, -1 on uncorrectable double-bit error,
        -2 if the 120 ns timing budget is exceeded.
        """
        return int(self._lib.shbt_recover())

    def remap(self, col_a: Any, col_b: Any, c: float, s: float) -> None:
        """Apply an AVX-512 Givens rotation to two equal-length double columns.

        ``col_a`` and ``col_b`` may be Python sequences, ``numpy.ndarray`` of
        ``float64``, or ``ctypes`` double buffers.  The inputs are modified in
        place.
        """
        n = len(col_a)
        if n != len(col_b):
            raise ValueError("col_a and col_b must have the same length")

        if isinstance(col_a, np.ndarray) and isinstance(col_b, np.ndarray):
            if col_a.dtype != np.float64 or col_b.dtype != np.float64:
                raise TypeError("numpy inputs must be float64")
            pa = ctypes.cast(col_a.ctypes.data, ctypes.POINTER(ctypes.c_double))
            pb = ctypes.cast(col_b.ctypes.data, ctypes.POINTER(ctypes.c_double))
        else:
            a_buf = (ctypes.c_double * n)(*col_a)
            b_buf = (ctypes.c_double * n)(*col_b)
            pa = a_buf
            pb = b_buf
            # copy back after the C call
            self._lib.shbt_remap(pa, pb, ctypes.c_double(c), ctypes.c_double(s), ctypes.c_size_t(n))
            for i in range(n):
                col_a[i] = pa[i]
                col_b[i] = pb[i]
            return

        self._lib.shbt_remap(pa, pb, ctypes.c_double(c), ctypes.c_double(s), ctypes.c_size_t(n))


# -----------------------------------------------------------------------------
# Smoke test when run directly
# -----------------------------------------------------------------------------
def _smoke() -> None:
    bridge = SHBTKernelBridge()

    data = 0xC0FFEE1234567890
    check = bridge.ecc_encode(data)
    dec = bridge.ecc_decode(data, check)
    assert dec.single_bit_error == 0 and dec.double_bit_error == 0
    assert dec.corrected_data == data

    for bit in (0, 7, 31, 63):
        bad = data ^ (1 << bit)
        dec = bridge.ecc_decode(bad, check)
        assert dec.single_bit_error == 1 and dec.double_bit_error == 0
        assert dec.corrected_data == data

    bad = data ^ (1 << 3) ^ (1 << 41)
    dec = bridge.ecc_decode(bad, check)
    assert dec.double_bit_error == 1

    regs = ShbtRegisters()
    regs.status = 1 << 1  # PLL_LOCK
    bridge.set_mmio(regs)
    assert bridge.recover() == 0
    assert regs.blank == 0 and regs.fault_latch == 0

    a = np.array([1.0, 0.0, 3.0, -2.0], dtype=np.float64)
    b = np.array([0.0, 1.0, 1.0, 4.0], dtype=np.float64)
    bridge.remap(a, b, c=0.6, s=0.8)
    assert np.allclose(a, [0.6, 0.8, 2.6, 2.0])
    assert np.allclose(b, [-0.8, 0.6, -1.8, 4.0])

    print("kernel_bridge smoke test passed")


if __name__ == "__main__":
    _smoke()
