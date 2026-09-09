"""Closed-loop Hardware-in-the-Loop (HIL) virtual testbench for SHBT-R QEC.

This module provides:

* ``SHBTMMIOBus`` -- a dictionary-backed virtual MMIO bus that mirrors the
  SHBT-MMIO-1 register block at ``0x70000000``.
* ``Secded7264`` / ``SECDEDOutcome`` -- a pure-Python SECDED Hamming(72,64)
  implementation matching the microkernel semantics.
* ``MatrixProductStateDecoder`` -- MPS tensor-network primitives with
  left-to-right transfer-matrix contraction and SVD truncation to a maximum
  bond dimension ``chi``.
* ``MPSDecoderEngine`` -- 312-channel syndrome/stabiliser decoder that maps
  MPS defects to channel indices and drives the ECC decoder.
* ``HILClosedLoop`` -- connects thermal-quench transients from the SHBT-R
  multi-physics simulator to MMIO interrupts and exercises the microkernel
  recovery routine through the ctypes bridge.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.hil_qec.kernel_bridge import SHBTKernelBridge, ShbtRegisters


# -----------------------------------------------------------------------------
# SHBT-MMIO-1 register aperture (kernel/include/shbt_hardware.h)
# -----------------------------------------------------------------------------
MMIO_BASE = 0x70000000
MMIO_SIZE = 0x1000
NUM_CHANNELS = 312

SHBT_STATUS_OVERTEMP = 1 << 0
SHBT_STATUS_PLL_LOCK = 1 << 1
SHBT_STATUS_ECC_ERR = 1 << 2
SHBT_STATUS_FAULT_ST = 1 << 3

_REG_FIELDS: Dict[int, str] = {
    0x00: "status",
    0x04: "blank",
    0x08: "fifo_data",
    0x0C: "pll_ctrl",
    0x10: "ecc_low",
    0x14: "ecc_high",
    0x18: "ecc_check",
    0x1C: "ecc_commit",
    0x20: "fault_latch",
    0x24: "channel_select",
    0x28: "abi_version",
    0x2C: "phase_offset",
    0x30: "ecc_counts",
    0x34: "control",
}

_U32 = 0xFFFFFFFF


@dataclass
class BusTransaction:
    kind: str          # "R" or "W"
    address: int
    value: int
    timestamp_ns: int


@dataclass
class SECDEDOutcome:
    """Python-side SECDED decode result (extends the C SECDEDResult with metadata)."""

    corrected_data: int
    corrected_check: int
    single_bit_error: int
    double_bit_error: int
    syndrome: int = 0
    status: str = "ok"


# -----------------------------------------------------------------------------
# SECDED Hamming(72,64)
# -----------------------------------------------------------------------------

class Secded7264:
    """SECDED Hamming(72,64): 7 positional check bits + overall parity."""

    def __init__(self) -> None:
        self.masks = [0] * 7
        self.data_of_pos = [-1] * 128
        d = 0
        for p in range(1, 128):
            if p & (p - 1) == 0:
                continue
            if d >= 64:
                break
            self.data_of_pos[p] = d
            for j in range(7):
                if p & (1 << j):
                    self.masks[j] |= 1 << d
            d += 1

    @staticmethod
    def _parity(x: int) -> int:
        return bin(x).count("1") & 1

    def hamming_bits(self, data: int) -> int:
        c = 0
        for j in range(7):
            c |= self._parity(data & self.masks[j]) << j
        return c

    def encode(self, data: int) -> int:
        c = self.hamming_bits(data)
        overall = self._parity(data) ^ self._parity(c)
        return c | (overall << 7)

    def decode(self, data: int, check: int) -> SECDEDOutcome:
        """Return an SECDEDOutcome matching the C microkernel semantics."""
        syn = (check ^ self.hamming_bits(data)) & 0x7F
        parity = self._parity(data) ^ self._parity(check)
        syndrome_reg = syn | (parity << 7)

        if syn == 0 and parity == 0:
            return SECDEDOutcome(data, check, 0, 0, syndrome_reg, "ok")
        if syn == 0 and parity == 1:
            return SECDEDOutcome(data, check ^ 0x80, 1, 0, syndrome_reg, "corrected_parity")
        if parity == 0:
            return SECDEDOutcome(data, check, 0, 1, syndrome_reg, "uncorrectable")
        if syn & (syn - 1) == 0:
            j = syn.bit_length() - 1
            return SECDEDOutcome(data, check ^ (1 << j), 1, 0, syndrome_reg, "corrected_check")
        db = self.data_of_pos[syn]
        if db < 0:
            return SECDEDOutcome(data, check, 0, 1, syndrome_reg, "uncorrectable")
        return SECDEDOutcome(data ^ (1 << db), check, 1, 0, syndrome_reg, "corrected_data")


# -----------------------------------------------------------------------------
# Strict SHBT-MMIO-1 virtual bus
# -----------------------------------------------------------------------------
class SHBTMMIOBus:
    """Dictionary-backed virtual MMIO bus conforming to SHBT-MMIO-1.

    The register layout is a direct mirror of ``kernel/include/shbt_hardware.h``
    with base address ``0x70000000`` and 4-byte aligned 32-bit accesses.
    """

    def __init__(self, base: int = MMIO_BASE, size: int = MMIO_SIZE) -> None:
        self.base = base
        self.size = size
        self.regs = ShbtRegisters()
        self.regs.abi_version = 1
        self.time_ns = 0
        self.log: List[BusTransaction] = []
        self.reset()

    def reset(self) -> None:
        for off, name in _REG_FIELDS.items():
            if name == "abi_version":
                continue
            setattr(self.regs, name, 0)
        self.regs.abi_version = 1
        self.log.clear()

    def _offset(self, address: int) -> int:
        if address % 4:
            raise ValueError(f"Unaligned MMIO access at 0x{address:08X}")
        off = address - self.base
        if not (0 <= off < self.size):
            raise ValueError(f"MMIO address 0x{address:08X} outside aperture")
        return off

    def read32(self, address: int) -> int:
        off = self._offset(address)
        name = _REG_FIELDS.get(off)
        if name is None:
            value = 0
        else:
            value = int(getattr(self.regs, name)) & _U32
        self.log.append(BusTransaction("R", address, value, self.time_ns))
        return value

    def write32(self, address: int, value: int) -> None:
        off = self._offset(address)
        value &= _U32
        self.log.append(BusTransaction("W", address, value, self.time_ns))
        name = _REG_FIELDS.get(off)
        if name is None:
            return
        if name == "abi_version":
            return
        setattr(self.regs, name, value)

    def set_status_bits(self, mask: int) -> None:
        self.regs.status = (self.regs.status | mask) & _U32

    def clear_status_bits(self, mask: int) -> None:
        self.regs.status = (self.regs.status & ~mask) & _U32

    @property
    def ecc_payload(self) -> int:
        return (self.regs.ecc_high << 32) | self.regs.ecc_low

    def set_ecc_payload(self, data: int) -> None:
        self.regs.ecc_low = data & _U32
        self.regs.ecc_high = (data >> 32) & _U32


# -----------------------------------------------------------------------------
# Matrix Product State tensor network
# -----------------------------------------------------------------------------
PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class MatrixProductStateDecoder:
    """Open-boundary MPS with tensors A[i] of shape (chi_l, 2, chi_r), chi <= 256.

    2-D lattices of shape (Lx, Ly) are snake-ordered onto the 1-D chain so that
    horizontal neighbours remain adjacent in the MPS and vertical neighbours are
    reached through ``neighbour_pairs_2d``.
    """

    def __init__(
        self,
        n_sites: Optional[int] = None,
        chi: int = 256,
        lattice: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = 0,
    ) -> None:
        if lattice is not None:
            self.lattice = (int(lattice[0]), int(lattice[1]))
            n_sites = self.lattice[0] * self.lattice[1]
        else:
            self.lattice = None
        if n_sites is None or n_sites < 2:
            raise ValueError("n_sites must be >= 2")
        self.n = int(n_sites)
        self.chi = int(chi)
        self.rng = np.random.default_rng(seed)
        self.tensors: List[np.ndarray] = []
        self.error_log: List[Tuple[str, int]] = []
        self.set_product_state(0)

    def set_product_state(self, basis: int = 0) -> None:
        self.tensors = []
        for _ in range(self.n):
            A = np.zeros((1, 2, 1), dtype=complex)
            A[0, basis, 0] = 1.0
            self.tensors.append(A)
        self.error_log = []

    def set_ghz_state(self) -> None:
        """|00..0> + |11..1> (bond dimension 2)."""
        self.tensors = []
        for i in range(self.n):
            if i == 0:
                A = np.zeros((1, 2, 2), dtype=complex)
                A[0, 0, 0] = A[0, 1, 1] = 1.0 / math.sqrt(2)
            elif i == self.n - 1:
                A = np.zeros((2, 2, 1), dtype=complex)
                A[0, 0, 0] = A[1, 1, 0] = 1.0
            else:
                A = np.zeros((2, 2, 2), dtype=complex)
                A[0, 0, 0] = A[1, 1, 1] = 1.0
            self.tensors.append(A)
        self.error_log = []

    def site_index(self, x: int, y: int) -> int:
        if self.lattice is None:
            raise ValueError("Not a 2-D lattice")
        Lx, Ly = self.lattice
        col = x if y % 2 == 0 else Lx - 1 - x
        return y * Lx + col

    def neighbour_pairs_2d(self) -> List[Tuple[int, int]]:
        Lx, Ly = self.lattice
        pairs = []
        for y in range(Ly):
            for x in range(Lx):
                i = self.site_index(x, y)
                if x + 1 < Lx:
                    pairs.append((i, self.site_index(x + 1, y)))
                if y + 1 < Ly:
                    pairs.append((i, self.site_index(x, y + 1)))
        return pairs

    def apply_single_site(self, site: int, op: np.ndarray) -> None:
        self.tensors[site] = np.einsum("ab,ibj->iaj", op, self.tensors[site])

    def inject_x_error(self, site: int) -> None:
        self.apply_single_site(site, PAULI_X)
        self.error_log.append(("X", site))

    def inject_z_error(self, site: int) -> None:
        self.apply_single_site(site, PAULI_Z)
        self.error_log.append(("Z", site))

    def inject_random_errors(self, p_x: float, p_z: float) -> List[Tuple[str, int]]:
        injected = []
        for i in range(self.n):
            if self.rng.random() < p_x:
                self.inject_x_error(i)
                injected.append(("X", i))
            if self.rng.random() < p_z:
                self.inject_z_error(i)
                injected.append(("Z", i))
        return injected

    @staticmethod
    def _transfer(E: np.ndarray, A: np.ndarray, B: np.ndarray, O: Optional[np.ndarray] = None) -> np.ndarray:
        """E_{cd} = sum_{a,b,i,j} E_{ab} conj(A)_{aic} O_{ij} B_{bjd}."""
        if O is not None:
            B = np.einsum("ij,bjd->bid", O, B)
        T = np.tensordot(E, B, axes=([1], [0]))
        return np.tensordot(A.conj(), T, axes=([0, 1], [0, 1]))

    def norm(self) -> float:
        E = np.ones((1, 1), dtype=complex)
        for A in self.tensors:
            E = self._transfer(E, A, A)
        return float(np.real(E[0, 0]))

    def normalize(self) -> None:
        nrm = math.sqrt(self.norm())
        self.tensors[0] = self.tensors[0] / nrm

    def expectation_string(self, ops: Dict[int, np.ndarray]) -> float:
        E = np.ones((1, 1), dtype=complex)
        for i, A in enumerate(self.tensors):
            E = self._transfer(E, A, A, ops.get(i))
        return float(np.real(E[0, 0])) / self.norm()

    def zz_parity(self, i: int, j: int) -> float:
        return self.expectation_string({i: PAULI_Z, j: PAULI_Z})

    def syndrome_parities(self) -> np.ndarray:
        """Nearest-neighbour Z_i Z_{i+1} parities along the chain (length n-1)."""
        return np.array([self.zz_parity(i, i + 1) for i in range(self.n - 1)])

    def syndrome_bits(self) -> np.ndarray:
        """1 where the ZZ parity is violated (<ZZ> < 0)."""
        return (self.syndrome_parities() < 0.0).astype(int)

    def decode_x_errors_from_syndrome(self) -> List[int]:
        """Minimum-weight decode of X errors on a chain from ZZ defects."""
        bits = self.syndrome_bits()
        regions_a: List[int] = []
        flipped = False
        for i in range(self.n):
            if flipped:
                regions_a.append(i)
            if i < self.n - 1 and bits[i]:
                flipped = not flipped
        regions_b = [i for i in range(self.n) if i not in set(regions_a)]
        return regions_a if len(regions_a) <= len(regions_b) else regions_b

    def truncate(self, chi: Optional[int] = None) -> float:
        """Left-to-right SVD sweep truncating bonds to chi; returns discarded weight."""
        chi = self.chi if chi is None else int(chi)
        discarded = 0.0
        for i in range(self.n - 1):
            A = self.tensors[i]
            dl, d, dr = A.shape
            M = A.reshape(dl * d, dr)
            U, S, Vh = np.linalg.svd(M, full_matrices=False)
            keep = min(chi, S.size)
            discarded += float(np.sum(S[keep:] ** 2))
            U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
            self.tensors[i] = U.reshape(dl, d, keep)
            self.tensors[i + 1] = np.einsum("ab,bic->aic", np.diag(S) @ Vh, self.tensors[i + 1])
        return discarded

    def apply_two_site(self, i: int, gate: np.ndarray) -> None:
        """Apply a 4x4 gate on sites (i, i+1) and re-split with chi truncation."""
        A, B = self.tensors[i], self.tensors[i + 1]
        theta = np.einsum("aib,bjc->aijc", A, B)
        dl, _, _, dr = theta.shape
        theta = np.einsum("klij,aijc->aklc", gate.reshape(2, 2, 2, 2), theta)
        M = theta.reshape(dl * 2, 2 * dr)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        keep = min(self.chi, int(np.sum(S > 1e-12)))
        keep = max(keep, 1)
        self.tensors[i] = U[:, :keep].reshape(dl, 2, keep)
        self.tensors[i + 1] = (np.diag(S[:keep]) @ Vh[:keep, :]).reshape(keep, 2, dr)


# -----------------------------------------------------------------------------
# 312-channel MPS QEC decoder engine
# -----------------------------------------------------------------------------
class MPSDecoderEngine:
    """MPS-based QEC decoder for the SHBT-R 312-channel control plane.

    The engine exposes SECDED encode/decode, stabiliser extraction via MPS
    tensor contraction (bond dimension ``chi`` up to 256), and a minimum-weight
    syndrome-chain decoder.  It can be bound to an ``SHBTMMIOBus`` and
    ``SHBTKernelBridge`` for HIL recovery.
    """

    def __init__(
        self,
        n_channels: int = NUM_CHANNELS,
        chi: int = 256,
        lattice: Optional[Tuple[int, int]] = None,
        seed: int = 0,
        bus: Optional[SHBTMMIOBus] = None,
        bridge: Optional[SHBTKernelBridge] = None,
    ) -> None:
        self.n_channels = int(n_channels)
        self.chi = int(chi)
        if lattice is not None and (lattice[0] * lattice[1] != n_channels):
            raise ValueError("lattice dimensions must multiply to n_channels")
        self.mps = MatrixProductStateDecoder(
            n_sites=n_channels if lattice is None else None,
            chi=chi,
            lattice=lattice,
            seed=seed,
        )
        self.ecc = Secded7264()
        self.bus = bus
        self.bridge = bridge
        self.last_ecc_result: Optional[SECDEDOutcome] = None

    def encode_ecc(self, data: int) -> int:
        return self.ecc.encode(data)

    def decode_ecc(self, data: int, check: int) -> SECDEDOutcome:
        res = self.ecc.decode(data, check)
        self.last_ecc_result = res
        return res

    def stabilizer_syndromes(self) -> np.ndarray:
        """ZZ stabiliser parities along the 1-D MPS chain."""
        return self.mps.syndrome_bits()

    def decode_syndrome_chain(self) -> List[int]:
        """Return a minimum-weight estimate of X-error locations from ZZ defects."""
        return self.mps.decode_x_errors_from_syndrome()

    def inject_x_error(self, channel: int) -> None:
        """Inject a bit-flip (X) error on the MPS site corresponding to ``channel``."""
        if not (0 <= channel < self.n_channels):
            raise ValueError(f"channel must be in [0, {self.n_channels})")
        self.mps.inject_x_error(channel)

    def truncate(self, chi: Optional[int] = None) -> float:
        """Truncate all MPS bonds to ``chi`` and return the discarded weight."""
        return self.mps.truncate(chi)

    def connect_thermal_quench(self, thermal_solver: Any, threshold_k: float = 5.0) -> bool:
        """Raise an MMIO OVERTEMP/FAULT interrupt if the solver max T exceeds ``threshold_k``.

        ``thermal_solver`` is expected to expose a 3-D ``T`` array in Kelvin,
        e.g. the ``CryogenicThermalSolver`` from ``thermal_fea_sp_sim``.
        """
        if self.bus is None:
            raise RuntimeError("No MMIO bus bound to decoder")
        t_max = float(np.max(getattr(thermal_solver, "T", np.array([]))))
        if t_max > threshold_k:
            self.bus.set_status_bits(SHBT_STATUS_OVERTEMP | SHBT_STATUS_FAULT_ST)
            return True
        return False


# -----------------------------------------------------------------------------
# Closed-loop HIL harness
# -----------------------------------------------------------------------------
class HILClosedLoop:
    """Binds the strict MMIO bus, MPS decoder, and ctypes microkernel bridge."""

    def __init__(
        self,
        bus: Optional[SHBTMMIOBus] = None,
        decoder: Optional[MPSDecoderEngine] = None,
        bridge: Optional[SHBTKernelBridge] = None,
    ) -> None:
        self.bus = bus or SHBTMMIOBus()
        self.bridge = bridge or SHBTKernelBridge()
        self.decoder = decoder or MPSDecoderEngine(bus=self.bus, bridge=self.bridge)
        self.bridge.set_mmio(self.bus.regs)
        # Warm the C recover path once so the first measured call is in cache.
        self.set_pll_locked(True)
        try:
            self.bridge.recover()
        except Exception:
            pass
        self.bus.reset()

    # -------------------------------------------------------------------------
    def set_pll_locked(self, locked: bool = True) -> None:
        if locked:
            self.bus.set_status_bits(SHBT_STATUS_PLL_LOCK)
        else:
            self.bus.clear_status_bits(SHBT_STATUS_PLL_LOCK)

    def trigger_quench_interrupt(self, channel: Optional[int] = None, thermal_solver: Optional[Any] = None) -> None:
        """Raise an OVERTEMP/FAULT_ST MMIO interrupt.

        If ``thermal_solver`` is supplied the interrupt is raised only when the
        solver's peak temperature exceeds the 4.2 K bath by at least 0.8 K.
        """
        if thermal_solver is not None:
            self.decoder.connect_thermal_quench(thermal_solver, threshold_k=5.0)
        else:
            self.bus.set_status_bits(SHBT_STATUS_OVERTEMP | SHBT_STATUS_FAULT_ST)
        if channel is not None:
            self.bus.regs.channel_select = int(channel) & _U32

    def inject_double_bit_ecc(self, data: int, bit_a: int, bit_b: int) -> None:
        """Latch a double-bit corrupted codeword into the ECC registers."""
        check = self.bridge.ecc_encode(data)
        bad = data ^ (1 << bit_a) ^ (1 << bit_b)
        self.bus.set_ecc_payload(bad)
        self.bus.regs.ecc_check = check & 0xFF

    def run_hardware_recovery(self) -> int:
        """Invoke the C microkernel recovery routine.

        Returns the same integer code as ``shbt_recover``:
          0  success within the 120 ns budget,
         -1  uncorrectable double-bit ECC error (blanking asserted),
         -2  timing budget exceeded.
        """
        # For HIL simulation the PLL lock bit must be present; real hardware
        # would set this after the VCO settles.
        self.set_pll_locked(True)
        return self.bridge.recover()


# -----------------------------------------------------------------------------
# Self-test entry point
# -----------------------------------------------------------------------------
def run_hil_tests() -> None:
    print("Running SHBT-R HIL / MPS-QEC self-tests...")

    bus = SHBTMMIOBus()
    assert bus.base == MMIO_BASE
    assert bus.regs.abi_version == 1

    bus.write32(MMIO_BASE + 0x04, 1)
    assert bus.read32(MMIO_BASE + 0x04) == 1, "blank readback"
    try:
        bus.read32(MMIO_BASE + 1)
        raise AssertionError("unaligned access must fail")
    except ValueError:
        pass
    try:
        bus.read32(MMIO_BASE - 4)
        raise AssertionError("out-of-aperture access must fail")
    except ValueError:
        pass
    print("  SHBTMMIOBus: alignment, aperture, and ABI version OK")

    ecc = Secded7264()
    rng = np.random.default_rng(1)
    for _ in range(500):
        d = int(rng.integers(0, 2**63)) | (int(rng.integers(0, 2)) << 63)
        c = ecc.encode(d)
        assert ecc.decode(d, c).status == "ok"
        b = int(rng.integers(0, 64))
        r = ecc.decode(d ^ (1 << b), c)
        assert r.status == "corrected_data" and r.corrected_data == d and r.corrected_check == c
        b2 = (b + 1 + int(rng.integers(0, 63))) % 64
        assert ecc.decode(d ^ (1 << b) ^ (1 << b2), c).double_bit_error
    print("  SECDED(72,64): single-bit correction and double-bit detection OK")

    mps = MatrixProductStateDecoder(n_sites=12, chi=256)
    assert mps.chi == 256
    assert abs(mps.norm() - 1.0) < 1e-12
    assert np.allclose(mps.syndrome_parities(), 1.0)
    mps.inject_x_error(4)
    par = mps.syndrome_parities()
    assert par[3] < 0 and par[4] < 0 and np.sum(par < 0) == 2
    assert mps.decode_x_errors_from_syndrome() == [4]
    mps.inject_z_error(2)
    assert np.sum(mps.syndrome_parities() < 0) == 2
    print("  MPS decoder: syndrome extraction and chain decoding OK")

    engine = MPSDecoderEngine(n_channels=312, chi=256)
    assert engine.n_channels == 312
    engine.inject_x_error(7)
    engine.inject_x_error(8)
    syns = engine.stabilizer_syndromes()
    assert syns[6] and syns[8]
    assert len(engine.decode_syndrome_chain()) == 2
    print("  MPSDecoderEngine: 312-channel syndrome chain OK")

    loop = HILClosedLoop(bus=bus)
    loop.trigger_quench_interrupt(channel=5)
    assert bus.regs.status & SHBT_STATUS_OVERTEMP
    assert bus.regs.status & SHBT_STATUS_FAULT_ST
    rc = loop.run_hardware_recovery()
    assert rc == 0
    assert bus.regs.blank == 0
    assert bus.regs.fault_latch == 0
    print("  HILClosedLoop: quench interrupt + hardware recovery OK")

    loop.inject_double_bit_ecc(0xC0FFEE1234567890, 3, 41)
    rc = loop.run_hardware_recovery()
    assert rc == -1
    assert bus.regs.blank == 1
    assert bus.regs.control == 0
    dec = loop.decoder.decode_ecc(loop.bus.ecc_payload, bus.regs.ecc_check)
    assert dec.double_bit_error
    print("  HILClosedLoop: double-bit ECC forces RF blanking OK")

    print("All HIL / MPS-QEC self-tests passed.")


if __name__ == "__main__":
    run_hil_tests()
