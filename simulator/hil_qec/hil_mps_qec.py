"""Closed-loop Hardware-in-the-Loop (HIL) virtual testbench for SHBT-R QEC.

Components
----------
VirtualMMIOBus
    Register-accurate model of the SHBT-R MMIO aperture at 0x70000000.  The
    layout mirrors ``kernel/include/shbt_hardware.h``: CTRL, STATUS, IRQ_*,
    PUMP_ATTEN, ECC_* (SECDED Hamming(72,64)), TIMER, TEMP and 312 PHASE
    registers.  Register side effects (write-1-to-clear, SECDED decode on
    ECC_CTRL scrub, fault latching) are emulated so firmware-style drivers can
    be exercised without silicon.

MatrixProductStateDecoder
    1-D / 2-D (snake-ordered) matrix-product-state tensor network with maximum
    bond dimension chi = 256.  Provides Pauli X/Z error injection, SVD
    truncation, norm/overlap contraction, and transfer-matrix contraction of
    nearest-neighbour Z_i Z_{i+1} syndrome parities.

HILVirtualTestbench
    Drives closed-loop transients (quench, ECC single/double-bit faults,
    phase slips derived from MPS syndromes) into the virtual registers, runs
    the recovery driver, and verifies interrupt/SECDED status updates.

``run_hil_tests()`` asserts that fault injection updates the latched fault
register at 0x70000010 and that SECDED status is reported correctly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Register map (mirrors kernel/include/shbt_hardware.h)
# ---------------------------------------------------------------------------
MMIO_BASE = 0x70000000
MMIO_SIZE = 0x10000
NUM_CHANNELS = 312

REG_CTRL = 0x0000
REG_STATUS = 0x0004
REG_IRQ_STATUS = 0x0008
REG_IRQ_ENABLE = 0x000C
REG_IRQ_CLEAR = 0x0010      # W1C; reads back the latched fault flags
REG_PUMP_ATTEN = 0x0014
REG_ECC_CTRL = 0x0018
REG_ECC_STATUS = 0x001C
REG_ECC_SYNDROME = 0x0020
REG_ECC_ADDR = 0x0024
REG_ECC_DATA_LO = 0x0028
REG_ECC_DATA_HI = 0x002C
REG_ECC_CHECK = 0x0030
REG_TIMER_LO = 0x0034
REG_TIMER_HI = 0x0038
REG_TEMP_MK = 0x003C
REG_PHASE_BASE = 0x1000

CTRL_ENABLE = 1 << 0
CTRL_SOFT_RESET = 1 << 1
CTRL_PUMP_ENABLE = 1 << 2
CTRL_PHASE_LATCH = 1 << 3
CTRL_PHASE_RESET = 1 << 4
CTRL_ECC_ENABLE = 1 << 5
CTRL_QUENCH_ACK = 1 << 6

STATUS_READY = 1 << 0
STATUS_BUSY = 1 << 1
STATUS_QUENCH = 1 << 2
STATUS_PUMP_ON = 1 << 3
STATUS_PHASE_LOCKED = 1 << 4
STATUS_ECC_ERR = 1 << 5
STATUS_OVERTEMP = 1 << 6
STATUS_STATE_SHIFT = 8
STATE_IDLE, STATE_RUN, STATE_QUENCHED, STATE_RECOVERING = 0, 1, 2, 3

IRQ_QUENCH = 1 << 0
IRQ_ECC_CE = 1 << 1
IRQ_ECC_UE = 1 << 2
IRQ_PHASE_SLIP = 1 << 3
IRQ_PUMP_FAULT = 1 << 4
IRQ_OVERTEMP = 1 << 5
IRQ_RING_FULL = 1 << 6
IRQ_TIMER = 1 << 7
IRQ_ALL = 0xFF

PUMP_ATTEN_APPLY = 1 << 31
PUMP_ATTEN_DB_MASK = 0xFFFF
PUMP_ATTEN_QUENCH_DB_X100 = 3000

ECC_CTRL_ENABLE = 1 << 0
ECC_CTRL_SCRUB = 1 << 1
ECC_CTRL_INJECT_SE = 1 << 2
ECC_CTRL_INJECT_DE = 1 << 3
ECC_CTRL_CLEAR = 1 << 4

ECC_STATUS_CE = 1 << 0
ECC_STATUS_UE = 1 << 1
ECC_STATUS_CE_CNT_SHIFT = 8
ECC_STATUS_UE_CNT_SHIFT = 16

ECC_SYN_HAMMING_MASK = 0x7F
ECC_SYN_PARITY = 1 << 7
ECC_SYN_VALID = 1 << 8

PHASE_VALUE_MASK = 0x00FFFFFF
PHASE_LOCK = 1 << 31

U32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# SECDED Hamming(72,64) — identical layout to the kernel implementation
# ---------------------------------------------------------------------------
class Secded7264:
    """SECDED Hamming(72,64): 7 positional check bits + overall parity."""

    def __init__(self) -> None:
        self.masks = [0] * 7
        self.data_of_pos = [-1] * 72
        d = 0
        for p in range(1, 72):
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

    def decode(self, data: int, check: int) -> Tuple[str, int, int, int]:
        """Return (result, corrected_data, corrected_check, syndrome)."""
        syn = (check ^ self.hamming_bits(data)) & 0x7F
        parity = self._parity(data) ^ self._parity(check)
        syndrome_reg = syn | (parity << 7)
        if syn == 0 and parity == 0:
            return "ok", data, check, syndrome_reg
        if syn == 0 and parity == 1:
            return "corrected_parity", data, check ^ 0x80, syndrome_reg
        if parity == 0:
            return "uncorrectable", data, check, syndrome_reg
        if syn & (syn - 1) == 0:
            j = syn.bit_length() - 1
            return "corrected_check", data, check ^ (1 << j), syndrome_reg
        db = self.data_of_pos[syn]
        if db < 0:
            return "uncorrectable", data, check, syndrome_reg
        return "corrected_data", data ^ (1 << db), check, syndrome_reg


# ---------------------------------------------------------------------------
# 1. Virtual MMIO bus
# ---------------------------------------------------------------------------
@dataclass
class BusTransaction:
    kind: str          # "R" or "W"
    address: int
    value: int
    timestamp_ns: int


class VirtualMMIOBus:
    """Register-accurate model of the SHBT-R MMIO aperture at 0x70000000."""

    def __init__(self, base: int = MMIO_BASE, size: int = MMIO_SIZE) -> None:
        self.base = base
        self.size = size
        self.regs: Dict[int, int] = {}
        self.ecc = Secded7264()
        self.time_ns = 0
        self.log: List[BusTransaction] = []
        self.write_hooks: Dict[int, callable] = {
            REG_CTRL: self._on_ctrl,
            REG_IRQ_CLEAR: self._on_irq_clear,
            REG_PUMP_ATTEN: self._on_pump_atten,
            REG_ECC_CTRL: self._on_ecc_ctrl,
        }
        self.reset()

    # ---- helpers ---------------------------------------------------------
    def _offset(self, address: int) -> int:
        if address % 4:
            raise ValueError(f"Unaligned MMIO access at 0x{address:08X}")
        off = address - self.base
        if not (0 <= off < self.size):
            raise ValueError(f"MMIO address 0x{address:08X} outside aperture")
        return off

    def reset(self) -> None:
        self.regs = {}
        self.regs[REG_STATUS] = STATUS_READY | (STATE_IDLE << STATUS_STATE_SHIFT)
        self.regs[REG_TEMP_MK] = 4200
        self.regs[REG_ECC_CTRL] = ECC_CTRL_ENABLE
        for i in range(NUM_CHANNELS):
            self.regs[REG_PHASE_BASE + 4 * i] = 0
        self.ce_count = 0
        self.ue_count = 0

    def tick(self, ns: int) -> None:
        self.time_ns += int(ns)

    def _reg(self, off: int) -> int:
        return self.regs.get(off, 0) & U32

    def _set(self, off: int, value: int) -> None:
        self.regs[off] = value & U32

    # ---- bus interface -----------------------------------------------------
    def read32(self, address: int) -> int:
        off = self._offset(address)
        if off == REG_TIMER_LO:
            value = self.time_ns & U32
        elif off == REG_TIMER_HI:
            value = (self.time_ns >> 32) & U32
        elif off == REG_IRQ_CLEAR:
            value = self._reg(REG_IRQ_STATUS)      # latched faults read back
        else:
            value = self._reg(off)
        self.log.append(BusTransaction("R", address, value, self.time_ns))
        self.tick(4)
        return value

    def write32(self, address: int, value: int) -> None:
        off = self._offset(address)
        value &= U32
        self.log.append(BusTransaction("W", address, value, self.time_ns))
        hook = self.write_hooks.get(off)
        if hook is not None:
            hook(value)
        elif off in (REG_STATUS, REG_IRQ_STATUS, REG_TIMER_LO, REG_TIMER_HI):
            pass                                    # read-only
        else:
            self._set(off, value)
        self.tick(4)

    def read_phase(self, channel: int) -> int:
        return self.read32(self.base + REG_PHASE_BASE + 4 * channel)

    def write_phase(self, channel: int, value: int) -> None:
        self.write32(self.base + REG_PHASE_BASE + 4 * channel, value)

    # ---- device-side fault injection (not visible to firmware) -------------
    def raise_fault(self, irq_bits: int) -> None:
        """Hardware raises an interrupt; STATUS mirrors the fault class."""
        self._set(REG_IRQ_STATUS, self._reg(REG_IRQ_STATUS) | irq_bits)
        status = self._reg(REG_STATUS)
        if irq_bits & IRQ_QUENCH:
            status = (status & ~(0xF << STATUS_STATE_SHIFT)) | STATUS_QUENCH | (STATE_QUENCHED << STATUS_STATE_SHIFT)
            status &= ~STATUS_PHASE_LOCKED
        if irq_bits & (IRQ_ECC_CE | IRQ_ECC_UE):
            status |= STATUS_ECC_ERR
        if irq_bits & IRQ_OVERTEMP:
            status |= STATUS_OVERTEMP
        if irq_bits & IRQ_PHASE_SLIP:
            status &= ~STATUS_PHASE_LOCKED
        self._set(REG_STATUS, status)

    def latch_ecc_word(self, data: int, check: int, address: int = 0) -> None:
        """Device latches a 72-bit codeword read from cryo SRAM and evaluates SECDED."""
        self._set(REG_ECC_DATA_LO, data & U32)
        self._set(REG_ECC_DATA_HI, (data >> 32) & U32)
        self._set(REG_ECC_CHECK, check & 0xFF)
        self._set(REG_ECC_ADDR, address)
        self._evaluate_secded()

    def _evaluate_secded(self) -> str:
        data = (self._reg(REG_ECC_DATA_HI) << 32) | self._reg(REG_ECC_DATA_LO)
        check = self._reg(REG_ECC_CHECK) & 0xFF
        result, _, _, syn = self.ecc.decode(data, check)
        self._set(REG_ECC_SYNDROME, syn | ECC_SYN_VALID)
        status = self._reg(REG_ECC_STATUS)
        if result.startswith("corrected"):
            self.ce_count = (self.ce_count + 1) & 0xFF
            status |= ECC_STATUS_CE
            self.raise_fault(IRQ_ECC_CE)
        elif result == "uncorrectable":
            self.ue_count = (self.ue_count + 1) & 0xFF
            status |= ECC_STATUS_UE
            self.raise_fault(IRQ_ECC_UE)
        status = (status & 0xFF) | (self.ce_count << ECC_STATUS_CE_CNT_SHIFT) | (self.ue_count << ECC_STATUS_UE_CNT_SHIFT)
        self._set(REG_ECC_STATUS, status)
        return result

    # ---- write side effects ----------------------------------------------
    def _on_ctrl(self, value: int) -> None:
        if value & CTRL_SOFT_RESET:
            self.reset()
            return
        if value & CTRL_PHASE_RESET:
            for i in range(NUM_CHANNELS):
                self._set(REG_PHASE_BASE + 4 * i, 0)
            self._set(REG_STATUS, self._reg(REG_STATUS) | STATUS_PHASE_LOCKED)
        status = self._reg(REG_STATUS)
        if value & CTRL_QUENCH_ACK:
            status = (status & ~(0xF << STATUS_STATE_SHIFT)) | (STATE_RECOVERING << STATUS_STATE_SHIFT)
        if value & CTRL_PUMP_ENABLE:
            status |= STATUS_PUMP_ON
        else:
            status &= ~STATUS_PUMP_ON
        if value & CTRL_ENABLE and not (status & STATUS_QUENCH):
            status = (status & ~(0xF << STATUS_STATE_SHIFT)) | (STATE_RUN << STATUS_STATE_SHIFT)
        self._set(REG_STATUS, status)
        self._set(REG_CTRL, value & ~(CTRL_SOFT_RESET | CTRL_PHASE_RESET))

    def _on_irq_clear(self, value: int) -> None:
        pending = self._reg(REG_IRQ_STATUS) & ~value
        self._set(REG_IRQ_STATUS, pending)
        status = self._reg(REG_STATUS)
        if not pending & IRQ_QUENCH:
            status &= ~STATUS_QUENCH
        if not pending & (IRQ_ECC_CE | IRQ_ECC_UE):
            status &= ~STATUS_ECC_ERR
        if not pending & IRQ_OVERTEMP:
            status &= ~STATUS_OVERTEMP
        if pending == 0 and (status >> STATUS_STATE_SHIFT) & 0xF == STATE_RECOVERING:
            status = (status & ~(0xF << STATUS_STATE_SHIFT)) | (STATE_RUN << STATUS_STATE_SHIFT)
        self._set(REG_STATUS, status)

    def _on_pump_atten(self, value: int) -> None:
        self._set(REG_PUMP_ATTEN, value & ~PUMP_ATTEN_APPLY)
        if value & PUMP_ATTEN_APPLY and (value & PUMP_ATTEN_DB_MASK) >= PUMP_ATTEN_QUENCH_DB_X100:
            self._set(REG_STATUS, self._reg(REG_STATUS) & ~STATUS_PUMP_ON)

    def _on_ecc_ctrl(self, value: int) -> None:
        if value & ECC_CTRL_CLEAR:
            self._set(REG_ECC_STATUS, (self.ce_count << ECC_STATUS_CE_CNT_SHIFT) | (self.ue_count << ECC_STATUS_UE_CNT_SHIFT))
            self._set(REG_ECC_SYNDROME, 0)
        if value & ECC_CTRL_INJECT_SE:
            self._set(REG_ECC_DATA_LO, self._reg(REG_ECC_DATA_LO) ^ (1 << 5))
            self._evaluate_secded()
        if value & ECC_CTRL_INJECT_DE:
            self._set(REG_ECC_DATA_LO, self._reg(REG_ECC_DATA_LO) ^ 0b11)
            self._evaluate_secded()
        if value & ECC_CTRL_SCRUB:
            data = (self._reg(REG_ECC_DATA_HI) << 32) | self._reg(REG_ECC_DATA_LO)
            check = self._reg(REG_ECC_CHECK) & 0xFF
            result, d2, c2, _ = self.ecc.decode(data, check)
            if result.startswith("corrected"):
                self._set(REG_ECC_DATA_LO, d2 & U32)
                self._set(REG_ECC_DATA_HI, (d2 >> 32) & U32)
                self._set(REG_ECC_CHECK, c2)
        self._set(REG_ECC_CTRL, value & (ECC_CTRL_ENABLE | ECC_CTRL_SCRUB))


# ---------------------------------------------------------------------------
# 2. Matrix product state decoder
# ---------------------------------------------------------------------------
PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class MatrixProductStateDecoder:
    """Open-boundary MPS with tensors A[i] of shape (chi_l, 2, chi_r), chi <= 256.

    2-D lattices of shape (Lx, Ly) are snake-ordered onto the 1-D chain so that
    horizontal neighbours remain adjacent in the MPS and vertical neighbours are
    reached through ``neighbour_pairs_2d``.
    """

    def __init__(self, n_sites: Optional[int] = None, chi: int = 256,
                 lattice: Optional[Tuple[int, int]] = None, seed: Optional[int] = 0) -> None:
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

    # ---- state construction ------------------------------------------------
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

    def set_random_state(self, bond_dim: int) -> None:
        """Random normalised MPS with the given uniform bond dimension."""
        self.tensors = []
        for i in range(self.n):
            dl = 1 if i == 0 else bond_dim
            dr = 1 if i == self.n - 1 else bond_dim
            A = self.rng.normal(size=(dl, 2, dr)) + 1j * self.rng.normal(size=(dl, 2, dr))
            self.tensors.append(A / math.sqrt(2 * bond_dim))
        self.normalize()
        self.error_log = []

    # ---- 2-D helpers -------------------------------------------------------
    def site_index(self, x: int, y: int) -> int:
        if self.lattice is None:
            raise ValueError("Not a 2-D lattice")
        Lx, Ly = self.lattice
        col = x if y % 2 == 0 else Lx - 1 - x       # snake ordering
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

    # ---- error injection ---------------------------------------------------
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

    # ---- contraction -------------------------------------------------------
    def bond_dimensions(self) -> List[int]:
        return [A.shape[2] for A in self.tensors[:-1]]

    @staticmethod
    def _transfer(E: np.ndarray, A: np.ndarray, B: np.ndarray, O: Optional[np.ndarray] = None) -> np.ndarray:
        """E_{cd} = sum_{a,b,i,j} E_{ab} conj(A)_{aic} O_{ij} B_{bjd}, contracted pairwise (O(chi^3 d))."""
        if O is not None:
            B = np.einsum("ij,bjd->bid", O, B)
        # T_{a,i,d} = sum_b E_{ab} B_{bid}
        T = np.tensordot(E, B, axes=([1], [0]))
        # E'_{cd} = sum_{a,i} conj(A)_{aic} T_{aid}
        return np.tensordot(A.conj(), T, axes=([0, 1], [0, 1]))

    def norm(self) -> float:
        E = np.ones((1, 1), dtype=complex)
        for A in self.tensors:
            E = self._transfer(E, A, A)
        return float(np.real(E[0, 0]))

    def normalize(self) -> None:
        nrm = math.sqrt(self.norm())
        self.tensors[0] = self.tensors[0] / nrm

    def overlap(self, other: "MatrixProductStateDecoder") -> complex:
        E = np.ones((1, 1), dtype=complex)
        for A, B in zip(self.tensors, other.tensors):
            E = self._transfer(E, A, B)
        return complex(E[0, 0])

    def expectation_string(self, ops: Dict[int, np.ndarray]) -> float:
        """<psi| prod_i O_i |psi> via left-to-right transfer-matrix contraction."""
        E = np.ones((1, 1), dtype=complex)
        for i, A in enumerate(self.tensors):
            E = self._transfer(E, A, A, ops.get(i))
        return float(np.real(E[0, 0])) / self.norm()

    def zz_parity(self, i: int, j: int) -> float:
        return self.expectation_string({i: PAULI_Z, j: PAULI_Z})

    def syndrome_parities(self) -> np.ndarray:
        """Nearest-neighbour Z_i Z_{i+1} parities along the chain (length n-1)."""
        return np.array([self.zz_parity(i, i + 1) for i in range(self.n - 1)])

    def syndrome_parities_2d(self) -> Dict[Tuple[int, int], float]:
        return {pair: self.zz_parity(*pair) for pair in self.neighbour_pairs_2d()}

    def syndrome_bits(self) -> np.ndarray:
        """1 where the ZZ parity is violated (<ZZ> < 0)."""
        return (self.syndrome_parities() < 0.0).astype(int)

    def decode_x_errors_from_syndrome(self) -> List[int]:
        """Minimum-weight decode of X errors on a chain from ZZ defects.

        Defects mark the boundaries of flipped regions; the lighter of the two
        complementary interval sets is returned.
        """
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

    # ---- compression -------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. HIL virtual testbench
# ---------------------------------------------------------------------------
@dataclass
class TransientEvent:
    t_ns: int
    kind: str                       # "quench" | "ecc_se" | "ecc_de" | "phase_slip" | "overtemp"
    channel: Optional[int] = None
    payload: Dict[str, int] = field(default_factory=dict)


@dataclass
class RecoveryTrace:
    event: TransientEvent
    irq_before: int
    irq_after: int
    ecc_status: int
    ecc_syndrome: int
    ecc_result: str
    status_after: int
    pump_off_during_recovery: bool
    latency_ns: int


class HILVirtualTestbench:
    """Closed-loop HIL harness: virtual bus + MPS decoder + firmware recovery driver."""

    def __init__(self, bus: Optional[VirtualMMIOBus] = None,
                 decoder: Optional[MatrixProductStateDecoder] = None,
                 recovery_bound_ns: float = 105.90) -> None:
        self.bus = bus or VirtualMMIOBus()
        self.decoder = decoder or MatrixProductStateDecoder(n_sites=NUM_CHANNELS // 6, chi=256)
        self.recovery_bound_ns = recovery_bound_ns
        self.ecc = Secded7264()
        self.traces: List[RecoveryTrace] = []
        self.reference_word = 0xC0FFEE1234567890

    def addr(self, off: int) -> int:
        return self.bus.base + off

    # ---- firmware-side bring-up -------------------------------------------
    def kernel_init(self) -> None:
        b = self.bus
        b.write32(self.addr(REG_IRQ_ENABLE), IRQ_QUENCH | IRQ_ECC_CE | IRQ_ECC_UE | IRQ_PHASE_SLIP | IRQ_OVERTEMP)
        b.write32(self.addr(REG_ECC_CTRL), ECC_CTRL_ENABLE)
        b.write32(self.addr(REG_CTRL), CTRL_ENABLE | CTRL_ECC_ENABLE | CTRL_PUMP_ENABLE)
        for ch in range(NUM_CHANNELS):
            b.write_phase(ch, ((ch * 0x1357) & PHASE_VALUE_MASK) | PHASE_LOCK)
        b.write32(self.addr(REG_CTRL), CTRL_ENABLE | CTRL_ECC_ENABLE | CTRL_PUMP_ENABLE | CTRL_PHASE_RESET)

    # ---- hardware-side transient injection ---------------------------------
    def inject(self, event: TransientEvent) -> None:
        b = self.bus
        b.time_ns = max(b.time_ns, event.t_ns)
        if event.kind == "quench":
            b.raise_fault(IRQ_QUENCH)
            for ch in range(NUM_CHANNELS):
                b._set(REG_PHASE_BASE + 4 * ch, (0x800000 + ch) & PHASE_VALUE_MASK)
        elif event.kind == "ecc_se":
            bit = event.payload.get("bit", 17)
            check = self.ecc.encode(self.reference_word)
            b.latch_ecc_word(self.reference_word ^ (1 << bit), check, address=event.payload.get("addr", 0x100))
        elif event.kind == "ecc_de":
            b1, b2 = event.payload.get("bits", (3, 41))
            check = self.ecc.encode(self.reference_word)
            b.latch_ecc_word(self.reference_word ^ (1 << b1) ^ (1 << b2), check, address=event.payload.get("addr", 0x104))
        elif event.kind == "phase_slip":
            b.raise_fault(IRQ_PHASE_SLIP)
            ch = event.channel or 0
            b._set(REG_PHASE_BASE + 4 * ch, (b._reg(REG_PHASE_BASE + 4 * ch) + 0x400000) & PHASE_VALUE_MASK)
        elif event.kind == "overtemp":
            b._set(REG_TEMP_MK, event.payload.get("temp_mk", 9000))
            b.raise_fault(IRQ_OVERTEMP)
        else:
            raise ValueError(f"Unknown transient {event.kind}")

    # ---- firmware-side 4-step recovery ------------------------------------
    def recover(self, event: TransientEvent) -> RecoveryTrace:
        b = self.bus
        t0 = b.time_ns
        irq_before = b.read32(self.addr(REG_IRQ_CLEAR))            # latched faults @0x70000010
        recovery_mask = IRQ_QUENCH | IRQ_ECC_CE | IRQ_ECC_UE | IRQ_PHASE_SLIP
        irq = irq_before & recovery_mask

        # Step 1: acknowledge interrupts
        b.write32(self.addr(REG_IRQ_CLEAR), irq)
        b.write32(self.addr(REG_CTRL), b.read32(self.addr(REG_CTRL)) | CTRL_QUENCH_ACK)

        # Step 2: pump attenuation
        b.write32(self.addr(REG_PUMP_ATTEN), PUMP_ATTEN_QUENCH_DB_X100 | (2 << 16) | PUMP_ATTEN_APPLY)
        pump_off = not (b.read32(self.addr(REG_STATUS)) & STATUS_PUMP_ON)

        # Step 3: SECDED correction
        data = (b.read32(self.addr(REG_ECC_DATA_HI)) << 32) | b.read32(self.addr(REG_ECC_DATA_LO))
        check = b.read32(self.addr(REG_ECC_CHECK)) & 0xFF
        result, d2, c2, syn = self.ecc.decode(data, check)
        b.write32(self.addr(REG_ECC_DATA_LO), d2 & U32)
        b.write32(self.addr(REG_ECC_DATA_HI), (d2 >> 32) & U32)
        b.write32(self.addr(REG_ECC_CHECK), c2)
        ecc_status = b.read32(self.addr(REG_ECC_STATUS))
        ecc_syndrome = b.read32(self.addr(REG_ECC_SYNDROME))
        b.write32(self.addr(REG_ECC_CTRL), ECC_CTRL_ENABLE | ECC_CTRL_CLEAR)

        # Step 4: phase reset
        b.write32(self.addr(REG_CTRL), CTRL_ENABLE | CTRL_ECC_ENABLE | CTRL_PUMP_ENABLE | CTRL_PHASE_RESET)

        irq_after = b.read32(self.addr(REG_IRQ_CLEAR))
        status_after = b.read32(self.addr(REG_STATUS))
        trace = RecoveryTrace(event, irq_before, irq_after, ecc_status, ecc_syndrome, result,
                              status_after, pump_off, b.time_ns - t0)
        self.traces.append(trace)
        return trace

    # ---- closed loop --------------------------------------------------------
    def run_closed_loop(self, events: Sequence[TransientEvent]) -> List[RecoveryTrace]:
        traces = []
        for ev in sorted(events, key=lambda e: e.t_ns):
            self.inject(ev)
            traces.append(self.recover(ev))
        return traces

    def phase_slips_from_syndrome(self, sites_per_channel: int = 1) -> List[TransientEvent]:
        """Map MPS syndrome defects onto PHASE_SLIP transients on the matching channels."""
        bits = self.decoder.syndrome_bits()
        events = []
        for i, bit in enumerate(bits):
            if bit:
                ch = (i * sites_per_channel) % NUM_CHANNELS
                events.append(TransientEvent(t_ns=self.bus.time_ns + 10 * (i + 1), kind="phase_slip", channel=ch))
        return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_hil_tests() -> None:
    print("Running SHBT-R HIL / MPS-QEC self-tests...")

    # ---- VirtualMMIOBus ---------------------------------------------------
    bus = VirtualMMIOBus()
    assert bus.base == 0x70000000
    bus.write32(0x70000014, 0x1234)
    assert bus.read32(0x70000014) == 0x1234, "PUMP_ATTEN readback"
    bus.write32(0x70001000 + 4 * 311, 0xABCDEF)
    assert bus.read32(0x70001000 + 4 * 311) == 0xABCDEF, "PHASE[311] readback"
    try:
        bus.read32(0x70000002)
        raise AssertionError("unaligned access must fail")
    except ValueError:
        pass
    try:
        bus.read32(0x6FFFFFFC)
        raise AssertionError("out-of-aperture access must fail")
    except ValueError:
        pass
    t0 = bus.read32(0x70000034)
    bus.tick(1000)
    assert bus.read32(0x70000034) - t0 >= 1000, "TIMER advances"
    assert bus.read32(0x70000010) == 0, "no faults latched at reset"
    # Fault injection must update the latched fault register at 0x70000010
    bus.raise_fault(IRQ_QUENCH)
    assert bus.read32(0x70000010) & IRQ_QUENCH, "fault injection must set 0x70000010 QUENCH bit"
    assert bus.read32(0x70000008) & IRQ_QUENCH, "IRQ_STATUS mirrors"
    assert bus.read32(0x70000004) & STATUS_QUENCH
    bus.write32(0x70000010, IRQ_QUENCH)                               # W1C
    assert bus.read32(0x70000010) == 0, "W1C must clear 0x70000010"
    assert not bus.read32(0x70000004) & STATUS_QUENCH
    print("  VirtualMMIOBus: aperture, alignment, W1C fault latch OK")

    # ---- SECDED model matches kernel semantics -----------------------------
    ecc = Secded7264()
    rng = np.random.default_rng(1)
    for _ in range(500):
        d = int(rng.integers(0, 2**63)) | (int(rng.integers(0, 2)) << 63)
        c = ecc.encode(d)
        assert ecc.decode(d, c)[0] == "ok"
        b = int(rng.integers(0, 64))
        r, d2, c2, _ = ecc.decode(d ^ (1 << b), c)
        assert r == "corrected_data" and d2 == d and c2 == c
        b2 = (b + 1 + int(rng.integers(0, 63))) % 64
        assert ecc.decode(d ^ (1 << b) ^ (1 << b2), c)[0] == "uncorrectable"
    print("  SECDED(72,64): SEC + DED OK")

    # ---- MatrixProductStateDecoder ----------------------------------------
    mps = MatrixProductStateDecoder(n_sites=12, chi=256)
    assert mps.chi == 256
    assert abs(mps.norm() - 1.0) < 1e-12
    assert np.allclose(mps.syndrome_parities(), 1.0), "product |0..0> has all ZZ = +1"
    mps.inject_x_error(4)
    par = mps.syndrome_parities()
    assert par[3] < 0 and par[4] < 0 and np.sum(par < 0) == 2, "X error creates two ZZ defects"
    assert mps.decode_x_errors_from_syndrome() == [4], "minimum-weight decode recovers site"
    mps.inject_x_error(5)
    assert mps.decode_x_errors_from_syndrome() == [4, 5]
    mps.inject_z_error(2)
    assert np.sum(mps.syndrome_parities() < 0) == 2, "Z errors are invisible to ZZ syndromes"
    assert abs(mps.norm() - 1.0) < 1e-12, "Paulis are unitary"

    ghz = MatrixProductStateDecoder(n_sites=8, chi=256)
    ghz.set_ghz_state()
    assert abs(ghz.norm() - 1.0) < 1e-12
    assert np.allclose(ghz.syndrome_parities(), 1.0), "GHZ is ZZ-stabilised"
    assert abs(ghz.expectation_string({0: PAULI_Z})) < 1e-12, "<Z_0> = 0 for GHZ"
    assert abs(ghz.expectation_string({i: PAULI_X for i in range(8)}) - 1.0) < 1e-12, "X^n stabiliser"
    ghz.inject_x_error(3)
    assert np.sum(ghz.syndrome_parities() < 0) == 2

    # Entangling gate growth + chi=256 truncation
    ent = MatrixProductStateDecoder(n_sites=10, chi=256, seed=3)
    ent.set_random_state(bond_dim=300)
    assert max(ent.bond_dimensions()) == 300
    ent_ref = MatrixProductStateDecoder(n_sites=10, chi=256, seed=3)
    ent_ref.tensors = [A.copy() for A in ent.tensors]
    discarded = ent.truncate(256)
    assert max(ent.bond_dimensions()) <= 256, "bond dimension capped at chi = 256"
    fid = abs(ent.overlap(ent_ref)) ** 2 / (ent.norm() * ent_ref.norm())
    assert 0.5 < fid <= 1.0 + 1e-9, "truncation retains dominant weight"
    assert discarded >= 0.0
    cnot = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
    ent.apply_two_site(4, cnot)
    assert max(ent.bond_dimensions()) <= 256

    # 2-D snake lattice
    lat = MatrixProductStateDecoder(lattice=(4, 3), chi=256)
    assert lat.n == 12
    pairs = lat.neighbour_pairs_2d()
    assert len(pairs) == 3 * 3 + 4 * 2, "4x3 lattice has 17 nearest-neighbour bonds"
    lat.inject_x_error(lat.site_index(1, 1))
    syn2d = lat.syndrome_parities_2d()
    defects = [p for p, v in syn2d.items() if v < 0]
    assert len(defects) == 4, "interior X error violates its four incident ZZ bonds"
    print("  MPS decoder: X/Z injection, ZZ syndromes, chi=256 truncation, 2-D lattice OK")

    # ---- HILVirtualTestbench ----------------------------------------------
    tb = HILVirtualTestbench()
    tb.kernel_init()
    assert tb.bus.read32(0x70000004) & STATUS_PHASE_LOCKED
    assert (tb.bus.read32(0x70000004) >> 8) & 0xF == STATE_RUN

    ev_q = TransientEvent(t_ns=1_000, kind="quench")
    tb.inject(ev_q)
    latched = tb.bus.read32(0x70000010)
    assert latched & IRQ_QUENCH, "quench fault must appear in 0x70000010"
    assert tb.bus.read32(0x70000004) & STATUS_QUENCH
    tr = tb.recover(ev_q)
    assert tr.irq_before & IRQ_QUENCH and not tr.irq_after & IRQ_QUENCH, "recovery clears 0x70000010"
    assert not tr.status_after & STATUS_QUENCH
    assert tr.status_after & STATUS_PHASE_LOCKED
    assert all(tb.bus.read_phase(ch) == 0 for ch in (0, 155, 311)), "phase registers reset"
    assert tr.pump_off_during_recovery, "pump attenuated during recovery"
    assert tr.status_after & STATUS_PUMP_ON, "pump re-enabled after phase reset"
    assert tb.bus.read32(0x70000014) & PUMP_ATTEN_DB_MASK == PUMP_ATTEN_QUENCH_DB_X100

    ev_se = TransientEvent(t_ns=2_000, kind="ecc_se", payload={"bit": 17})
    tb.inject(ev_se)
    assert tb.bus.read32(0x70000010) & IRQ_ECC_CE, "SE fault must set ECC_CE in 0x70000010"
    st = tb.bus.read32(0x7000001C)
    assert st & ECC_STATUS_CE and (st >> 8) & 0xFF == 1, "SECDED status: CE latched, count 1"
    assert tb.bus.read32(0x70000020) & ECC_SYN_VALID
    tr = tb.recover(ev_se)
    assert tr.ecc_result == "corrected_data"
    corrected = (tb.bus.read32(0x7000002C) << 32) | tb.bus.read32(0x70000028)
    assert corrected == tb.reference_word, "firmware wrote corrected word back"
    assert not tr.irq_after & IRQ_ECC_CE
    assert not tb.bus.read32(0x7000001C) & (ECC_STATUS_CE | ECC_STATUS_UE), "ECC_CTRL_CLEAR clears flags"
    assert (tb.bus.read32(0x7000001C) >> 8) & 0xFF == 1, "CE counter persists"

    ev_de = TransientEvent(t_ns=3_000, kind="ecc_de", payload={"bits": (3, 41)})
    tb.inject(ev_de)
    assert tb.bus.read32(0x70000010) & IRQ_ECC_UE, "DE fault must set ECC_UE in 0x70000010"
    st = tb.bus.read32(0x7000001C)
    assert st & ECC_STATUS_UE and (st >> 16) & 0xFF == 1, "SECDED status: UE latched"
    tr = tb.recover(ev_de)
    assert tr.ecc_result == "uncorrectable"
    assert not (tr.ecc_syndrome & ECC_SYN_PARITY) and (tr.ecc_syndrome & 0x7F) != 0, "DED signature"

    # Closed loop driven from MPS syndromes
    tb.decoder.set_product_state(0)
    tb.decoder.inject_x_error(7)
    slips = tb.phase_slips_from_syndrome(sites_per_channel=6)
    assert len(slips) == 2 and {e.channel for e in slips} == {36, 42}
    traces = tb.run_closed_loop(slips + [TransientEvent(t_ns=tb.bus.time_ns + 500, kind="overtemp")])
    assert len(traces) == 3
    assert all(t.irq_before & (IRQ_PHASE_SLIP | IRQ_OVERTEMP) for t in traces)
    assert all(not t.irq_after & IRQ_PHASE_SLIP for t in traces)
    assert tb.bus.read32(0x70000010) & IRQ_OVERTEMP, "OVERTEMP is not part of the recovery mask"
    tb.bus.write32(0x70000010, IRQ_OVERTEMP)
    assert tb.bus.read32(0x70000010) == 0
    assert all(t.latency_ns > 0 for t in tb.traces)
    n_bus_ops = len(tb.bus.log)
    assert n_bus_ops > 400, "non-trivial bus activity"
    print(f"  HIL testbench: {len(tb.traces)} transients recovered, {n_bus_ops} bus transactions, "
          f"model latency {tb.traces[0].latency_ns} ns (4 ns/access)")

    print("All HIL / MPS-QEC self-tests passed.")


if __name__ == "__main__":
    run_hil_tests()
