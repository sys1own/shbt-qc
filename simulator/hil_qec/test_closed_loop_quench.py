"""Closed-loop HIL QEC benchmark tests for the SHBT-R quantum control stack.

These tests exercise the ctypes microkernel bridge, the dictionary-backed
SHBT-MMIO-1 virtual bus, and the MPS-QEC decoder through the high-level
``HILClosedLoop`` harness.
"""

from __future__ import annotations

import pytest

from simulator.hil_qec.hil_mps_qec import (
    HILClosedLoop,
    SHBT_STATUS_FAULT_ST,
    SHBT_STATUS_OVERTEMP,
)


# -----------------------------------------------------------------------------
# Recovery timing and ECC blanking tests
# -----------------------------------------------------------------------------

def test_quench_recovery_time():
    """Trigger a thermal-quench recovery and assert completion within 120 ns.

    The microkernel's ``shbt_recover`` enforces the 120 ns budget internally
    and returns ``0`` on success, ``-2`` if the budget is exceeded.  This test
    therefore uses the C return code as the authoritative timing verdict.
    """
    loop = HILClosedLoop()
    loop.trigger_quench_interrupt(channel=5)

    rc = loop.run_hardware_recovery()

    assert rc == 0, f"recovery exceeded 120 ns timing budget (rc={rc})"
    assert loop.bus.regs.blank == 0, "RF blanking must be de-asserted after recovery"
    assert loop.bus.regs.fault_latch == 0, "fault latch must be cleared after recovery"
    assert loop.bus.regs.pll_ctrl == 1, "PLL lock request must remain asserted"
    assert not (loop.bus.regs.status & (SHBT_STATUS_OVERTEMP | SHBT_STATUS_FAULT_ST)), \
        "overtemperature and fault status bits must be cleared"


def test_double_bit_ecc_blanking():
    """Inject a double-bit ECC error and verify it is detected and forces blanking."""
    loop = HILClosedLoop()
    payload = 0xC0FFEE1234567890
    bit_a, bit_b = 3, 41

    # First, verify the pure decoder returns double_bit_error=True.
    check = loop.bridge.ecc_encode(payload)
    corrupted = payload ^ (1 << bit_a) ^ (1 << bit_b)
    dec = loop.bridge.ecc_decode(corrupted, check)
    assert dec.double_bit_error, "double-bit ECC error must be detected by decoder"
    assert not dec.single_bit_error

    # Latch the corrupted word into the MMIO bus and run hardware recovery.
    loop.bus.set_ecc_payload(corrupted)
    loop.bus.regs.ecc_check = check & 0xFF

    rc = loop.run_hardware_recovery()

    assert rc == -1, f"double-bit error must force fail-closed state (rc={rc})"
    assert loop.bus.regs.blank == 1, "global RF blanking must be asserted"
    assert loop.bus.regs.control == 0, "control outputs must be disabled"
