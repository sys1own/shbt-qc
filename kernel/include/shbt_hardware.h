/*
 * shbt_hardware.h - SHBT-MMIO-1 normative memory-mapped hardware interface.
 *
 * Freestanding: depends only on compiler-provided <stdint.h> and <stddef.h>.
 *
 * The MMIO aperture for the SHBT-R quantum computer control plane is a
 * compact, 56-byte volatile register block at 0x70000000.  The ABI version
 * register reports interface revision 1.
 */
#ifndef SHBT_HARDWARE_H
#define SHBT_HARDWARE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Aperture and ABI version
 * -------------------------------------------------------------------------- */
#define SHBT_MMIO_BASE              0x70000000U
#define SHBT_MMIO_ABI_VERSION       1U

/* --------------------------------------------------------------------------
 * status register bitmasks (SHBT-MMIO-1 §3.1)
 * -------------------------------------------------------------------------- */
#define SHBT_STATUS_OVERTEMP        (1U << 0)
#define SHBT_STATUS_PLL_LOCK        (1U << 1)
#define SHBT_STATUS_ECC_ERR         (1U << 2)
#define SHBT_STATUS_FAULT_ST        (1U << 3)

/* --------------------------------------------------------------------------
 * Normative MMIO register block
 *
 * Offsets 0x00 .. 0x34 inclusive, 14 x uint32_t fields, total span 0x38 bytes.
 * -------------------------------------------------------------------------- */
typedef struct {
    volatile uint32_t status;           /* 0x00 */
    volatile uint32_t blank;            /* 0x04 */
    volatile uint32_t fifo_data;        /* 0x08 */
    volatile uint32_t pll_ctrl;         /* 0x0C */
    volatile uint32_t ecc_low;          /* 0x10 */
    volatile uint32_t ecc_high;         /* 0x14 */
    volatile uint32_t ecc_check;        /* 0x18 */
    volatile uint32_t ecc_commit;       /* 0x1C */
    volatile uint32_t fault_latch;      /* 0x20 */
    volatile uint32_t channel_select;   /* 0x24 */
    volatile uint32_t abi_version;      /* 0x28 */
    volatile uint32_t phase_offset;     /* 0x2C */
    volatile uint32_t ecc_counts;       /* 0x30 */
    volatile uint32_t control;          /* 0x34 */
} ShbtRegisters;                          /* 0x38 */

#define SHBT_MMIO   ((ShbtRegisters *)SHBT_MMIO_BASE)

/* --------------------------------------------------------------------------
 * Compile-time layout verification
 * -------------------------------------------------------------------------- */
_Static_assert(offsetof(ShbtRegisters, status)         == 0x00U, "ShbtRegisters.status offset");
_Static_assert(offsetof(ShbtRegisters, blank)          == 0x04U, "ShbtRegisters.blank offset");
_Static_assert(offsetof(ShbtRegisters, fifo_data)      == 0x08U, "ShbtRegisters.fifo_data offset");
_Static_assert(offsetof(ShbtRegisters, pll_ctrl)       == 0x0CU, "ShbtRegisters.pll_ctrl offset");
_Static_assert(offsetof(ShbtRegisters, ecc_low)        == 0x10U, "ShbtRegisters.ecc_low offset");
_Static_assert(offsetof(ShbtRegisters, ecc_high)       == 0x14U, "ShbtRegisters.ecc_high offset");
_Static_assert(offsetof(ShbtRegisters, ecc_check)      == 0x18U, "ShbtRegisters.ecc_check offset");
_Static_assert(offsetof(ShbtRegisters, ecc_commit)     == 0x1CU, "ShbtRegisters.ecc_commit offset");
_Static_assert(offsetof(ShbtRegisters, fault_latch)    == 0x20U, "ShbtRegisters.fault_latch offset");
_Static_assert(offsetof(ShbtRegisters, channel_select) == 0x24U, "ShbtRegisters.channel_select offset");
_Static_assert(offsetof(ShbtRegisters, abi_version)    == 0x28U, "ShbtRegisters.abi_version offset");
_Static_assert(offsetof(ShbtRegisters, phase_offset)   == 0x2CU, "ShbtRegisters.phase_offset offset");
_Static_assert(offsetof(ShbtRegisters, ecc_counts)     == 0x30U, "ShbtRegisters.ecc_counts offset");
_Static_assert(offsetof(ShbtRegisters, control)         == 0x34U, "ShbtRegisters.control offset");
_Static_assert(sizeof(ShbtRegisters)                   == 0x38U, "ShbtRegisters total size");

#ifdef __cplusplus
}
#endif

#endif /* SHBT_HARDWARE_H */
