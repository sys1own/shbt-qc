/*
 * shbt_user_mmio.c - User-space shim for the SHBT-R microkernel.
 *
 * This wrapper exposes a relocatable SHBT_MMIO pointer so that the same
 * freestanding microkernel can be loaded as a shared library and exercised
 * from Python/ctypes without assuming a fixed physical address.
 *
 * Do not include this file in a real bare-metal build; it is intended only
 * for host-side HIL simulation.
 */

#include "shbt_hardware.h"

/* Global MMIO aperture pointer.  Python sets this before calling shbt_recover(). */
ShbtRegisters *shbt_mmio = (ShbtRegisters *)0;

void shbt_set_mmio(ShbtRegisters *ptr)
{
    shbt_mmio = ptr;
}

ShbtRegisters *shbt_get_mmio(void)
{
    return shbt_mmio;
}

/* Redirect the bare-metal macro to the relocatable pointer. */
#undef SHBT_MMIO
#define SHBT_MMIO shbt_mmio

#include "shbt_core_runtime.c"
