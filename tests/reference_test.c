/*
 * tests/reference_test.c - C reference test suite for the SHBT-R microkernel.
 *
 * This is a hosted test runner that exercises the freestanding core functions
 * through the user-space MMIO relocation shim in kernel/src/shbt_user_mmio.c.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "shbt_hardware.h"

/* Pull in the user-space shim and the entire freestanding microkernel as a
 * single translation unit so the secded_result_t definition is visible. */
#include "../kernel/src/shbt_user_mmio.c"

static int fail(const char *msg)
{
    fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main(void)
{
    ShbtRegisters regs;

    /* --- register block header assertions are checked at compile time --- */
    if (sizeof(ShbtRegisters) != 0x38U)
        return fail("ShbtRegisters size mismatch");

    /* --- ECC clean roundtrip --- */
    uint64_t data = 0x123456789ABCDEF0ULL;
    uint8_t code = shbt_ecc_encode(data);
    secded_result_t dec = shbt_ecc_decode(data, code);
    if (dec.corrected_data != data || dec.corrected_check != code ||
        dec.single_bit_error || dec.double_bit_error)
        return fail("ECC clean roundtrip");

    /* --- Correct all 64 single-bit data errors --- */
    for (int i = 0; i < 64; ++i) {
        uint64_t bad = data ^ (1ULL << i);
        secded_result_t r = shbt_ecc_decode(bad, code);
        if (r.corrected_data != data || !r.single_bit_error || r.double_bit_error)
            return fail("single data-bit correction");
    }

    /* --- Correct all 8 single-bit check-code errors --- */
    for (int i = 0; i < 8; ++i) {
        uint8_t bad_code = code ^ (1U << i);
        secded_result_t r = shbt_ecc_decode(data, bad_code);
        if (r.corrected_check != code || !r.single_bit_error || r.double_bit_error)
            return fail("single check-bit correction");
    }

    /* --- Detect double-bit data errors --- */
    for (int i = 0; i < 64; i += 7) {
        int j = (i + 17) % 64;
        if (i == j) continue;
        uint64_t bad = data ^ (1ULL << i) ^ (1ULL << j);
        secded_result_t r = shbt_ecc_decode(bad, code);
        if (!r.double_bit_error)
            return fail("double data-bit detection");
    }

    /* --- Recovery: overtemp quench clears within budget and PLL re-locks --- */
    memset(&regs, 0, sizeof(regs));
    shbt_set_mmio(&regs);
    regs.abi_version = SHBT_MMIO_ABI_VERSION;
    regs.status = SHBT_STATUS_OVERTEMP | SHBT_STATUS_PLL_LOCK;

    int32_t rc = shbt_recover();
    if (rc != 0)
        return fail("shbt_recover() returned non-zero on overtemp quench");
    if (regs.blank != 0U)
        return fail("blank must be de-asserted after recovery");
    if (regs.fault_latch != 0U)
        return fail("fault_latch must be cleared after recovery");
    if (regs.status & (SHBT_STATUS_OVERTEMP | SHBT_STATUS_ECC_ERR | SHBT_STATUS_FAULT_ST))
        return fail("fault status bits must be cleared");
    if (!(regs.status & SHBT_STATUS_PLL_LOCK))
        return fail("PLL lock must remain set after recovery");

    /* --- Recovery: double-bit ECC error forces fail-closed blanking --- */
    memset(&regs, 0, sizeof(regs));
    shbt_set_mmio(&regs);
    regs.abi_version = SHBT_MMIO_ABI_VERSION;
    regs.status = SHBT_STATUS_OVERTEMP | SHBT_STATUS_PLL_LOCK;
    regs.fault_latch = 1U;

    uint64_t payload = 0xC0FFEE1234567890ULL;
    uint8_t good_code = shbt_ecc_encode(payload);
    uint64_t bad_payload = payload ^ (1ULL << 3) ^ (1ULL << 41);
    regs.ecc_low = (uint32_t)bad_payload;
    regs.ecc_high = (uint32_t)(bad_payload >> 32);
    regs.ecc_check = good_code;

    rc = shbt_recover();
    if (rc != -1)
        return fail("double-bit ECC error must return -1 (fail-closed)");
    if (regs.blank != 1U)
        return fail("blank must remain asserted on uncorrectable ECC error");
    if (regs.control != 0U)
        return fail("control must be zeroed on uncorrectable ECC error");

    /* --- AVX-512 Givens rotation column remapping --- */
    double orig_a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double orig_b[8] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    double a[8], b[8];
    memcpy(a, orig_a, sizeof(a));
    memcpy(b, orig_b, sizeof(b));

    double c = 0.6, s = 0.8;
    shbt_remap(a, b, c, s, 8);

    const double eps = 1e-12;
    for (int i = 0; i < 8; ++i) {
        double exp_a = c * orig_a[i] + s * orig_b[i];
        double exp_b = -s * orig_a[i] + c * orig_b[i];
        if (fabs(a[i] - exp_a) > eps || fabs(b[i] - exp_b) > eps)
            return fail("Givens remap mismatch");
    }

    printf("OK: SHBT-R C reference tests passed\n");
    return 0;
}
