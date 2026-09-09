/*
 * shbt_core_runtime.c - SHBT-R bare-metal microkernel core.
 *
 * Freestanding C11.  No standard C runtime is used: there is no heap, no
 * libc, and no hosted assumptions.  The only headers are compiler-provided
 * (<stdint.h>, <stddef.h>, <immintrin.h> for AVX-512) and the hardware MMIO
 * definition in kernel/include/shbt_hardware.h.
 */

#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>
#include "shbt_hardware.h"

/* --------------------------------------------------------------------------
 * Platform timing (x86 TSC; fallback is intentionally empty for non-x86)
 * -------------------------------------------------------------------------- */
#ifndef SHBT_TSC_HZ
#define SHBT_TSC_HZ 3000000000ULL   /* 3.0 GHz default; override at build time */
#endif

static inline uint64_t shbt_cycles(void)
{
#if defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ __volatile__("lfence\n\trdtsc" : "=a"(lo), "=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | lo;
#else
    return 0ULL;
#endif
}

/* --------------------------------------------------------------------------
 * SECDED Hamming(72,64) ECC
 * -------------------------------------------------------------------------- */

/* Data bits covered by each of the 7 Hamming check bits. */
static const uint64_t shbt_ecc_mask[7] = {
    UINT64_C(0xab55555556aaad5b),
    UINT64_C(0xcd9999999b33366d),
    UINT64_C(0xf1e1e1e1e3c3c78e),
    UINT64_C(0x01fe01fe03fc07f0),
    UINT64_C(0x001fffe0003fff800),
    UINT64_C(0x001fffffffc000000),
    UINT64_C(0xfe00000000000000),
};

/* Map Hamming syndrome (codeword position) to data bit index, or -1. */
static const int8_t shbt_ecc_data_of_pos[128] = {
     -1,  -1,  -1,   0,  -1,   1,   2,   3,
     -1,   4,   5,   6,   7,   8,   9,  10,
     -1,  11,  12,  13,  14,  15,  16,  17,
     18,  19,  20,  21,  22,  23,  24,  25,
     -1,  26,  27,  28,  29,  30,  31,  32,
     33,  34,  35,  36,  37,  38,  39,  40,
     41,  42,  43,  44,  45,  46,  47,  48,
     49,  50,  51,  52,  53,  54,  55,  56,
     -1,  57,  58,  59,  60,  61,  62,  63,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
};

typedef struct {
    uint64_t corrected_data;
    uint8_t  corrected_check;
    uint8_t  single_bit_error;
    uint8_t  double_bit_error;
} secded_result_t;

static inline uint8_t shbt_ecc_hamming_bits(uint64_t data)
{
    uint8_t c = 0;
    for (int j = 0; j < 7; ++j)
        c |= (uint8_t)(__builtin_parityll(data & shbt_ecc_mask[j]) << j);
    return c;
}

uint8_t shbt_ecc_encode(uint64_t data)
{
    uint8_t c = shbt_ecc_hamming_bits(data);
    uint8_t overall = (uint8_t)(__builtin_parityll(data) ^ __builtin_parity(c));
    return (uint8_t)(c | (overall << 7));
}

secded_result_t shbt_ecc_decode(uint64_t data, uint8_t check_code)
{
    uint8_t expected = shbt_ecc_hamming_bits(data);
    uint8_t syn = (uint8_t)((check_code ^ expected) & 0x7FU);
    uint8_t parity = (uint8_t)(__builtin_parityll(data) ^ __builtin_parity(check_code));

    secded_result_t r = { data, check_code, 0, 0 };

    if (syn == 0 && parity == 0)
        return r;

    if (syn == 0 && parity == 1) {
        /* Single-bit error in the overall parity bit. */
        r.corrected_check ^= 0x80U;
        r.single_bit_error = 1;
        return r;
    }

    if (parity == 0) {
        /* Even number of bit flips (detected but uncorrectable). */
        r.double_bit_error = 1;
        return r;
    }

    /* Single-bit error at codeword position `syn`. */
    if ((syn & (syn - 1)) == 0) {
        /* Check-bit position (power of two in 1..64). */
        int j = __builtin_ctz((unsigned int)syn);
        r.corrected_check ^= (uint8_t)(1U << j);
        r.single_bit_error = 1;
        return r;
    }

    int8_t db = shbt_ecc_data_of_pos[syn];
    if (db < 0) {
        /* Syndrome points outside the valid 72-bit codeword. */
        r.double_bit_error = 1;
        return r;
    }

    r.corrected_data ^= (UINT64_C(1) << db);
    r.single_bit_error = 1;
    return r;
}

/* --------------------------------------------------------------------------
 * Post-quench recovery driver
 * -------------------------------------------------------------------------- */

#define SHBT_RECOVERY_BUDGET_NS 120ULL
#define SHBT_RECOVERY_PLL_WAIT_ITERS 1000000U

int32_t shbt_recover(void)
{
    ShbtRegisters *hw = SHBT_MMIO;
    uint64_t t0 = shbt_cycles();

    /* 1) Inspect/quarantine the fault condition. */
    (void)hw->status;
    (void)hw->fault_latch;

    /* 2) Assert global RF blanking. */
    hw->blank = 1U;

    /* 3) Flush and correct the latched ECC word. */
    uint64_t data = ((uint64_t)hw->ecc_high << 32) | (uint64_t)hw->ecc_low;
    uint8_t check = (uint8_t)hw->ecc_check;
    secded_result_t dec = shbt_ecc_decode(data, check);

    if (dec.double_bit_error) {
        /* Fail-closed: disable control outputs and leave the system blanked. */
        hw->control = 0U;
        return -1;
    }

    if (dec.single_bit_error) {
        hw->ecc_low = (uint32_t)(dec.corrected_data);
        hw->ecc_high = (uint32_t)(dec.corrected_data >> 32);
        hw->ecc_check = dec.corrected_check;
        hw->ecc_commit = 1U;
    }

    /* 4) Request PLL lock, wait for confirmation, then clear the fault state. */
    hw->pll_ctrl = 1U;

    uint32_t wait = 0;
    while (!(hw->status & SHBT_STATUS_PLL_LOCK)) {
        if (++wait >= SHBT_RECOVERY_PLL_WAIT_ITERS)
            return -2;
    }

    /* Clear fault status bits before de-asserting RF blanking. */
    hw->status &= ~(SHBT_STATUS_OVERTEMP | SHBT_STATUS_ECC_ERR | SHBT_STATUS_FAULT_ST);
    hw->blank = 0U;
    hw->fault_latch = 0U;

    uint64_t delta = shbt_cycles() - t0;
    uint64_t budget_cycles = (SHBT_TSC_HZ * SHBT_RECOVERY_BUDGET_NS) / 1000000000ULL;
    if (delta > budget_cycles)
        return -2;

    return 0;
}

/* --------------------------------------------------------------------------
 * AVX-512 Givens rotation column remapping
 * -------------------------------------------------------------------------- */

void shbt_remap(double *col_a, double *col_b, double c, double s, size_t n)
{
#if defined(__AVX512F__)
    __m512d vc = _mm512_set1_pd(c);
    __m512d vs = _mm512_set1_pd(s);

    for (; n >= 8; n -= 8, col_a += 8, col_b += 8) {
        __m512d a = _mm512_loadu_pd(col_a);
        __m512d b = _mm512_loadu_pd(col_b);

        __m512d an = _mm512_add_pd(_mm512_mul_pd(vc, a), _mm512_mul_pd(vs, b));
        __m512d bn = _mm512_sub_pd(_mm512_mul_pd(vc, b), _mm512_mul_pd(vs, a));

        _mm512_storeu_pd(col_a, an);
        _mm512_storeu_pd(col_b, bn);
    }
#endif

    for (size_t i = 0; i < n; ++i) {
        double a = col_a[i];
        double b = col_b[i];
        col_a[i] = c * a + s * b;
        col_b[i] = c * b - s * a;
    }
}
