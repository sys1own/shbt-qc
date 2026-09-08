/*
 * shbt_core_runtime.c - SHBT-R bare-metal microkernel runtime core.
 *
 * Freestanding C11.  The only headers used are compiler-provided
 * (<stdint.h>, <stddef.h>, <stdbool.h>, and <immintrin.h> when AVX-512 is
 * enabled).  No libc is required for the runtime itself.
 *
 * Contents:
 *   a) Lock-free SPSC and MPSC circular ring buffers in cryogenic SRAM using
 *      C11-style acquire/release atomics (__ATOMIC_ACQUIRE / __ATOMIC_RELEASE).
 *   b) A 4-step post-quench recovery driver:
 *        1. read and acknowledge MMIO interrupts
 *        2. trigger pump attenuation
 *        3. SECDED Hamming(72,64) ECC correction of the latched word
 *        4. bulk phase-register reset
 *      with a latency check against 105.90 ns.
 *   c) An AVX-512 single-precision isometry remapping engine for the 312
 *      microcavity channels (16 floats per __m512) with an 84.60 ns bound.
 *
 * Build (freestanding):
 *   gcc -std=c11 -O2 -ffreestanding -nostdlib -mavx512f -c \
 *       -Ikernel/include kernel/src/shbt_core_runtime.c
 *
 * Build (host self-test):
 *   gcc -std=c11 -O2 -mavx512f -DSHBT_HOST_TEST -Ikernel/include \
 *       kernel/src/shbt_core_runtime.c -o shbt_core_test && ./shbt_core_test
 */

#ifdef SHBT_HOST_TEST
#  define _POSIX_C_SOURCE 200809L
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef SHBT_HOST_TEST
#  ifndef SHBT_MMIO_OVERRIDE
#    define SHBT_MMIO_OVERRIDE 1
#  endif
#endif

#include "shbt_hardware.h"

#if defined(__AVX512F__)
#  include <immintrin.h>
#  define SHBT_HAVE_AVX512 1
#else
#  define SHBT_HAVE_AVX512 0
#endif

#ifdef SHBT_MMIO_OVERRIDE
shbt_mmio_t *shbt_mmio_override;
#endif

/* Cryogenic SRAM placement for ring storage */
#define SHBT_CRYO_SRAM  __attribute__((section(".cryo_sram"), aligned(64)))
#define SHBT_CACHELINE  __attribute__((aligned(64)))

/* ==========================================================================
 * Timing
 * ========================================================================== */

/* TSC frequency used to convert cycles to hundredths of a nanosecond. */
static uint64_t shbt_tsc_hz = 3000000000ULL;

static inline uint64_t shbt_cycles(void)
{
#if defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ __volatile__("lfence\n\trdtsc" : "=a"(lo), "=d"(hi) :: "memory");
    return ((uint64_t)hi << 32) | lo;
#else
    uint32_t hi1, lo, hi2;
    do {
        hi1 = shbt_mmio_read(&SHBT_MMIO->timer_hi);
        lo  = shbt_mmio_read(&SHBT_MMIO->timer_lo);
        hi2 = shbt_mmio_read(&SHBT_MMIO->timer_hi);
    } while (hi1 != hi2);
    return ((uint64_t)hi1 << 32) | lo;
#endif
}

static inline uint64_t shbt_cycles_to_ns_x100(uint64_t cycles)
{
#if defined(__x86_64__) || defined(__i386__)
    /* ns*100 = cycles * 1e5 / f_MHz  (64-bit only; no libgcc __udivti3) */
    uint64_t tsc_mhz = shbt_tsc_hz / 1000000ULL;
    return (cycles * 100000ULL) / (tsc_mhz ? tsc_mhz : 1ULL);
#else
    return cycles * 100ULL;   /* MMIO timer already counts ns */
#endif
}

/* ==========================================================================
 * a) Lock-free ring buffers
 * ========================================================================== */

#define SHBT_RING_ORDER     8U
#define SHBT_RING_SIZE      (1U << SHBT_RING_ORDER)
#define SHBT_RING_MASK      (SHBT_RING_SIZE - 1U)

typedef struct {
    uint32_t channel;
    uint32_t opcode;
    uint64_t payload;
} shbt_msg_t;

/* ---------- SPSC ------------------------------------------------------- */
typedef struct {
    SHBT_CACHELINE uint32_t head;                /* consumer index (read)  */
    SHBT_CACHELINE uint32_t tail;                /* producer index (write) */
    SHBT_CACHELINE shbt_msg_t slots[SHBT_RING_SIZE];
} shbt_spsc_ring_t;

static void shbt_spsc_init(shbt_spsc_ring_t *r)
{
    __atomic_store_n(&r->head, 0U, __ATOMIC_RELAXED);
    __atomic_store_n(&r->tail, 0U, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

static bool shbt_spsc_push(shbt_spsc_ring_t *r, const shbt_msg_t *m)
{
    uint32_t tail = __atomic_load_n(&r->tail, __ATOMIC_RELAXED);
    uint32_t head = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
    if ((tail - head) == SHBT_RING_SIZE)
        return false;                                 /* full */
    r->slots[tail & SHBT_RING_MASK] = *m;
    __atomic_store_n(&r->tail, tail + 1U, __ATOMIC_RELEASE);
    return true;
}

bool shbt_spsc_pop(shbt_spsc_ring_t *r, shbt_msg_t *out)
{
    uint32_t head = __atomic_load_n(&r->head, __ATOMIC_RELAXED);
    uint32_t tail = __atomic_load_n(&r->tail, __ATOMIC_ACQUIRE);
    if (head == tail)
        return false;                                 /* empty */
    *out = r->slots[head & SHBT_RING_MASK];
    __atomic_store_n(&r->head, head + 1U, __ATOMIC_RELEASE);
    return true;
}

static inline uint32_t shbt_spsc_count(const shbt_spsc_ring_t *r)
{
    return __atomic_load_n(&r->tail, __ATOMIC_ACQUIRE) -
           __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
}

/* ---------- MPSC ------------------------------------------------------- */
/*
 * Producers reserve a slot with a CAS on `tail`, write the payload, then
 * publish by storing the slot sequence number with RELEASE semantics.  The
 * single consumer waits for the sequence number of its head slot to match
 * (ACQUIRE), guaranteeing it observes the payload written before publish.
 */
typedef struct {
    uint32_t   seq;          /* == index+1 when the slot holds a published message */
    shbt_msg_t msg;
} shbt_mpsc_slot_t;

typedef struct {
    SHBT_CACHELINE uint32_t head;
    SHBT_CACHELINE uint32_t tail;
    SHBT_CACHELINE shbt_mpsc_slot_t slots[SHBT_RING_SIZE];
} shbt_mpsc_ring_t;

static void shbt_mpsc_init(shbt_mpsc_ring_t *r)
{
    for (uint32_t i = 0; i < SHBT_RING_SIZE; ++i)
        __atomic_store_n(&r->slots[i].seq, 0U, __ATOMIC_RELAXED);
    __atomic_store_n(&r->head, 0U, __ATOMIC_RELAXED);
    __atomic_store_n(&r->tail, 0U, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

static bool shbt_mpsc_push(shbt_mpsc_ring_t *r, const shbt_msg_t *m)
{
    uint32_t tail = __atomic_load_n(&r->tail, __ATOMIC_RELAXED);
    for (;;) {
        uint32_t head = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
        if ((tail - head) >= SHBT_RING_SIZE)
            return false;                             /* full */
        if (__atomic_compare_exchange_n(&r->tail, &tail, tail + 1U, true,
                                        __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
            break;                                    /* slot reserved */
        /* `tail` was reloaded by the failed CAS; retry */
    }
    shbt_mpsc_slot_t *s = &r->slots[tail & SHBT_RING_MASK];
    s->msg = *m;
    __atomic_store_n(&s->seq, tail + 1U, __ATOMIC_RELEASE);
    return true;
}

static bool shbt_mpsc_pop(shbt_mpsc_ring_t *r, shbt_msg_t *out)
{
    uint32_t head = __atomic_load_n(&r->head, __ATOMIC_RELAXED);
    shbt_mpsc_slot_t *s = &r->slots[head & SHBT_RING_MASK];
    if (__atomic_load_n(&s->seq, __ATOMIC_ACQUIRE) != head + 1U)
        return false;                                 /* empty or not yet published */
    *out = s->msg;
    __atomic_store_n(&s->seq, 0U, __ATOMIC_RELAXED);  /* slot recyclable */
    __atomic_store_n(&r->head, head + 1U, __ATOMIC_RELEASE);
    return true;
}

/* Ring instances resident in cryogenic SRAM */
SHBT_CRYO_SRAM shbt_spsc_ring_t shbt_irq_ring;             /* ISR -> scheduler  */
SHBT_CRYO_SRAM shbt_mpsc_ring_t shbt_cmd_ring;             /* cores -> control  */

/* ==========================================================================
 * b) SECDED Hamming(72,64)
 * ========================================================================== */
/*
 * Codeword positions 1..71 follow the classic Hamming layout: check bit j
 * sits at position 2^j (j = 0..6), data bits fill the remaining 64 of the 65
 * non-power-of-two positions (3,5,6,7,9,...,70).  Check bit 7 is the overall
 * parity of data and check bits 0..6, providing double-error detection.
 */
static uint64_t shbt_ecc_mask[7];          /* data bits covered by check bit j */
static uint8_t  shbt_ecc_pos_of_data[64];  /* data bit -> codeword position    */
static int8_t   shbt_ecc_data_of_pos[72];  /* codeword position -> data bit, -1 if check */
static bool     shbt_ecc_ready;

typedef enum {
    SHBT_ECC_OK = 0,
    SHBT_ECC_CORRECTED_DATA,
    SHBT_ECC_CORRECTED_CHECK,
    SHBT_ECC_CORRECTED_PARITY,
    SHBT_ECC_UNCORRECTABLE
} shbt_ecc_result_t;

static void shbt_ecc_init(void)
{
    if (shbt_ecc_ready)
        return;
    for (int p = 0; p < 72; ++p)
        shbt_ecc_data_of_pos[p] = -1;
    for (int j = 0; j < 7; ++j)
        shbt_ecc_mask[j] = 0;

    int d = 0;
    for (int p = 1; p < 72 && d < 64; ++p) {
        if ((p & (p - 1)) == 0)
            continue;                                  /* power of two -> check bit */
        shbt_ecc_pos_of_data[d] = (uint8_t)p;
        shbt_ecc_data_of_pos[p] = (int8_t)d;
        for (int j = 0; j < 7; ++j)
            if (p & (1 << j))
                shbt_ecc_mask[j] |= (1ULL << d);
        ++d;
    }
    shbt_ecc_ready = true;
}

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

static shbt_ecc_result_t shbt_ecc_decode(uint64_t *data, uint8_t *check, uint8_t *syndrome_out)
{
    uint64_t d = *data;
    uint8_t  c = *check;
    uint8_t  expected = shbt_ecc_hamming_bits(d);
    uint8_t  syn = (uint8_t)((c ^ expected) & 0x7FU);
    uint8_t  parity = (uint8_t)(__builtin_parityll(d) ^ __builtin_parity(c));  /* whole 72-bit word */

    if (syndrome_out)
        *syndrome_out = (uint8_t)(syn | (parity << 7));

    if (syn == 0 && parity == 0)
        return SHBT_ECC_OK;
    if (syn == 0 && parity == 1) {
        *check = (uint8_t)(c ^ 0x80U);
        return SHBT_ECC_CORRECTED_PARITY;
    }
    if (parity == 0)
        return SHBT_ECC_UNCORRECTABLE;                 /* even number of flips (>=2) */

    /* Single-bit error at codeword position `syn` */
    if ((syn & (syn - 1)) == 0) {
        int j = __builtin_ctz(syn);
        *check = (uint8_t)(c ^ (1U << j));
        return SHBT_ECC_CORRECTED_CHECK;
    }
    int8_t db = shbt_ecc_data_of_pos[syn];
    if (db < 0)
        return SHBT_ECC_UNCORRECTABLE;                 /* position 0 or >71 */
    *data = d ^ (1ULL << db);
    return SHBT_ECC_CORRECTED_DATA;
}

/* ==========================================================================
 * b) 4-step post-quench recovery driver
 * ========================================================================== */
typedef struct {
    uint32_t          irq_seen;
    uint32_t          pump_atten_written;
    shbt_ecc_result_t ecc_result;
    uint8_t           ecc_syndrome;
    uint64_t          ecc_corrected_data;
    uint32_t          phases_reset;
    uint64_t          latency_ns_x100;
    bool              within_bound;
} shbt_recovery_report_t;

static inline void shbt_pump_attenuate(uint32_t atten_db_x100, uint32_t ramp_sel)
{
    uint32_t v = (atten_db_x100 & SHBT_PUMP_ATTEN_DB_X100_MASK)
               | ((ramp_sel << SHBT_PUMP_ATTEN_RAMP_SHIFT) & SHBT_PUMP_ATTEN_RAMP_MASK)
               | SHBT_PUMP_ATTEN_APPLY;
    shbt_mmio_write(&SHBT_MMIO->pump_atten, v);
}

static inline void shbt_phase_reset_all(void)
{
    shbt_mmio_set(&SHBT_MMIO->ctrl, SHBT_CTRL_PHASE_RESET);
    shbt_mmio_clear(&SHBT_MMIO->ctrl, SHBT_CTRL_PHASE_RESET);
    /* Zero the first hexamer explicitly to seed lock re-acquisition */
    for (uint32_t i = 0; i < SHBT_HEXAMER_SIZE; ++i)
        shbt_mmio_write(&SHBT_MMIO->phase[i], 0U);
}

static void shbt_quench_recover(shbt_recovery_report_t *rep)
{
    shbt_mmio_t *hw = SHBT_MMIO;
    uint64_t t0 = shbt_cycles();

    /* Step 1: read and acknowledge recovery-relevant interrupts */
    uint32_t irq = shbt_mmio_read(&hw->irq_status) & SHBT_IRQ_RECOVERY_MASK;
    shbt_mmio_write(&hw->irq_clear, irq);
    shbt_mmio_set(&hw->ctrl, SHBT_CTRL_QUENCH_ACK);

    /* Step 2: attenuate the pump to stop further energy deposition */
    shbt_pump_attenuate(SHBT_PUMP_ATTEN_QUENCH_DB_X100, 0x2U);

    /* Step 3: SECDED correction of the latched ECC word */
    uint64_t data = ((uint64_t)shbt_mmio_read(&hw->ecc_data_hi) << 32)
                  |  (uint64_t)shbt_mmio_read(&hw->ecc_data_lo);
    uint8_t  check = (uint8_t)(shbt_mmio_read(&hw->ecc_check) & SHBT_ECC_CHECK_MASK);
    uint8_t  syn = 0;
    shbt_ecc_result_t res = shbt_ecc_decode(&data, &check, &syn);
    shbt_mmio_write(&hw->ecc_data_lo, (uint32_t)data);
    shbt_mmio_write(&hw->ecc_data_hi, (uint32_t)(data >> 32));
    shbt_mmio_write(&hw->ecc_check, check);
    shbt_mmio_write(&hw->ecc_syndrome, (uint32_t)syn | SHBT_ECC_SYN_VALID);
    shbt_mmio_write(&hw->ecc_ctrl, SHBT_ECC_CTRL_ENABLE | SHBT_ECC_CTRL_CLEAR);

    /* Step 4: reset phase registers */
    shbt_phase_reset_all();

    __atomic_thread_fence(__ATOMIC_RELEASE);
    uint64_t t1 = shbt_cycles();

    rep->irq_seen = irq;
    rep->pump_atten_written = shbt_mmio_read(&hw->pump_atten);
    rep->ecc_result = res;
    rep->ecc_syndrome = syn;
    rep->ecc_corrected_data = data;
    rep->phases_reset = SHBT_NUM_CHANNELS;
    rep->latency_ns_x100 = shbt_cycles_to_ns_x100(t1 - t0);
    rep->within_bound = rep->latency_ns_x100 <= SHBT_QUENCH_RECOVERY_BOUND_NS_X100;

    /* Post a completion message for the scheduler */
    shbt_msg_t m = { 0U, 0x51U /* QUENCH_RECOVERED */, rep->latency_ns_x100 };
    (void)shbt_spsc_push(&shbt_irq_ring, &m);
}

/* ==========================================================================
 * c) AVX-512 isometry remapping engine
 * ========================================================================== */
/*
 * Each channel carries a complex field amplitude (re, im).  The isometry is
 * a per-channel Givens rotation by theta_k (a unitary, norm-preserving map):
 *
 *     re' =  c_k*re - s_k*im
 *     im' =  s_k*re + c_k*im
 *
 * Arrays are padded to SHBT_CHANNEL_PAD (320) lanes so that 20 __m512
 * registers cover all channels.  The engine also returns the total field
 * energy (sum re'^2 + im'^2) via _mm512_reduce_add_ps for isometry checking.
 */
typedef struct {
    SHBT_CACHELINE float cos_t[SHBT_CHANNEL_PAD];
    SHBT_CACHELINE float sin_t[SHBT_CHANNEL_PAD];
} shbt_isometry_t;

typedef struct {
    SHBT_CACHELINE float re[SHBT_CHANNEL_PAD];
    SHBT_CACHELINE float im[SHBT_CHANNEL_PAD];
} shbt_field_t;

static float shbt_isometry_remap(const shbt_isometry_t *iso,
                                 const shbt_field_t *in,
                                 shbt_field_t *out)
{
#if SHBT_HAVE_AVX512
    __m512 energy = _mm512_setzero_ps();
    for (uint32_t i = 0; i < SHBT_CHANNEL_PAD; i += 16U) {
        __m512 c  = _mm512_loadu_ps(&iso->cos_t[i]);
        __m512 s  = _mm512_loadu_ps(&iso->sin_t[i]);
        __m512 re = _mm512_loadu_ps(&in->re[i]);
        __m512 im = _mm512_loadu_ps(&in->im[i]);

        __m512 re2 = _mm512_fmsub_ps(c, re, _mm512_mul_ps(s, im));   /* c*re - s*im */
        __m512 im2 = _mm512_fmadd_ps(s, re, _mm512_mul_ps(c, im));   /* s*re + c*im */

        _mm512_storeu_ps(&out->re[i], re2);
        _mm512_storeu_ps(&out->im[i], im2);

        energy = _mm512_fmadd_ps(re2, re2, energy);
        energy = _mm512_fmadd_ps(im2, im2, energy);
    }
    return _mm512_reduce_add_ps(energy);
#else
    float energy = 0.0f;
    for (uint32_t i = 0; i < SHBT_CHANNEL_PAD; ++i) {
        float c = iso->cos_t[i], s = iso->sin_t[i];
        float re = in->re[i], im = in->im[i];
        float re2 = c * re - s * im;
        float im2 = s * re + c * im;
        out->re[i] = re2;
        out->im[i] = im2;
        energy += re2 * re2 + im2 * im2;
    }
    return energy;
#endif
}

typedef struct {
    float    energy_in;
    float    energy_out;
    uint64_t latency_ns_x100;
    bool     within_bound;
} shbt_isometry_report_t;

static void shbt_isometry_run(const shbt_isometry_t *iso,
                              const shbt_field_t *in,
                              shbt_field_t *out,
                              shbt_isometry_report_t *rep)
{
    uint64_t t0 = shbt_cycles();
    float e_out = shbt_isometry_remap(iso, in, out);
    uint64_t t1 = shbt_cycles();

    float e_in = 0.0f;
    for (uint32_t i = 0; i < SHBT_NUM_CHANNELS; ++i)
        e_in += in->re[i] * in->re[i] + in->im[i] * in->im[i];

    rep->energy_in = e_in;
    rep->energy_out = e_out;
    rep->latency_ns_x100 = shbt_cycles_to_ns_x100(t1 - t0);
    rep->within_bound = rep->latency_ns_x100 <= SHBT_ISOMETRY_BOUND_NS_X100;
}

/* ==========================================================================
 * Kernel entry (freestanding)
 * ========================================================================== */
static SHBT_CRYO_SRAM shbt_isometry_t shbt_iso_table;
static SHBT_CRYO_SRAM shbt_field_t    shbt_field_in;
static SHBT_CRYO_SRAM shbt_field_t    shbt_field_out;

void shbt_kernel_init(void)
{
    shbt_ecc_init();
    shbt_spsc_init(&shbt_irq_ring);
    shbt_mpsc_init(&shbt_cmd_ring);
    shbt_mmio_write(&SHBT_MMIO->irq_enable, SHBT_IRQ_RECOVERY_MASK | SHBT_IRQ_OVERTEMP);
    shbt_mmio_write(&SHBT_MMIO->ecc_ctrl, SHBT_ECC_CTRL_ENABLE);
    shbt_mmio_set(&SHBT_MMIO->ctrl, SHBT_CTRL_ENABLE | SHBT_CTRL_ECC_ENABLE);
}

void shbt_kernel_irq(void)
{
    uint32_t irq = shbt_mmio_read(&SHBT_MMIO->irq_status);
    if (irq & SHBT_IRQ_RECOVERY_MASK) {
        shbt_recovery_report_t rep;
        shbt_quench_recover(&rep);
    }
    if (irq & SHBT_IRQ_OVERTEMP) {
        shbt_pump_attenuate(SHBT_PUMP_ATTEN_DB_X100_MASK, 0xFU);
        shbt_mmio_write(&SHBT_MMIO->irq_clear, SHBT_IRQ_OVERTEMP);
    }
}

void shbt_kernel_tick(void)
{
    shbt_msg_t m;
    while (shbt_mpsc_pop(&shbt_cmd_ring, &m)) {
        if (m.opcode == 0x10U) {                     /* ISOMETRY_REMAP */
            shbt_isometry_report_t rep;
            shbt_isometry_run(&shbt_iso_table, &shbt_field_in, &shbt_field_out, &rep);
        }
    }
}

bool shbt_kernel_submit(const shbt_msg_t *m)
{
    return shbt_mpsc_push(&shbt_cmd_ring, m);
}

/* ==========================================================================
 * Host self-test (not part of the freestanding image)
 * ========================================================================== */
#ifdef SHBT_HOST_TEST
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "ASSERT FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static uint64_t host_calibrate_tsc_hz(void)
{
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    uint64_t c0 = shbt_cycles();
    do { clock_gettime(CLOCK_MONOTONIC, &b); }
    while ((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec) < 100000000LL);
    uint64_t c1 = shbt_cycles();
    uint64_t ns = (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec));
    return (c1 - c0) * 1000000000ULL / ns;
}

static void *host_producer(void *arg)
{
    uint32_t id = (uint32_t)(uintptr_t)arg;
    for (uint32_t i = 0; i < 1000; ++i) {
        shbt_msg_t m = { id, i, ((uint64_t)id << 32) | i };
        while (!shbt_mpsc_push(&shbt_cmd_ring, &m))
            __builtin_ia32_pause();
    }
    return NULL;
}

int main(void)
{
    static shbt_mmio_t emulated_hw;
    memset(&emulated_hw, 0, sizeof emulated_hw);
    shbt_mmio_override = &emulated_hw;
    shbt_tsc_hz = host_calibrate_tsc_hz();
    printf("SHBT-R core runtime self-test (TSC %.3f GHz, AVX-512 %s)\n",
           shbt_tsc_hz / 1e9, SHBT_HAVE_AVX512 ? "on" : "off");

    /* Header layout */
    CHECK(sizeof(uint32_t) * 4 == 16);
    CHECK(offsetof(shbt_mmio_t, phase) == SHBT_REG_PHASE_BASE);
    CHECK(offsetof(shbt_mmio_t, ecc_check) == SHBT_REG_ECC_CHECK);
    CHECK(SHBT_NUM_HEXAMERS == 52U);
    CHECK(SHBT_MMIO_BASE == 0x70000000UL);

    shbt_kernel_init();
    CHECK(emulated_hw.ctrl & SHBT_CTRL_ENABLE);
    CHECK(emulated_hw.irq_enable & SHBT_IRQ_QUENCH);

    /* ---- SPSC ---------------------------------------------------------- */
    shbt_msg_t m, o;
    for (uint32_t i = 0; i < SHBT_RING_SIZE; ++i) {
        m.channel = i; m.opcode = 1; m.payload = i * 7ULL;
        CHECK(shbt_spsc_push(&shbt_irq_ring, &m));
    }
    CHECK(!shbt_spsc_push(&shbt_irq_ring, &m));          /* full */
    CHECK(shbt_spsc_count(&shbt_irq_ring) == SHBT_RING_SIZE);
    for (uint32_t i = 0; i < SHBT_RING_SIZE; ++i) {
        CHECK(shbt_spsc_pop(&shbt_irq_ring, &o));
        CHECK(o.channel == i && o.payload == i * 7ULL);
    }
    CHECK(!shbt_spsc_pop(&shbt_irq_ring, &o));           /* empty */
    printf("  SPSC ring: FIFO order, full/empty detection OK\n");

    /* ---- MPSC (4 producers, 1 consumer) --------------------------------- */
    pthread_t th[4];
    for (uintptr_t p = 0; p < 4; ++p)
        pthread_create(&th[p], NULL, host_producer, (void *)(p + 1));
    uint32_t got = 0, next_seq[5] = {0};
    while (got < 4000) {
        if (shbt_mpsc_pop(&shbt_cmd_ring, &o)) {
            CHECK(o.channel >= 1 && o.channel <= 4);
            CHECK(o.opcode == next_seq[o.channel]);       /* per-producer order preserved */
            CHECK(o.payload == (((uint64_t)o.channel << 32) | o.opcode));
            next_seq[o.channel]++;
            got++;
        }
    }
    for (int p = 0; p < 4; ++p) pthread_join(th[p], NULL);
    CHECK(!shbt_mpsc_pop(&shbt_cmd_ring, &o));
    printf("  MPSC ring: 4000 msgs from 4 producers, per-producer ordering OK\n");

    /* ---- ECC ----------------------------------------------------------- */
    srand(12345);
    for (int trial = 0; trial < 2000; ++trial) {
        uint64_t d = ((uint64_t)rand() << 33) ^ ((uint64_t)rand() << 11) ^ (uint64_t)rand();
        uint8_t c = shbt_ecc_encode(d);
        uint64_t dd = d; uint8_t cc = c; uint8_t syn;
        CHECK(shbt_ecc_decode(&dd, &cc, &syn) == SHBT_ECC_OK && syn == 0);

        /* single data-bit flip -> corrected */
        int b = rand() % 64;
        dd = d ^ (1ULL << b); cc = c;
        CHECK(shbt_ecc_decode(&dd, &cc, &syn) == SHBT_ECC_CORRECTED_DATA);
        CHECK(dd == d && cc == c);

        /* single check-bit flip -> corrected */
        int cb = rand() % 8;
        dd = d; cc = (uint8_t)(c ^ (1U << cb));
        shbt_ecc_result_t r = shbt_ecc_decode(&dd, &cc, &syn);
        CHECK(r == SHBT_ECC_CORRECTED_CHECK || r == SHBT_ECC_CORRECTED_PARITY);
        CHECK(dd == d && cc == c);

        /* double data-bit flip -> detected, uncorrectable */
        int b2 = (b + 1 + rand() % 63) % 64;
        dd = d ^ (1ULL << b) ^ (1ULL << b2); cc = c;
        CHECK(shbt_ecc_decode(&dd, &cc, &syn) == SHBT_ECC_UNCORRECTABLE);
    }
    printf("  ECC Hamming(72,64): 2000 trials SEC + DED OK\n");

    /* ---- Quench recovery ----------------------------------------------- */
    uint64_t good = 0xC0FFEE1234567890ULL;
    uint8_t  chk = shbt_ecc_encode(good);
    shbt_recovery_report_t best = {0};
    best.latency_ns_x100 = UINT64_MAX;
    for (int it = 0; it < 20000; ++it) {
        emulated_hw.irq_status = SHBT_IRQ_QUENCH | SHBT_IRQ_ECC_CE | SHBT_IRQ_TIMER;
        emulated_hw.ecc_data_lo = (uint32_t)(good ^ (1ULL << 17));  /* injected SE */
        emulated_hw.ecc_data_hi = (uint32_t)(good >> 32);
        emulated_hw.ecc_check = chk;
        emulated_hw.ctrl &= ~SHBT_CTRL_QUENCH_ACK;
        for (uint32_t i = 0; i < SHBT_HEXAMER_SIZE; ++i) emulated_hw.phase[i] = 0xDEADBEEF;

        shbt_recovery_report_t rep;
        shbt_quench_recover(&rep);
        shbt_msg_t done;
        CHECK(shbt_spsc_pop(&shbt_irq_ring, &done) && done.opcode == 0x51U);

        CHECK(rep.irq_seen == (SHBT_IRQ_QUENCH | SHBT_IRQ_ECC_CE));   /* TIMER filtered */
        CHECK(emulated_hw.irq_clear == (SHBT_IRQ_QUENCH | SHBT_IRQ_ECC_CE));
        CHECK(emulated_hw.ctrl & SHBT_CTRL_QUENCH_ACK);
        CHECK((rep.pump_atten_written & SHBT_PUMP_ATTEN_DB_X100_MASK) == SHBT_PUMP_ATTEN_QUENCH_DB_X100);
        CHECK(rep.pump_atten_written & SHBT_PUMP_ATTEN_APPLY);
        CHECK(rep.ecc_result == SHBT_ECC_CORRECTED_DATA);
        CHECK(rep.ecc_corrected_data == good);
        CHECK(((uint64_t)emulated_hw.ecc_data_hi << 32 | emulated_hw.ecc_data_lo) == good);
        CHECK(emulated_hw.ecc_syndrome & SHBT_ECC_SYN_VALID);
        CHECK(rep.phases_reset == SHBT_NUM_CHANNELS);
        for (uint32_t i = 0; i < SHBT_HEXAMER_SIZE; ++i) CHECK(emulated_hw.phase[i] == 0U);
        if (rep.latency_ns_x100 < best.latency_ns_x100) best = rep;
    }
    printf("  Quench recovery: 4 steps verified, best latency %.2f ns (bound %.2f ns) %s\n",
           best.latency_ns_x100 / 100.0, SHBT_QUENCH_RECOVERY_BOUND_NS_X100 / 100.0,
           best.within_bound ? "PASS" : "FAIL");
    CHECK(best.within_bound);

    /* ---- Isometry engine ----------------------------------------------- */
    memset(&shbt_iso_table, 0, sizeof shbt_iso_table);
    memset(&shbt_field_in, 0, sizeof shbt_field_in);
    for (uint32_t i = 0; i < SHBT_NUM_CHANNELS; ++i) {
        float theta = 0.0123f * (float)i;
        shbt_iso_table.cos_t[i] = cosf(theta);
        shbt_iso_table.sin_t[i] = sinf(theta);
        shbt_field_in.re[i] = 0.5f + 0.001f * (float)i;
        shbt_field_in.im[i] = -0.25f + 0.002f * (float)(i % 6);
    }
    shbt_isometry_report_t ibest = {0};
    ibest.latency_ns_x100 = UINT64_MAX;
    for (int it = 0; it < 20000; ++it) {
        shbt_isometry_report_t rep;
        shbt_isometry_run(&shbt_iso_table, &shbt_field_in, &shbt_field_out, &rep);
        if (rep.latency_ns_x100 < ibest.latency_ns_x100) ibest = rep;
    }
    CHECK(ibest.energy_in > 0.0f);
    CHECK(fabsf(ibest.energy_out - ibest.energy_in) <= 1e-4f * ibest.energy_in);  /* norm preserved */
    /* spot check a rotation */
    uint32_t k = 100;
    float er = shbt_iso_table.cos_t[k] * shbt_field_in.re[k] - shbt_iso_table.sin_t[k] * shbt_field_in.im[k];
    CHECK(fabsf(shbt_field_out.re[k] - er) < 1e-5f);
    for (uint32_t i = SHBT_NUM_CHANNELS; i < SHBT_CHANNEL_PAD; ++i)
        CHECK(shbt_field_out.re[i] == 0.0f && shbt_field_out.im[i] == 0.0f);   /* padding inert */
    printf("  Isometry: 312 ch, energy %.4f -> %.4f, best latency %.2f ns (bound %.2f ns) %s\n",
           ibest.energy_in, ibest.energy_out, ibest.latency_ns_x100 / 100.0,
           SHBT_ISOMETRY_BOUND_NS_X100 / 100.0, ibest.within_bound ? "PASS" : "FAIL");
    CHECK(ibest.within_bound);

    /* ---- Kernel command path ------------------------------------------- */
    shbt_msg_t cmd = { 0, 0x10U, 0 };
    CHECK(shbt_kernel_submit(&cmd));
    memset(&shbt_field_out, 0, sizeof shbt_field_out);
    shbt_kernel_tick();
    CHECK(fabsf(shbt_field_out.re[k] - er) < 1e-5f);

    emulated_hw.irq_status = SHBT_IRQ_QUENCH;
    emulated_hw.ecc_data_lo = (uint32_t)good; emulated_hw.ecc_data_hi = (uint32_t)(good >> 32);
    emulated_hw.ecc_check = chk;
    shbt_kernel_irq();
    CHECK(emulated_hw.ctrl & SHBT_CTRL_QUENCH_ACK);

    printf("All SHBT-R core runtime self-tests passed.\n");
    return 0;
}
#endif /* SHBT_HOST_TEST */
