/*
 * shbt_hardware.h - SHBT-R Quantum Computer memory-mapped hardware definitions.
 *
 * Freestanding: depends only on compiler-provided <stdint.h>.
 *
 * All control, status, interrupt, and ECC registers live in a single MMIO
 * aperture at SHBT_MMIO_BASE (0x70000000).  Register access goes through the
 * volatile `shbt_mmio_t` structure so the compiler never caches or reorders
 * device reads/writes.
 */
#ifndef SHBT_HARDWARE_H
#define SHBT_HARDWARE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Aperture
 * -------------------------------------------------------------------------- */
#define SHBT_MMIO_BASE              0x70000000UL
#define SHBT_MMIO_SIZE              0x00010000UL

#define SHBT_NUM_CHANNELS           312U
#define SHBT_HEXAMER_SIZE           6U
#define SHBT_NUM_HEXAMERS           (SHBT_NUM_CHANNELS / SHBT_HEXAMER_SIZE)   /* 52 */
#define SHBT_CHANNEL_PAD            320U   /* 312 rounded up to 16-lane multiple */

/* Timing budgets (nanoseconds, fixed point with 2 decimals -> hundredths) */
#define SHBT_QUENCH_RECOVERY_BOUND_NS_X100   10590U   /* 105.90 ns */
#define SHBT_ISOMETRY_BOUND_NS_X100          8460U    /*  84.60 ns */

/* --------------------------------------------------------------------------
 * Register byte offsets (relative to SHBT_MMIO_BASE)
 * -------------------------------------------------------------------------- */
#define SHBT_REG_CTRL               0x0000U
#define SHBT_REG_STATUS             0x0004U
#define SHBT_REG_IRQ_STATUS         0x0008U
#define SHBT_REG_IRQ_ENABLE         0x000CU
#define SHBT_REG_IRQ_CLEAR          0x0010U
#define SHBT_REG_PUMP_ATTEN         0x0014U
#define SHBT_REG_ECC_CTRL           0x0018U
#define SHBT_REG_ECC_STATUS         0x001CU
#define SHBT_REG_ECC_SYNDROME       0x0020U
#define SHBT_REG_ECC_ADDR           0x0024U
#define SHBT_REG_ECC_DATA_LO        0x0028U
#define SHBT_REG_ECC_DATA_HI        0x002CU
#define SHBT_REG_ECC_CHECK          0x0030U
#define SHBT_REG_TIMER_LO           0x0034U
#define SHBT_REG_TIMER_HI           0x0038U
#define SHBT_REG_TEMP_MK            0x003CU
#define SHBT_REG_PHASE_BASE         0x1000U   /* 312 x uint32 phase registers */

/* --------------------------------------------------------------------------
 * CTRL register bits
 * -------------------------------------------------------------------------- */
#define SHBT_CTRL_ENABLE            (1U << 0)
#define SHBT_CTRL_SOFT_RESET        (1U << 1)
#define SHBT_CTRL_PUMP_ENABLE       (1U << 2)
#define SHBT_CTRL_PHASE_LATCH       (1U << 3)
#define SHBT_CTRL_PHASE_RESET       (1U << 4)
#define SHBT_CTRL_ECC_ENABLE        (1U << 5)
#define SHBT_CTRL_QUENCH_ACK        (1U << 6)

/* --------------------------------------------------------------------------
 * STATUS register bits / masks
 * -------------------------------------------------------------------------- */
#define SHBT_STATUS_READY           (1U << 0)
#define SHBT_STATUS_BUSY            (1U << 1)
#define SHBT_STATUS_QUENCH          (1U << 2)
#define SHBT_STATUS_PUMP_ON         (1U << 3)
#define SHBT_STATUS_PHASE_LOCKED    (1U << 4)
#define SHBT_STATUS_ECC_ERR         (1U << 5)
#define SHBT_STATUS_OVERTEMP        (1U << 6)
#define SHBT_STATUS_STATE_SHIFT     8U
#define SHBT_STATUS_STATE_MASK      (0xFU << SHBT_STATUS_STATE_SHIFT)
#define SHBT_STATE_IDLE             0x0U
#define SHBT_STATE_RUN              0x1U
#define SHBT_STATE_QUENCHED         0x2U
#define SHBT_STATE_RECOVERING       0x3U
#define SHBT_STATUS_FAULT_MASK      (SHBT_STATUS_QUENCH | SHBT_STATUS_ECC_ERR | SHBT_STATUS_OVERTEMP)

/* --------------------------------------------------------------------------
 * Interrupt register bits (STATUS / ENABLE / CLEAR share the same layout)
 * -------------------------------------------------------------------------- */
#define SHBT_IRQ_QUENCH             (1U << 0)
#define SHBT_IRQ_ECC_CE             (1U << 1)   /* correctable error   */
#define SHBT_IRQ_ECC_UE             (1U << 2)   /* uncorrectable error */
#define SHBT_IRQ_PHASE_SLIP         (1U << 3)
#define SHBT_IRQ_PUMP_FAULT         (1U << 4)
#define SHBT_IRQ_OVERTEMP           (1U << 5)
#define SHBT_IRQ_RING_FULL          (1U << 6)
#define SHBT_IRQ_TIMER              (1U << 7)
#define SHBT_IRQ_ALL                0xFFU
#define SHBT_IRQ_RECOVERY_MASK      (SHBT_IRQ_QUENCH | SHBT_IRQ_ECC_CE | SHBT_IRQ_ECC_UE | SHBT_IRQ_PHASE_SLIP)

/* --------------------------------------------------------------------------
 * PUMP_ATTEN register: attenuation in units of 0.01 dB (bits 15:0),
 * ramp-time selector (bits 19:16), APPLY strobe (bit 31)
 * -------------------------------------------------------------------------- */
#define SHBT_PUMP_ATTEN_DB_X100_MASK   0x0000FFFFU
#define SHBT_PUMP_ATTEN_RAMP_SHIFT     16U
#define SHBT_PUMP_ATTEN_RAMP_MASK      (0xFU << SHBT_PUMP_ATTEN_RAMP_SHIFT)
#define SHBT_PUMP_ATTEN_APPLY          (1U << 31)
#define SHBT_PUMP_ATTEN_QUENCH_DB_X100 3000U    /* 30.00 dB post-quench attenuation */

/* --------------------------------------------------------------------------
 * ECC control / status bitfields (SECDED Hamming(72,64))
 * -------------------------------------------------------------------------- */
#define SHBT_ECC_CTRL_ENABLE        (1U << 0)
#define SHBT_ECC_CTRL_SCRUB         (1U << 1)
#define SHBT_ECC_CTRL_INJECT_SE     (1U << 2)   /* test: inject single error */
#define SHBT_ECC_CTRL_INJECT_DE     (1U << 3)   /* test: inject double error */
#define SHBT_ECC_CTRL_CLEAR         (1U << 4)

#define SHBT_ECC_STATUS_CE          (1U << 0)   /* correctable error latched     */
#define SHBT_ECC_STATUS_UE          (1U << 1)   /* uncorrectable error latched   */
#define SHBT_ECC_STATUS_CE_CNT_SHIFT  8U
#define SHBT_ECC_STATUS_CE_CNT_MASK   (0xFFU << SHBT_ECC_STATUS_CE_CNT_SHIFT)
#define SHBT_ECC_STATUS_UE_CNT_SHIFT  16U
#define SHBT_ECC_STATUS_UE_CNT_MASK   (0xFFU << SHBT_ECC_STATUS_UE_CNT_SHIFT)

/* SYNDROME register: bits 6:0 Hamming syndrome, bit 7 overall parity, bit 8 valid */
#define SHBT_ECC_SYN_HAMMING_MASK   0x7FU
#define SHBT_ECC_SYN_PARITY         (1U << 7)
#define SHBT_ECC_SYN_VALID          (1U << 8)

/* CHECK register: 8 check bits (7 Hamming + 1 overall parity) in bits 7:0 */
#define SHBT_ECC_CHECK_MASK         0xFFU

/* --------------------------------------------------------------------------
 * Phase register: 24-bit fixed-point phase (2*pi == 1<<24), lock flag in bit 31
 * -------------------------------------------------------------------------- */
#define SHBT_PHASE_VALUE_MASK       0x00FFFFFFU
#define SHBT_PHASE_LOCK             (1U << 31)

/* --------------------------------------------------------------------------
 * ECC bitfield view of a 72-bit codeword (64 data + 8 check bits)
 * -------------------------------------------------------------------------- */
typedef struct {
    uint64_t data;          /* 64 data bits            */
    uint8_t  check;         /* 7 Hamming + 1 parity    */
} shbt_ecc_word_t;

typedef union {
    uint32_t raw;
    struct {
        uint32_t ce        : 1;
        uint32_t ue        : 1;
        uint32_t reserved0 : 6;
        uint32_t ce_count  : 8;
        uint32_t ue_count  : 8;
        uint32_t reserved1 : 8;
    } f;
} shbt_ecc_status_t;

typedef union {
    uint32_t raw;
    struct {
        uint32_t hamming   : 7;
        uint32_t parity    : 1;
        uint32_t valid     : 1;
        uint32_t reserved  : 23;
    } f;
} shbt_ecc_syndrome_t;

/* --------------------------------------------------------------------------
 * Volatile MMIO register block (must match the offsets above)
 * -------------------------------------------------------------------------- */
typedef struct {
    volatile uint32_t ctrl;             /* 0x0000 */
    volatile uint32_t status;           /* 0x0004 */
    volatile uint32_t irq_status;       /* 0x0008 */
    volatile uint32_t irq_enable;       /* 0x000C */
    volatile uint32_t irq_clear;        /* 0x0010 (W1C) */
    volatile uint32_t pump_atten;       /* 0x0014 */
    volatile uint32_t ecc_ctrl;         /* 0x0018 */
    volatile uint32_t ecc_status;       /* 0x001C */
    volatile uint32_t ecc_syndrome;     /* 0x0020 */
    volatile uint32_t ecc_addr;         /* 0x0024 */
    volatile uint32_t ecc_data_lo;      /* 0x0028 */
    volatile uint32_t ecc_data_hi;      /* 0x002C */
    volatile uint32_t ecc_check;        /* 0x0030 */
    volatile uint32_t timer_lo;         /* 0x0034 free-running ns counter */
    volatile uint32_t timer_hi;         /* 0x0038 */
    volatile uint32_t temp_mk;          /* 0x003C stage temperature in mK */
    volatile uint32_t reserved[(SHBT_REG_PHASE_BASE - 0x0040U) / 4U];
    volatile uint32_t phase[SHBT_NUM_CHANNELS];   /* 0x1000 .. */
} shbt_mmio_t;

/* Compile-time layout verification */
#define SHBT_STATIC_ASSERT(cond, name) typedef char shbt_static_assert_##name[(cond) ? 1 : -1]
SHBT_STATIC_ASSERT(__builtin_offsetof(shbt_mmio_t, ecc_ctrl)  == SHBT_REG_ECC_CTRL,   ecc_ctrl_off);
SHBT_STATIC_ASSERT(__builtin_offsetof(shbt_mmio_t, timer_lo)  == SHBT_REG_TIMER_LO,   timer_off);
SHBT_STATIC_ASSERT(__builtin_offsetof(shbt_mmio_t, phase)     == SHBT_REG_PHASE_BASE, phase_off);

/*
 * The runtime uses SHBT_MMIO to reach the device.  A host-side test build may
 * define SHBT_MMIO_OVERRIDE to point at emulated register memory.
 */
#ifdef SHBT_MMIO_OVERRIDE
extern shbt_mmio_t *shbt_mmio_override;
#define SHBT_MMIO   (shbt_mmio_override)
#else
#define SHBT_MMIO   ((shbt_mmio_t *)(uintptr_t)SHBT_MMIO_BASE)
#endif

static inline uint32_t shbt_mmio_read(volatile const uint32_t *reg)
{
    return *reg;
}

static inline void shbt_mmio_write(volatile uint32_t *reg, uint32_t value)
{
    *reg = value;
}

static inline void shbt_mmio_set(volatile uint32_t *reg, uint32_t bits)
{
    *reg = *reg | bits;
}

static inline void shbt_mmio_clear(volatile uint32_t *reg, uint32_t bits)
{
    *reg = *reg & ~bits;
}

#ifdef __cplusplus
}
#endif

#endif /* SHBT_HARDWARE_H */
