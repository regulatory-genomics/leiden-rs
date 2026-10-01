//! Random number generation, ported from igraph's `src/random/`.
//!
//! igraph's default generator is PCG32 (`src/random/rng_pcg32.c`, originally
//! from https://github.com/imneme/pcg-c, see `vendor/pcg/pcg_variants.h`).
//! All sampling helpers (`unif01`, `unif`, `integer`, `shuffle`) are ported
//! literally from `src/random/random.c` and `src/core/vector.pmt` so that the
//! random number stream consumed by the Leiden algorithm is identical to C
//! igraph given the same seed. This is what makes bit-for-bit differential
//! testing against the C implementation possible.
//!
//! Like igraph, generators only need to produce 32 random bits
//! ([`Rng::get`]); uniform reals are built by filling a double mantissa with
//! 52 random bits, and bounded integers use Lemire's debiased multiplication
//! method.

/// Trait mirroring igraph's `igraph_rng_type_t` + generic sampling layer.
pub trait Rng {
    /// Generate 32 random bits. Mirrors `igraph_i_rng_get_random_bits(rng, 32)`
    /// for a 32-bit generator (`type->get(state) >> 0`).
    fn get(&mut self) -> u32;

    /// Mirrors `igraph_i_rng_get_random_bits(rng, bits)` for `bits <= 32`.
    fn random_bits(&mut self, bits: u8) -> u32 {
        debug_assert!(bits <= 32);
        // rng_bitwidth (32) >= bits: keep the high bits.
        self.get() >> (32 - bits)
    }

    /// Mirrors `igraph_i_rng_get_random_bits_uint64(rng, bits)` for `bits <= 64`.
    fn random_bits_u64(&mut self, bits: u8) -> u64 {
        debug_assert!(bits <= 64);
        let mut result: u64 = 0;
        if bits <= 32 {
            return self.random_bits(bits) as u64;
        }
        // rng_bitwidth (32) < bits.
        let mut bits = bits;
        loop {
            result = (result << 32).wrapping_add(self.get() as u64);
            bits -= 32;
            if bits <= 32 {
                break;
            }
        }
        // Last piece.
        (result << bits).wrapping_add((self.get() >> (32 - bits)) as u64)
    }

    /// Mirrors `igraph_rng_get_unif01()`: uniform in [0, 1).
    ///
    /// The generic path extracts 52 random bits into the mantissa of a double
    /// between 1 (inclusive) and 2 (exclusive), then subtracts 1.
    fn unif01(&mut self) -> f64 {
        let value: u64 = (self.random_bits_u64(52) & 0xF_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000;
        f64::from_bits(value) - 1.0
    }

    /// Mirrors `igraph_rng_get_unif(rng, l, h)`: uniform in [l, h).
    ///
    /// Ensures that `h` is never produced due to numerical roundoff errors,
    /// except when `l == h`.
    fn unif(&mut self, l: f64, h: f64) -> f64 {
        debug_assert!(h >= l);
        if l == h {
            return h;
        }
        loop {
            let r = self.unif01() * (h - l) + l;
            if r != h {
                return r;
            }
        }
    }

    /// Mirrors `igraph_rng_get_uint_bounded()` for `range <= u32::MAX`:
    /// Lemire's debiased integer multiplication
    /// (https://www.pcg-random.org/posts/bounded-rands.html).
    fn uint_bounded(&mut self, range: u64) -> u64 {
        debug_assert!(range > 0);
        if range <= u32::MAX as u64 {
            // igraph_i_rng_get_uint32_bounded()
            let range = range as u32;
            let t = range.wrapping_neg() % range;
            loop {
                let x = self.get();
                let m = x as u64 * range as u64;
                let l = m as u32;
                if l >= t {
                    return m >> 32;
                }
            }
        } else {
            // igraph_i_rng_get_uint64_bounded()
            let t = range.wrapping_neg() % range;
            loop {
                let x = self.random_bits_u64(64);
                let m = (x as u128) * (range as u128);
                let l = m as u64;
                if l >= t {
                    return (m >> 64) as u64;
                }
            }
        }
    }

    /// Mirrors `igraph_rng_get_integer(rng, l, h)`: uniform integer in [l, h].
    fn integer(&mut self, l: i64, h: i64) -> i64 {
        debug_assert!(h >= l);
        if h == l {
            return l;
        }
        // In the Leiden code l is always 0 and h >= 0, so the range is
        // h - l + 1 and fits comfortably; port the general case anyway.
        let range = (h as i128 - l as i128 + 1) as u64;
        l.wrapping_add(self.uint_bounded(range) as i64)
    }
}

/// Mirrors `igraph_vector_shuffle()` (`src/core/vector.pmt`):
/// Fisher-Yates shuffle drawing `k = RNG_INTEGER(0, n - 1)`.
///
/// A free function rather than a trait method so that [`Rng`] stays
/// dyn-compatible.
pub fn shuffle<T>(rng: &mut dyn Rng, v: &mut [T]) {
    let mut n = v.len();
    while n > 1 {
        let k = rng.integer(0, n as i64 - 1) as usize;
        n -= 1;
        v.swap(n, k);
    }
}

/// The PCG32 generator, igraph's default RNG.
/// Ported from `src/random/rng_pcg32.c` and `vendor/pcg/pcg_variants.h`
/// (set-seq 64-bit state, XSH-RR 32-bit output).
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

/// `PCG_STATE_SETSEQ_64_INITIALIZER` / `PCG32_INITIALIZER`.
const PCG32_INITIALIZER_STATE: u64 = 0x853c_49e6_748f_ea9b;
const PCG32_INITIALIZER_INC: u64 = 0xda3e_39cb_94b9_5bdb;

/// `PCG_DEFAULT_MULTIPLIER_64`.
const PCG_DEFAULT_MULTIPLIER_64: u64 = 6364136223846793005;

impl Pcg32 {
    /// Mirrors `igraph_rng_pcg32_seed()`: the seed fills the sequence number
    /// and the state comes from `PCG32_INITIALIZER`. Seed 0 uses
    /// `PCG32_INITIALIZER.inc >> 1` as the sequence number.
    pub fn seeded(seed: u64) -> Self {
        let mut rng = Pcg32 { state: 0, inc: 0 };
        if seed == 0 {
            rng.srandom(PCG32_INITIALIZER_STATE, PCG32_INITIALIZER_INC >> 1);
        } else {
            rng.srandom(PCG32_INITIALIZER_STATE, seed);
        }
        rng
    }

    /// Mirrors `pcg_setseq_64_srandom_r()`.
    fn srandom(&mut self, initstate: u64, initseq: u64) {
        self.state = 0;
        self.inc = (initseq << 1) | 1;
        self.step();
        self.state = self.state.wrapping_add(initstate);
        self.step();
    }

    /// Mirrors `pcg_setseq_64_step_r()`.
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_DEFAULT_MULTIPLIER_64)
            .wrapping_add(self.inc);
    }
}

impl Default for Pcg32 {
    fn default() -> Self {
        // Mirrors igraph_rng_pcg32_init(): seeds with 0.
        Pcg32::seeded(0)
    }
}

/// Mirrors `pcg_rotr_32()` (portable branch).
fn rotr_32(value: u32, rot: u32) -> u32 {
    value.rotate_right(rot & 31)
}

/// Mirrors `pcg_output_xsh_rr_64_32()`.
fn output_xsh_rr_64_32(state: u64) -> u32 {
    // Note: the XOR-shift is done on the full 64-bit state and only the
    // result of the >> 27 is truncated to 32 bits.
    let x = ((state >> 18) ^ state) >> 27;
    rotr_32(x as u32, (state >> 59) as u32)
}

impl Rng for Pcg32 {
    /// Mirrors `pcg_setseq_64_xsh_rr_32_random_r()`.
    fn get(&mut self) -> u32 {
        let oldstate = self.state;
        self.step();
        output_xsh_rr_64_32(oldstate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unif01_in_unit_interval() {
        let mut rng = Pcg32::seeded(123);
        for _ in 0..10_000 {
            let r = rng.unif01();
            assert!((0.0..1.0).contains(&r));
        }
    }

    #[test]
    fn integer_respects_bounds() {
        let mut rng = Pcg32::seeded(42);
        for n in 1..200_i64 {
            for _ in 0..100 {
                let k = rng.integer(0, n - 1);
                assert!((0..n).contains(&k));
            }
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Pcg32::seeded(7);
        let mut v: Vec<i64> = (0..1000).collect();
        shuffle(&mut rng, &mut v);
        v.sort_unstable();
        assert_eq!(v, (0..1000).collect::<Vec<i64>>());
    }

    #[test]
    fn seeded_stream_is_reproducible() {
        let mut a = Pcg32::seeded(1556328619);
        let mut b = Pcg32::seeded(1556328619);
        for _ in 0..1000 {
            assert_eq!(a.get(), b.get());
        }
    }
}
