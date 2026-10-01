use leiden_rs::{Pcg32, Rng};

/// Compares the RNG stream with C igraph's default PCG32, using the exact
/// same call sequence as `tests/c/rng_stream.c` (the reference program
/// that generated the hardcoded values below).
#[test]
fn rng_stream_vs_c_same_sequence() {
    let mut rng = Pcg32::seeded(579819);

    // C: 10x get_integer(0, 0xFFFFFFFF) -> Lemire 64-bit bounded with
    // range = 2^32; consumes 2 draws and returns the high 32 bits.
    let mut c_gets = vec![
        3873096464u32,
        3273603383,
        124573543,
        2047532982,
        969754779,
        832771703,
        1173228637,
        594130977,
        3804619189,
        3629146721,
    ];
    for expected in c_gets.drain(..) {
        let range: u64 = 1 << 32;
        let x = rng.random_bits_u64(64);
        // t = 0 for range = 2^32, so no rejection loop.
        let value = ((x as u128 * range as u128) >> 64) as u64;
        assert_eq!(value as u32, expected, "get_integer(0, 2^32-1)");
    }

    // C: 20x get_integer(0, 13).
    let c_ints = [
        0i64, 12, 1, 11, 0, 0, 13, 9, 1, 8, 11, 1, 11, 13, 6, 7, 13, 6, 3, 3,
    ];
    for expected in c_ints {
        assert_eq!(rng.integer(0, 13), expected, "get_integer(0, 13)");
    }

    // C: 10x unif01.
    let c_unif = [
        0.6259500307055812_f64,
        0.2793140594869441,
        0.664_350_134_083_100_8,
        0.445_818_137_911_619_5,
        0.505_866_107_522_939_4,
        0.01592724636882803,
        0.716_668_467_035_922_2,
        0.578_412_992_651_008_2,
        0.11367069196796376,
        0.744_128_121_944_745_5,
    ];
    for expected in c_unif {
        let r = rng.unif01();
        assert!((r - expected).abs() < 1e-16, "unif01: {r} vs {expected}");
    }
}
