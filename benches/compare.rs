//! Benchmark harness comparing this Rust port of igraph's Leiden
//! implementation against the C implementation.
//!
//! Graphs are generated deterministically here and written to an edge-list
//! file consumed by both sides, so both implementations run the exact same
//! computation. The C side is a driver in `tests/c/leiden_bench.c`, compiled
//! against the local igraph checkout at bench time (see `common`).
//!
//! Run with `cargo bench`. For an unambiguous comparison the reference
//! igraph should be built at default `-O3` (with FMA contraction); the
//! `-ffp-contract=off` build used by the differential tests is the
//! canonical *numeric* reference but may be marginally slower.

#[path = "../tests/common/mod.rs"]
mod common;

use std::time::Instant;

use leiden_rs::{leiden_simple, Graph, Objective, Pcg32};

/// Cheap xorshift RNG for deterministic graph generation.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

struct CaseDef {
    name: &'static str,
    n: usize,
    directed: bool,
    edges: Vec<(i64, i64)>,
    weights: Option<Vec<f64>>,
}

/// Erdős–Rényi G(n, p).
fn gen_random(n: usize, p: f64, rng: &mut Lcg) -> Vec<(i64, i64)> {
    let mut edges = Vec::new();
    for u in 0..n as u64 {
        for v in u + 1..n as u64 {
            if rng.unit() < p {
                edges.push((u as i64, v as i64));
            }
        }
    }
    edges
}

/// Planted partition: `blocks` blocks of `block_size` vertices, dense
/// within blocks, sparse between them.
fn gen_planted(
    blocks: usize,
    block_size: usize,
    p_in: f64,
    p_out: f64,
    rng: &mut Lcg,
) -> Vec<(i64, i64)> {
    let n = blocks * block_size;
    let mut edges = Vec::new();
    for u in 0..n as u64 {
        for v in u + 1..n as u64 {
            let p = if u / block_size as u64 == v / block_size as u64 {
                p_in
            } else {
                p_out
            };
            if rng.unit() < p {
                edges.push((u as i64, v as i64));
            }
        }
    }
    edges
}

/// Barabási–Albert scale-free graph: each new vertex attaches to `m`
/// existing vertices with probability proportional to degree.
fn gen_ba(n: usize, m: usize, rng: &mut Lcg) -> Vec<(i64, i64)> {
    let mut edges = Vec::new();
    // Endpoint list for proportional-to-degree sampling.
    let mut repeated: Vec<u64> = Vec::with_capacity(2 * n * m);
    for i in 0..m as u64 {
        for j in 0..i {
            edges.push((i as i64, j as i64));
            repeated.push(i);
            repeated.push(j);
        }
    }
    for v in m as u64..n as u64 {
        let mut chosen = std::collections::HashSet::new();
        while chosen.len() < m {
            let t = repeated[rng.below(repeated.len() as u64) as usize];
            chosen.insert(t);
        }
        for &t in &chosen {
            edges.push((v as i64, t as i64));
            repeated.push(v);
            repeated.push(t);
        }
    }
    edges
}

fn weights_for(m: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..m)
        .map(|_| (rng.below(1000) as f64 + 1.0) / 100.0)
        .collect()
}

fn cases() -> Vec<CaseDef> {
    let mut rng = Lcg(0xBADA_55EE);
    let karate_edges: Vec<(i64, i64)> = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 17),
        (0, 19),
        (0, 21),
        (0, 31),
        (1, 2),
        (1, 3),
        (1, 7),
        (1, 13),
        (1, 17),
        (1, 19),
        (1, 21),
        (1, 30),
        (2, 3),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 13),
        (2, 27),
        (2, 28),
        (2, 32),
        (3, 7),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 10),
        (5, 16),
        (6, 16),
        (8, 30),
        (8, 32),
        (8, 33),
        (9, 33),
        (13, 33),
        (14, 32),
        (14, 33),
        (15, 32),
        (15, 33),
        (18, 32),
        (18, 33),
        (19, 33),
        (20, 32),
        (20, 33),
        (22, 32),
        (22, 33),
        (23, 25),
        (23, 27),
        (23, 29),
        (23, 32),
        (23, 33),
        (24, 25),
        (24, 27),
        (24, 31),
        (25, 31),
        (26, 29),
        (26, 33),
        (27, 33),
        (28, 31),
        (28, 33),
        (29, 32),
        (29, 33),
        (30, 32),
        (30, 33),
        (31, 32),
        (31, 33),
        (32, 33),
    ];
    let random_100k = gen_random(10_000, 0.002, &mut rng);
    let planted_2k = gen_planted(10, 200, 0.05, 0.005, &mut rng);
    let planted_10k = gen_planted(50, 200, 0.05, 0.001, &mut rng);
    let ba_1k = gen_ba(1_000, 4, &mut rng);
    let ba_10k = gen_ba(10_000, 4, &mut rng);
    let random_100k_w = weights_for(random_100k.len(), &mut rng);
    let planted_10k_w = weights_for(planted_10k.len(), &mut rng);

    vec![
        CaseDef {
            name: "karate",
            n: 34,
            directed: false,
            edges: karate_edges,
            weights: None,
        },
        CaseDef {
            name: "random-1k",
            n: 1_000,
            directed: false,
            edges: gen_random(1_000, 0.02, &mut rng),
            weights: None,
        },
        CaseDef {
            name: "random-100k",
            n: 10_000,
            directed: false,
            edges: random_100k.clone(),
            weights: None,
        },
        CaseDef {
            name: "planted-2k",
            n: 2_000,
            directed: false,
            edges: planted_2k,
            weights: None,
        },
        CaseDef {
            name: "planted-10k",
            n: 10_000,
            directed: false,
            edges: planted_10k.clone(),
            weights: None,
        },
        CaseDef {
            name: "ba-1k",
            n: 1_000,
            directed: false,
            edges: ba_1k,
            weights: None,
        },
        CaseDef {
            name: "ba-10k",
            n: 10_000,
            directed: false,
            edges: ba_10k,
            weights: None,
        },
        CaseDef {
            name: "random-100k-w",
            n: 10_000,
            directed: false,
            edges: random_100k.clone(),
            weights: Some(random_100k_w),
        },
        CaseDef {
            name: "planted-10k-w",
            n: 10_000,
            directed: false,
            edges: planted_10k.clone(),
            weights: Some(planted_10k_w),
        },
    ]
}

fn runs_per_case(edges: usize) -> i32 {
    if edges < 5_000 {
        50
    } else if edges < 50_000 {
        20
    } else {
        10
    }
}

fn median(times: &mut [f64]) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

/// Run the C driver on `input`, returning (run times in µs, nb_clusters).
fn bench_c(bin: &std::path::PathBuf, input: &str, k: usize) -> (Vec<f64>, i64) {
    let out = common::run_helper(bin, input);
    let mut lines = out.lines();
    let times: Vec<f64> = lines
        .next()
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    assert_eq!(
        times.len(),
        k,
        "C driver did not report {k} run times: {out:?}"
    );
    let last = lines.next().unwrap_or("");
    let nb: i64 = last
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(-1);
    (times, nb)
}

/// Run the Rust port `k` times in-process, returning (run times in µs,
/// nb_clusters of the last run).
fn bench_rust(
    graph: &Graph,
    weights: Option<&[f64]>,
    objective: Objective,
    resolution: f64,
    n_iterations: i64,
    seed: u64,
    k: usize,
) -> (Vec<f64>, i64) {
    let mut times = Vec::with_capacity(k);
    let mut nb = -1_i64;
    for _ in 0..k {
        let mut membership: Vec<i64> = Vec::new();
        let mut rng = Pcg32::seeded(seed);
        let t0 = Instant::now();
        let outcome = leiden_simple(
            graph,
            weights,
            objective,
            resolution,
            0.01,
            false,
            n_iterations,
            &mut membership,
            &mut rng,
        )
        .unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1e6);
        nb = outcome.nb_clusters;
    }
    (times, nb)
}

fn main() {
    let bin = common::c_helper("leiden_bench").expect("failed to build the C benchmark driver");
    let seed = 20_241_001;
    // Optional case filter: LEIDEN_BENCH_ONLY=<substring> benches a subset.
    let only = std::env::var("LEIDEN_BENCH_ONLY").ok();

    println!(
        "{:<14} {:>8} {:<11} {:>4} {:>12} {:>12} {:>12} {:>12} {:>8} match",
        "graph",
        "edges",
        "objective",
        "nit",
        "C best µs",
        "C med µs",
        "Rs best µs",
        "Rs med µs",
        "C/Rs"
    );
    println!("{}", "-".repeat(105));

    let mut total_c = 0.0_f64;
    let mut total_r = 0.0_f64;

    for case in cases() {
        if let Some(pattern) = &only {
            if !case.name.contains(pattern.as_str()) {
                continue;
            }
        }
        let graph = Graph::new(case.n, case.directed, &case.edges).unwrap();
        let k = runs_per_case(case.edges.len()) as usize;
        let weighted = u8::from(case.weights.is_some());

        // Edge-list section of the driver input, shared by all combos.
        let mut edges_input = String::with_capacity(24 + 32 * case.edges.len());
        edges_input.push_str(&format!(
            "{} {} {} {}\n",
            case.n,
            case.edges.len(),
            u8::from(case.directed),
            weighted
        ));
        for (i, &(a, b)) in case.edges.iter().enumerate() {
            let w = case.weights.as_ref().map_or(1.0, |w| w[i]);
            edges_input.push_str(&format!("{a} {b} {w}\n"));
        }

        for &objective in &[Objective::Modularity, Objective::Cpm, Objective::Er] {
            let obj_name = match objective {
                Objective::Modularity => "modularity",
                Objective::Cpm => "cpm",
                Objective::Er => "er",
            };
            let resolution = match objective {
                Objective::Modularity => 1.0,
                Objective::Cpm => 0.1,
                Objective::Er => 1.0,
            };
            for &n_iterations in &[2_i64, -1] {
                let mut input = edges_input.clone();
                input.push_str(&format!(
                    "{resolution} 0.01 {} {n_iterations} {seed} {k}\n",
                    match objective {
                        Objective::Modularity => 1,
                        Objective::Cpm => 2,
                        Objective::Er => 3,
                    }
                ));

                let (c_times, c_nb) = bench_c(&bin, &input, k);
                let (r_times, r_nb) = bench_rust(
                    &graph,
                    case.weights.as_deref(),
                    objective,
                    resolution,
                    n_iterations,
                    seed,
                    k,
                );

                let c_best = c_times.iter().cloned().fold(f64::INFINITY, f64::min);
                let c_med = median(&mut c_times.clone());
                let r_best = r_times.iter().cloned().fold(f64::INFINITY, f64::min);
                let r_med = median(&mut r_times.clone());
                let speedup = c_med / r_med;
                let matched = c_nb == r_nb;

                total_c += c_med * k as f64;
                total_r += r_med * k as f64;

                println!(
                    "{:<14} {:>8} {:<11} {:>4} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>7.2}x {}",
                    case.name,
                    case.edges.len(),
                    obj_name,
                    if n_iterations < 0 { "-1" } else { "2" },
                    c_best,
                    c_med,
                    r_best,
                    r_med,
                    speedup,
                    if matched { "ok" } else { "MISMATCH" }
                );
            }
        }
    }

    println!("{}", "-".repeat(105));
    println!(
        "total median-weighted time: C {total_c:.0} µs, Rust {total_r:.0} µs (C/Rust = {:.2}x)",
        total_c / total_r
    );
}
