//! Differential test: the Rust port vs the C igraph implementation.
//!
//! The C driver (`tests/c/leiden_driver.c`, compiled against the local
//! igraph checkout at test time) runs `igraph_community_leiden()` /
//! `igraph_community_leiden_simple()` with a given seed. The Rust port
//! consumes an identical random number stream (ported PCG32), so identical
//! seeds and inputs must produce identical memberships and quality values,
//! bit for bit.
//!
//! The igraph checkout is located via the `IGRAPH_DIR` environment variable
//! or a sibling `../igraph` directory; the test is skipped when the
//! reference implementation is unavailable.

mod common;

use leiden_rs::{leiden, leiden_simple, Graph, Objective, Pcg32, Rng};

fn run_c(input: &str) -> String {
    let bin = common::c_helper("leiden_driver").expect("failed to build C driver");
    common::run_helper(&bin, input)
}

struct Case {
    n: usize,
    directed: bool,
    edges: Vec<(i64, i64)>,
    weights: Option<Vec<f64>>,
    resolution: f64,
    beta: f64,
    start: bool,
    membership: Option<Vec<i64>>,
    n_iterations: i64,
    seed: u64,
}

/// Quality comparison tolerating infinities and NaN (e.g. empty graphs,
/// where the quality is 0/0).
fn quality_matches(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan())
        || a == b
        || (a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-15 * f64::max(b.abs(), 1.0))
}

fn compare(case: &Case) {
    let m = case.edges.len();
    let mut input = String::new();

    // Generic interface: vertex out/in weights are all 1 (as used by CPM).
    input.push_str("generic\n");
    input.push_str(&format!(
        "{} {} {}\n",
        case.n,
        if case.directed { 1 } else { 0 },
        m
    ));
    for (i, &(a, b)) in case.edges.iter().enumerate() {
        let w = case.weights.as_ref().map_or(1.0, |w| w[i]);
        input.push_str(&format!("{a} {b} {w:.17}\n"));
    }
    input.push_str(&format!(
        "{:.17} {:.17} {} {} {}\n",
        case.resolution,
        case.beta,
        if case.start { 1 } else { 0 },
        case.n_iterations,
        case.seed
    ));
    let start_membership: Vec<i64> = match &case.membership {
        Some(mem) => mem.clone(),
        None => (0..case.n as i64).collect(),
    };
    for v in &start_membership {
        input.push_str(&format!("{v} "));
    }
    input.push('\n');

    let c_out = run_c(&input);

    // Run the Rust port (generic interface).
    let g = Graph::new(case.n, case.directed, &case.edges).unwrap();
    let mut membership = start_membership.clone();
    let mut rng = Pcg32::seeded(case.seed);
    let outcome = leiden(
        &g,
        case.weights.as_deref(),
        None,
        None,
        case.resolution,
        case.beta,
        case.start,
        case.n_iterations,
        &mut membership,
        &mut rng,
    )
    .unwrap_or_else(|e| panic!("Rust port failed: {e}"));

    // Parse C output.
    let mut lines = c_out.lines();
    let err_line = lines.next().unwrap().trim();
    assert_eq!(err_line, "err 0", "C driver errored: {c_out}");
    let stats = lines.next().unwrap().trim();
    let mut stats = stats.split_whitespace();
    let c_nb_clusters: i64 = stats.next().unwrap().parse().unwrap();
    let c_quality: f64 = stats.next().unwrap().parse().unwrap();
    let c_membership: Vec<i64> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();

    assert_eq!(
        outcome.nb_clusters, c_nb_clusters,
        "nb_clusters mismatch (edges={:?}, seed={})",
        case.edges, case.seed
    );
    assert!(
        quality_matches(outcome.quality, c_quality),
        "quality mismatch: rust={} c={} (edges={:?}, seed={})",
        outcome.quality,
        c_quality,
        case.edges,
        case.seed
    );
    assert_eq!(
        membership, c_membership,
        "membership mismatch (edges={:?}, seed={})",
        case.edges, case.seed
    );
}

fn compare_simple(case: &Case, objective: Objective) {
    let objective_num = match objective {
        Objective::Modularity => 1,
        Objective::Cpm => 2,
        Objective::Er => 3,
    };
    let m = case.edges.len();
    let mut input = String::new();
    input.push_str("simple\n");
    input.push_str(&format!(
        "{} {} {}\n",
        case.n,
        if case.directed { 1 } else { 0 },
        m
    ));
    for (i, &(a, b)) in case.edges.iter().enumerate() {
        let w = case.weights.as_ref().map_or(1.0, |w| w[i]);
        input.push_str(&format!("{a} {b} {w:.17}\n"));
    }
    input.push_str(&format!(
        "{:.17} {:.17} {} {} {}\n",
        case.resolution,
        case.beta,
        if case.start { 1 } else { 0 },
        case.n_iterations,
        case.seed
    ));
    input.push_str(&format!("{objective_num}\n"));
    let start_membership: Vec<i64> = match &case.membership {
        Some(mem) => mem.clone(),
        None => (0..case.n as i64).collect(),
    };
    for v in &start_membership {
        input.push_str(&format!("{v} "));
    }
    input.push('\n');

    let c_out = run_c(&input);

    let g = Graph::new(case.n, case.directed, &case.edges).unwrap();
    let mut membership = start_membership.clone();
    let mut rng = Pcg32::seeded(case.seed);
    let outcome = leiden_simple(
        &g,
        case.weights.as_deref(),
        objective,
        case.resolution,
        case.beta,
        case.start,
        case.n_iterations,
        &mut membership,
        &mut rng,
    )
    .unwrap_or_else(|e| panic!("Rust port failed: {e}"));

    let mut lines = c_out.lines();
    let err_line = lines.next().unwrap().trim();
    assert_eq!(err_line, "err 0", "C driver errored: {c_out}");
    let stats = lines.next().unwrap().trim();
    let mut stats = stats.split_whitespace();
    let c_nb_clusters: i64 = stats.next().unwrap().parse().unwrap();
    let c_quality: f64 = stats.next().unwrap().parse().unwrap();
    let c_membership: Vec<i64> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();

    assert_eq!(
        outcome.nb_clusters, c_nb_clusters,
        "nb_clusters mismatch (edges={:?}, seed={})",
        case.edges, case.seed
    );
    assert!(
        quality_matches(outcome.quality, c_quality),
        "quality mismatch: rust={} c={} (edges={:?}, seed={})",
        outcome.quality,
        c_quality,
        case.edges,
        case.seed
    );
    assert_eq!(
        membership, c_membership,
        "membership mismatch (edges={:?}, seed={})",
        case.edges, case.seed
    );
}

#[test]
fn differential_fixed_cases() {
    if common::skip_if_no_igraph() {
        return;
    }
    // Triangle.
    let triangle = Case {
        n: 3,
        directed: false,
        edges: vec![(0, 1), (1, 2), (2, 0)],
        weights: None,
        resolution: 1.0,
        beta: 0.01,
        start: false,
        membership: None,
        n_iterations: 2,
        seed: 123,
    };
    compare(&triangle);
    compare_simple(&triangle, Objective::Modularity);
    compare_simple(&triangle, Objective::Cpm);
    compare_simple(&triangle, Objective::Er);

    // Karate club graph.
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
    let karate = Case {
        n: 34,
        directed: false,
        edges: karate_edges,
        weights: None,
        resolution: 1.0,
        beta: 0.01,
        start: false,
        membership: None,
        n_iterations: 2,
        seed: 123,
    };
    compare(&karate);
    compare_simple(&karate, Objective::Modularity);
    compare_simple(&karate, Objective::Cpm);

    // Directed graph with in-weights ignored (leiden with None in-weights on
    // a directed graph means in == out).
    let directed = Case {
        n: 4,
        directed: true,
        edges: vec![(0, 2), (0, 3), (1, 2), (3, 1), (3, 2)],
        weights: None,
        resolution: 1.0,
        beta: 0.01,
        start: false,
        membership: None,
        n_iterations: 2,
        seed: 42,
    };
    compare(&directed);

    // Start from a provided membership.
    let started = Case {
        n: 5,
        directed: false,
        edges: vec![(0, 1), (1, 2), (3, 4), (0, 2), (3, 2)],
        weights: Some(vec![1.0, 2.0, 1.5, 1.0, 0.5]),
        resolution: 0.5,
        beta: 0.01,
        start: true,
        membership: Some(vec![0, 0, 0, 1, 1]),
        n_iterations: -1,
        seed: 7,
    };
    compare(&started);
    compare_simple(&started, Objective::Cpm);
    compare_simple(&started, Objective::Modularity);
    compare_simple(&started, Objective::Er);
}

#[test]
fn differential_randomized() {
    if common::skip_if_no_igraph() {
        return;
    }
    let mut state: u64 = 0xDEADBEEF;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for trial in 0..60 {
        let n = (rand() % 20 + 1) as usize;
        let directed = rand() % 2 == 0;
        let m = (rand() % 40) as usize;
        let mut edges = Vec::with_capacity(m);
        for _ in 0..m {
            let a = (rand() % n as u64) as i64;
            let b = (rand() % n as u64) as i64;
            edges.push((a, b));
        }
        let weighted = rand() % 2 == 0;
        let weights: Option<Vec<f64>> = if weighted {
            Some(
                (0..m)
                    .map(|_| ((rand() % 1000) as f64 + 1.0) / 100.0)
                    .collect(),
            )
        } else {
            None
        };
        let n_iterations = if rand() % 3 == 0 {
            -1
        } else {
            (rand() % 3) as i64
        };
        let seed = rand() % 1_000_000;

        let case = Case {
            n,
            directed,
            edges,
            weights,
            resolution: 1.0,
            beta: 0.01,
            start: false,
            membership: None,
            n_iterations,
            seed,
        };
        compare(&case);
        compare_simple(&case, Objective::Modularity);
        compare_simple(&case, Objective::Cpm);
        compare_simple(&case, Objective::Er);
        let _ = trial;
    }
}

// Suppress unused import warning for Rng (used transitively via Pcg32).
#[allow(unused)]
fn _rng_trait_in_scope(_r: &mut dyn Rng) {}
