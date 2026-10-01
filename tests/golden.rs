//! Golden test ported from igraph's `tests/unit/community_leiden.c`
//! (commit `8225c3a4b`), validating quality and membership output against
//! igraph's own expected output in `tests/unit/community_leiden.out`.
//!
//! The expected values were verified to be reproducible with the reference
//! C build compiled with `-ffp-contract=off` (matching Rust's guaranteed
//! lack of floating-point contraction).

use leiden_rs::{leiden, leiden_simple, modularity, strength, Graph, Objective, Pcg32};

const TOL: f64 = 1e-15;

/// Port of `igraph_almost_equals(a, b, eps)` (`src/math/utils.c`,
/// `igraph_cmp_epsilon()`).
fn almost_equals(a: f64, b: f64, eps: f64) -> bool {
    if a == b {
        return true;
    }
    let diff = a - b;
    let abs_diff = diff.abs();
    let sum = a.abs() + b.abs();
    if a == 0.0 || b == 0.0 || sum < f64::MIN_POSITIVE {
        abs_diff < eps * f64::MIN_POSITIVE
    } else if !sum.is_finite() {
        abs_diff < eps * a.abs() + eps * b.abs()
    } else {
        abs_diff / sum < eps
    }
}

/// Format a quality value like the C test's `printf("%.5f")` output,
/// including the `fabs(quality) < TOL` clamping to zero and NaN handling.
fn format_quality(q: f64) -> String {
    if q.is_nan() {
        "nan".to_string()
    } else if q.abs() < TOL {
        format!("{:.5}", 0.0)
    } else {
        format!("{q:.5}")
    }
}

#[allow(clippy::too_many_arguments)]
fn check_case(
    graph: &Graph,
    weights: Option<&[f64]>,
    objective: Objective,
    generic_resolution: f64,
    simple_resolution: f64,
    expected_nb: i64,
    expected_quality: &str,
    expected_membership: &[i64],
) {
    let n = graph.vcount() as usize;

    // Generic interface.
    let (out_w, in_w) = match objective {
        Objective::Modularity => {
            let out = strength(graph, false, weights);
            let in_w = if graph.is_directed() {
                Some(strength(graph, true, weights))
            } else {
                None
            };
            (Some(out), in_w)
        }
        _ => (None, None),
    };
    let mut membership: Vec<i64> = vec![0; n];
    let mut rng = Pcg32::seeded(123);
    let outcome = leiden(
        graph,
        weights,
        out_w.as_deref(),
        in_w.as_deref(),
        generic_resolution,
        0.01,
        false,
        2,
        &mut membership,
        &mut rng,
    )
    .unwrap();
    let quality = outcome.quality;

    // Cross-check the quality against the independently computed modularity.
    if objective == Objective::Modularity {
        let quality2 = modularity(graph, &membership, weights, 1.0, true).unwrap();
        if quality.is_nan() {
            assert!(quality2.is_nan());
        } else {
            assert!(
                almost_equals(quality, quality2, TOL),
                "quality {quality} does not match modularity {quality2}"
            );
        }
    }

    // Simplified interface with the same seed must give the same quality.
    let mut membership2: Vec<i64> = vec![0; n];
    let mut rng = Pcg32::seeded(123);
    let outcome2 = leiden_simple(
        graph,
        weights,
        objective,
        simple_resolution,
        0.01,
        false,
        2,
        &mut membership2,
        &mut rng,
    )
    .unwrap();
    let quality2 = outcome2.quality;
    assert!(
        (quality.is_nan() && quality2.is_nan()) || almost_equals(quality, quality2, TOL),
        "generic quality {quality} does not match simple quality {quality2}"
    );

    // Compare with the golden output.
    assert_eq!(outcome.nb_clusters, expected_nb, "nb_clusters");
    assert_eq!(
        format_quality(quality),
        expected_quality,
        "quality (nb={})",
        expected_nb
    );
    assert_eq!(membership, expected_membership, "membership");
}

fn run_leiden_modularity(
    graph: &Graph,
    weights: Option<&[f64]>,
    expected_nb: i64,
    expected_quality: &str,
    expected_membership: &[i64],
) {
    let directed = graph.is_directed();
    let dm = if directed { 1.0 } else { 2.0 };
    let m = match weights {
        Some(w) => w.iter().sum(),
        None => graph.ecount() as f64,
    };
    check_case(
        graph,
        weights,
        Objective::Modularity,
        1.0 / (dm * m),
        1.0,
        expected_nb,
        expected_quality,
        expected_membership,
    );
}

fn run_leiden_cpm(
    graph: &Graph,
    weights: Option<&[f64]>,
    resolution: f64,
    expected_nb: i64,
    expected_quality: &str,
    expected_membership: &[i64],
) {
    check_case(
        graph,
        weights,
        Objective::Cpm,
        resolution,
        resolution,
        expected_nb,
        expected_quality,
        expected_membership,
    );
}

#[test]
fn golden_community_leiden() {
    // Simple unweighted graph.
    let graph = Graph::new(
        10,
        false,
        &[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (6, 7),
            (6, 8),
            (6, 9),
            (7, 8),
            (7, 9),
            (8, 9),
            (0, 5),
        ],
    )
    .unwrap();
    run_leiden_modularity(&graph, None, 2, "0.45238", &[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

    // Same simple graph, with uniform edge weights.
    let weights = vec![2.0; graph.ecount() as usize];
    run_leiden_modularity(
        &graph,
        Some(&weights),
        2,
        "0.45238",
        &[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    );

    // Same simple graph, but directed with reciprocal edges.
    let mut mutual: Vec<(i64, i64)> = Vec::new();
    for e in 0..graph.ecount() {
        let (a, b) = (graph.from(e), graph.to(e));
        mutual.push((a, b));
        mutual.push((b, a));
    }
    let directed = Graph::new(10, true, &mutual).unwrap();
    run_leiden_modularity(
        &directed,
        None,
        2,
        "0.45238",
        &[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    );

    // Tiny directed graph; optimal community structure is different if
    // ignoring edge directions.
    let tiny_edges = [(0, 2), (0, 3), (1, 2), (3, 1), (3, 2)];
    let tiny = Graph::new(4, true, &tiny_edges).unwrap();
    run_leiden_modularity(&tiny, None, 2, "0.08000", &[0, 1, 1, 0]);
    let tiny_undirected = Graph::new(4, false, &tiny_edges).unwrap();
    run_leiden_modularity(&tiny_undirected, None, 1, "0.00000", &[0, 0, 0, 0]);

    // Larger directed graph; optimal community structure is different if
    // ignoring edge directions.
    let large_edges = [
        (0, 3),
        (0, 4),
        (1, 0),
        (1, 4),
        (2, 1),
        (3, 0),
        (3, 2),
        (4, 0),
        (4, 3),
        (4, 7),
        (5, 0),
        (5, 1),
        (5, 3),
        (5, 6),
        (5, 8),
        (5, 9),
        (7, 0),
        (8, 2),
        (8, 3),
        (9, 1),
        (9, 3),
        (9, 8),
    ];
    let large = Graph::new(10, true, &large_edges).unwrap();
    run_leiden_modularity(&large, None, 3, "0.20868", &[0, 1, 1, 0, 0, 2, 2, 0, 2, 2]);
    let large_undirected = Graph::new(10, false, &large_edges).unwrap();
    run_leiden_modularity(
        &large_undirected,
        None,
        2,
        "0.18079",
        &[0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    );

    // Simple nonuniform weighted graph, with and without weights.
    let graph = Graph::new(
        6,
        false,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
    )
    .unwrap();
    let weights = [10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    run_leiden_modularity(&graph, None, 2, "0.17969", &[0, 0, 1, 1, 1, 1]);
    run_leiden_modularity(&graph, Some(&weights), 2, "0.17086", &[0, 0, 0, 1, 1, 1]);

    // Zachary Karate club.
    let karate_edges = [
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
    let karate = Graph::new(34, false, &karate_edges).unwrap();
    run_leiden_modularity(
        &karate,
        None,
        4,
        "0.41979",
        &[
            0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 3, 3, 3, 2, 3, 3,
            2, 2, 3, 2, 2,
        ],
    );
    run_leiden_cpm(
        &karate,
        None,
        0.06,
        2,
        "0.64949",
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ],
    );

    // Simple disconnected graph with isolates.
    let graph = Graph::new(
        9,
        false,
        &[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (4, 7),
            (5, 6),
            (5, 7),
            (6, 7),
        ],
    )
    .unwrap();
    run_leiden_modularity(&graph, None, 3, "0.50000", &[0, 0, 0, 0, 1, 1, 1, 1, 2]);

    // Disjoint union of two rings.
    let graph = Graph::new(
        20,
        false,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (0, 9),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (10, 19),
        ],
    )
    .unwrap();
    run_leiden_modularity(
        &graph,
        None,
        4,
        "0.55000",
        &[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    );
    run_leiden_cpm(
        &graph,
        None,
        0.05,
        2,
        "0.75000",
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    );

    // Completely empty graph.
    let graph = Graph::new(10, false, &[]).unwrap();
    run_leiden_modularity(&graph, None, 10, "nan", &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    // Ring graph without loop edges.
    let ring = Graph::new(6, false, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]).unwrap();
    run_leiden_cpm(&ring, None, 0.4, 2, "0.06667", &[0, 0, 1, 1, 1, 0]);

    // Ring graph with loop edges.
    let ring_loops = Graph::new(
        6,
        false,
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ],
    )
    .unwrap();
    run_leiden_cpm(&ring_loops, None, 0.4, 2, "0.53333", &[0, 0, 1, 1, 1, 0]);

    // Regression test -- graph with two vertices and two edges.
    let graph = Graph::new(2, false, &[(0, 0), (1, 1)]).unwrap();
    run_leiden_modularity(&graph, None, 2, "0.50000", &[0, 1]);

    // The next two tests need an empty weight vector.

    // Null graph.
    let graph = Graph::new(0, false, &[]).unwrap();
    run_leiden_modularity(&graph, Some(&[]), 0, "nan", &[]);

    // Edgeless graph.
    let graph = Graph::new(5, false, &[]).unwrap();
    run_leiden_modularity(&graph, Some(&[]), 5, "nan", &[0, 1, 2, 3, 4]);
}

/// Every cluster found by Leiden must be connected.
fn check_clusters_connected(graph: &Graph, membership: &[i64], nb_clusters: i64) {
    for c in 0..nb_clusters {
        let members: Vec<usize> = (0..graph.vcount() as usize)
            .filter(|&v| membership[v] == c)
            .collect();
        assert!(!members.is_empty());
        // Weak connectivity via BFS.
        let start = members[0];
        let mut visited = vec![false; graph.vcount() as usize];
        visited[start] = true;
        let mut queue = vec![start];
        while let Some(v) = queue.pop() {
            for e in graph.incident(v as i64) {
                let u = graph.other(e, v as i64) as usize;
                if !visited[u] {
                    visited[u] = true;
                    queue.push(u);
                }
            }
        }
        for &v in &members {
            assert!(visited[v], "cluster {c} is not connected");
        }
    }
}

/// Port of `test_last_level_moves_are_kept()`: on this instance, local
/// moving on the last aggregation level of an iteration moves vertices.
/// Those moves used to be lost: the result then put the unconnected pairs
/// {1, 9} and {2, 3} into one cluster for any number of iterations, and
/// `n_iterations = -1` never returned.
#[test]
fn golden_last_level_moves_are_kept() {
    let edges = [(9, 1), (3, 2), (8, 6)];
    let weights = [1.7255847013682128, 1.2353514434357051, 0.7294595456799059];
    let start = [9, 0, 0, 9, 0, 8, 8, 4, 10, 1, 10];
    let graph = Graph::new(11, false, &edges).unwrap();

    for &budget in &[1_i64, -1] {
        let mut membership: Vec<i64> = start.to_vec();
        let mut rng = Pcg32::seeded(1556328619);
        let outcome = leiden_simple(
            &graph,
            Some(&weights),
            Objective::Er,
            1.2087812986446955,
            0.01,
            true,
            budget,
            &mut membership,
            &mut rng,
        )
        .unwrap();
        check_clusters_connected(&graph, &membership, outcome.nb_clusters);
    }
}

/// Port of the input-validation checks at the end of the C golden test.
#[test]
fn golden_input_validation() {
    let graph = Graph::new(4, false, &[(0, 1), (1, 2), (2, 0), (0, 3), (3, 3)]).unwrap();
    let weights: Vec<f64> = (1..=graph.ecount()).map(|i| i as f64).collect();
    let mut membership: Vec<i64> = Vec::new();
    let mut rng = Pcg32::seeded(0);

    // Omitting membership should raise no error (the Rust API always takes a
    // membership vector; with `start = false` it is reinitialized).
    leiden_simple(
        &graph,
        Some(&weights),
        Objective::Modularity,
        1.0,
        0.01,
        false,
        1,
        &mut membership,
        &mut rng,
    )
    .unwrap();

    // Negative weight.
    let mut neg = weights.clone();
    neg[0] = -1.0;
    let err = leiden_simple(
        &graph,
        Some(&neg),
        Objective::Modularity,
        1.0,
        0.01,
        false,
        1,
        &mut membership,
        &mut rng,
    )
    .unwrap_err();
    assert_eq!(
        err.message,
        "Edge weights must not be negative for Leiden community detection with modularity objective function, got -1."
    );
    let err = leiden_simple(
        &graph,
        Some(&neg),
        Objective::Er,
        1.0,
        0.01,
        false,
        1,
        &mut membership,
        &mut rng,
    )
    .unwrap_err();
    assert_eq!(
        err.message,
        "Edge weights must not be negative for Leiden community detection with ER objective function, got -1."
    );

    // NaN weight.
    let mut nan = weights.clone();
    nan[0] = f64::NAN;
    let err = leiden_simple(
        &graph,
        Some(&nan),
        Objective::Cpm,
        1.0,
        0.01,
        false,
        1,
        &mut membership,
        &mut rng,
    )
    .unwrap_err();
    assert_eq!(
        err.message,
        "Edge weights must not be infinite or NaN, got nan."
    );

    // Invalid weight vector length.
    let too_long: Vec<f64> = (1..=graph.ecount() + 1).map(|i| i as f64).collect();
    let err = leiden_simple(
        &graph,
        Some(&too_long),
        Objective::Modularity,
        1.0,
        0.01,
        false,
        1,
        &mut membership,
        &mut rng,
    )
    .unwrap_err();
    assert_eq!(
        err.message,
        "Edge weight vector length does not match number of edges."
    );
}
