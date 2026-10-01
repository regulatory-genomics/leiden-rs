//! Property tests: invariants that must hold for any Leiden run, checked on
//! randomly generated graphs.
//!
//! - Determinism: identical seeds and inputs produce identical output.
//! - Membership validity: correct length, consecutive cluster indices.
//! - Every cluster is (weakly) connected.
//! - More iterations never decrease the quality.
//! - The modularity objective's quality matches the independent
//!   `modularity()` port.
//! - `leiden` and `leiden_simple` agree for the modularity objective.
//! - The parallel variants (`leiden_parallel`, `leiden_simple_parallel`)
//!   satisfy the same structural guarantees, are deterministic run-to-run,
//!   and land in the same quality ballpark as the sequential path.

use leiden_rs::{
    leiden, leiden_parallel, leiden_simple, leiden_simple_parallel, modularity, strength, Graph,
    Objective, Pcg32,
};

/// A randomly generated test case: (vcount, directed, edges, weights).
type RandomCase = (usize, bool, Vec<(i64, i64)>, Option<Vec<f64>>);

/// Cheap xorshift RNG for generating random test graphs.
struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_graphs(trials: usize) -> Vec<RandomCase> {
    let mut rng = Xorshift(0x5EED_5EED);
    let mut cases: Vec<RandomCase> = Vec::with_capacity(trials);
    for _ in 0..trials {
        let n = rng.below(25) as usize + 1;
        let directed = rng.below(2) == 0;
        let m = rng.below(50) as usize;
        let mut edges = Vec::with_capacity(m);
        for _ in 0..m {
            let a = rng.below(n as u64) as i64;
            let b = rng.below(n as u64) as i64;
            edges.push((a, b));
        }
        let weighted = rng.below(2) == 0;
        let weights = if weighted {
            Some(
                (0..m)
                    .map(|_| (rng.below(1000) as f64 + 1.0) / 100.0)
                    .collect(),
            )
        } else {
            None
        };
        cases.push((n, directed, edges, weights));
    }
    cases
}

/// Every cluster found by Leiden must be (weakly) connected, and the
/// membership must be valid.
fn check_membership(graph: &Graph, membership: &[i64], nb_clusters: i64) {
    assert_eq!(membership.len() as i64, graph.vcount());
    assert!(nb_clusters >= 0 && nb_clusters <= graph.vcount());
    for &c in membership {
        assert!((0..nb_clusters).contains(&c), "invalid cluster id {c}");
    }
    // Cluster indices are consecutive (membership is reindexed).
    let max = membership.iter().max().copied().unwrap_or(-1);
    assert_eq!(max + 1, nb_clusters, "cluster indices are not consecutive");

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
            assert!(visited[v], "cluster {c} is not (weakly) connected");
        }
    }
}

#[test]
fn properties_invariants() {
    let cases = random_graphs(60);
    for (trial, (n, directed, edges, weights)) in cases.iter().enumerate() {
        let graph = Graph::new(*n, *directed, edges).unwrap();

        // Generic interface with modularity vertex weights.
        let out = strength(&graph, false, weights.as_deref());
        let in_w = if *directed {
            Some(strength(&graph, true, weights.as_deref()))
        } else {
            None
        };
        let dm = if *directed { 1.0 } else { 2.0 };
        let m = match weights {
            Some(w) => w.iter().sum(),
            None => graph.ecount() as f64,
        };
        let resolution = if m > 0.0 { 1.0 / (dm * m) } else { 1.0 };

        // n_iterations = -1 must terminate and produce a valid clustering.
        let mut membership: Vec<i64> = vec![0; *n];
        let mut rng = Pcg32::seeded(42);
        let outcome = leiden(
            &graph,
            weights.as_deref(),
            Some(&out),
            in_w.as_deref(),
            resolution,
            0.01,
            false,
            -1,
            &mut membership,
            &mut rng,
        )
        .unwrap();
        check_membership(&graph, &membership, outcome.nb_clusters);

        // The quality must match the independent modularity computation.
        let q2 = modularity(&graph, &membership, weights.as_deref(), 1.0, true).unwrap();
        if outcome.quality.is_nan() {
            assert!(q2.is_nan());
        } else {
            assert!(
                (outcome.quality - q2).abs() <= 1e-15 * f64::max(q2.abs(), 1.0),
                "trial {trial}: quality {} != modularity {q2}",
                outcome.quality
            );
        }

        // Determinism: the same seed reproduces the same result.
        let mut membership2: Vec<i64> = vec![0; *n];
        let mut rng = Pcg32::seeded(42);
        let outcome2 = leiden(
            &graph,
            weights.as_deref(),
            Some(&out),
            in_w.as_deref(),
            resolution,
            0.01,
            false,
            -1,
            &mut membership2,
            &mut rng,
        )
        .unwrap();
        assert_eq!(membership, membership2, "trial {trial}: not deterministic");
        assert_eq!(outcome.nb_clusters, outcome2.nb_clusters);
        assert!(
            outcome.quality == outcome2.quality
                || (outcome.quality.is_nan() && outcome2.quality.is_nan()),
            "trial {trial}: not deterministic (quality)"
        );

        // More iterations never decrease the quality.
        let mut membership3: Vec<i64> = vec![0; *n];
        let mut rng = Pcg32::seeded(42);
        let outcome3 = leiden(
            &graph,
            weights.as_deref(),
            Some(&out),
            in_w.as_deref(),
            resolution,
            0.01,
            false,
            1,
            &mut membership3,
            &mut rng,
        )
        .unwrap();
        if outcome.quality.is_finite() && outcome3.quality.is_finite() {
            assert!(
                outcome.quality >= outcome3.quality - 1e-12 * f64::max(outcome3.quality.abs(), 1.0),
                "trial {trial}: quality decreased with more iterations ({} < {})",
                outcome.quality,
                outcome3.quality
            );
        }

        // leiden_simple (modularity, resolution 1.0) agrees with the generic
        // interface given the same seed.
        let mut membership4: Vec<i64> = vec![0; *n];
        let mut rng = Pcg32::seeded(42);
        let outcome4 = leiden_simple(
            &graph,
            weights.as_deref(),
            Objective::Modularity,
            1.0,
            0.01,
            false,
            -1,
            &mut membership4,
            &mut rng,
        )
        .unwrap();
        if outcome.quality.is_nan() {
            assert!(outcome4.quality.is_nan());
        } else {
            assert!(
                (outcome.quality - outcome4.quality).abs()
                    <= 1e-15 * f64::max(outcome4.quality.abs(), 1.0),
                "trial {trial}: generic and simple quality disagree"
            );
        }
    }
}

#[test]
fn properties_parallel_invariants() {
    let cases = random_graphs(60);
    for (trial, (n, directed, edges, weights)) in cases.iter().enumerate() {
        let graph = Graph::new(*n, *directed, edges).unwrap();

        let out = strength(&graph, false, weights.as_deref());
        let in_w = if *directed {
            Some(strength(&graph, true, weights.as_deref()))
        } else {
            None
        };
        let dm = if *directed { 1.0 } else { 2.0 };
        let m = match weights {
            Some(w) => w.iter().sum(),
            None => graph.ecount() as f64,
        };
        let resolution = if m > 0.0 { 1.0 / (dm * m) } else { 1.0 };

        // n_iterations = -1 must terminate and produce a valid clustering.
        let mut membership: Vec<i64> = vec![0; *n];
        let mut rng = Pcg32::seeded(42);
        let outcome = leiden_parallel(
            &graph,
            weights.as_deref(),
            Some(&out),
            in_w.as_deref(),
            resolution,
            0.01,
            false,
            -1,
            &mut membership,
            &mut rng,
        )
        .unwrap();
        check_membership(&graph, &membership, outcome.nb_clusters);

        // The parallel quality must match the independent modularity
        // computation.
        let q2 = modularity(&graph, &membership, weights.as_deref(), 1.0, true).unwrap();
        if outcome.quality.is_nan() {
            assert!(q2.is_nan());
        } else {
            assert!(
                (outcome.quality - q2).abs() <= 1e-15 * f64::max(q2.abs(), 1.0),
                "trial {trial}: parallel quality {} != modularity {q2}",
                outcome.quality
            );
        }

        // Determinism: the same seed reproduces the same result.
        let mut membership2: Vec<i64> = vec![0; *n];
        let mut rng = Pcg32::seeded(42);
        let outcome2 = leiden_parallel(
            &graph,
            weights.as_deref(),
            Some(&out),
            in_w.as_deref(),
            resolution,
            0.01,
            false,
            -1,
            &mut membership2,
            &mut rng,
        )
        .unwrap();
        assert_eq!(membership, membership2, "trial {trial}: not deterministic");
        assert_eq!(outcome.nb_clusters, outcome2.nb_clusters);
        assert!(
            outcome.quality == outcome2.quality
                || (outcome.quality.is_nan() && outcome2.quality.is_nan()),
            "trial {trial}: not deterministic (quality)"
        );

        // More iterations never decrease the quality (parallel path).
        let mut membership3: Vec<i64> = vec![0; *n];
        let mut rng = Pcg32::seeded(42);
        let outcome3 = leiden_parallel(
            &graph,
            weights.as_deref(),
            Some(&out),
            in_w.as_deref(),
            resolution,
            0.01,
            false,
            1,
            &mut membership3,
            &mut rng,
        )
        .unwrap();
        if outcome.quality.is_finite() && outcome3.quality.is_finite() {
            assert!(
                outcome.quality >= outcome3.quality - 1e-12 * f64::max(outcome3.quality.abs(), 1.0),
                "trial {trial}: parallel quality decreased with more iterations ({} < {})",
                outcome.quality,
                outcome3.quality
            );
        }

        // Parallel and sequential land in the same quality ballpark: the
        // level-0 local-moving phase is identical between the two paths;
        // only the refinement phase draws from different random streams, so
        // the final partitions (and qualities) may differ slightly. This is
        // a sanity bound, not an exactness guarantee.
        let mut membership_s: Vec<i64> = vec![0; *n];
        let mut rng = Pcg32::seeded(42);
        let out_s = leiden(
            &graph,
            weights.as_deref(),
            Some(&out),
            in_w.as_deref(),
            resolution,
            0.01,
            false,
            -1,
            &mut membership_s,
            &mut rng,
        )
        .unwrap();
        check_membership(&graph, &membership_s, out_s.nb_clusters);
        if outcome.quality.is_finite() && out_s.quality.is_finite() {
            let scale = f64::max(f64::max(outcome.quality.abs(), out_s.quality.abs()), 1.0);
            assert!(
                (outcome.quality - out_s.quality).abs() <= 0.10 * scale,
                "trial {trial}: parallel quality {} far from sequential {}",
                outcome.quality,
                out_s.quality
            );
        }
    }
}

#[test]
fn properties_parallel_thread_pool_determinism() {
    use rayon::ThreadPoolBuilder;

    // A larger graph so that multiple chunks are used with several threads.
    let mut rng = Xorshift(0xC0FFEE);
    let n = 200_usize;
    let mut edges = Vec::new();
    for _ in 0..800 {
        let a = rng.below(n as u64) as i64;
        let b = rng.below(n as u64) as i64;
        edges.push((a, b));
    }
    let graph = Graph::new(n, false, &edges).unwrap();

    let run = |pool: &rayon::ThreadPool| {
        pool.install(|| {
            let mut membership: Vec<i64> = vec![0; n];
            let mut rng = Pcg32::seeded(99);
            let outcome = leiden_simple_parallel(
                &graph,
                None,
                Objective::Modularity,
                1.0,
                0.01,
                false,
                -1,
                &mut membership,
                &mut rng,
            )
            .unwrap();
            (membership, outcome)
        })
    };

    let pool1 = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool4 = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    let (mem1a, out1a) = run(&pool1);
    let (mem1b, _out1b) = run(&pool1);
    let (mem4a, out4a) = run(&pool4);
    let (mem4b, _out4b) = run(&pool4);

    // Deterministic run-to-run within the same thread pool configuration.
    assert_eq!(mem1a, mem1b, "1-thread pool: not deterministic");
    assert_eq!(mem4a, mem4b, "4-thread pool: not deterministic");

    // Every configuration produces a valid partition.
    check_membership(&graph, &mem1a, out1a.nb_clusters);
    check_membership(&graph, &mem4a, out4a.nb_clusters);
}

#[test]
fn properties_start_from_membership() {
    let cases = random_graphs(30);
    for (n, directed, edges, weights) in cases.iter() {
        let graph = Graph::new(*n, *directed, edges).unwrap();
        let mut rng = Pcg32::seeded(7);
        // Arbitrary (valid) starting membership: all singletons, shuffled.
        let mut start: Vec<i64> = (0..*n as i64).collect();
        leiden_rs::shuffle(&mut rng, &mut start);
        let mut membership = start.clone();
        let outcome = leiden_simple(
            &graph,
            weights.as_deref(),
            Objective::Cpm,
            0.1,
            0.01,
            true,
            2,
            &mut membership,
            &mut rng,
        )
        .unwrap();
        check_membership(&graph, &membership, outcome.nb_clusters);
    }
}

#[test]
fn properties_modularity_errors() {
    let graph = Graph::new(3, false, &[(0, 1)]).unwrap();

    // Membership of the wrong length.
    let err = modularity(&graph, &[0, 1], None, 1.0, false).unwrap_err();
    assert_eq!(
        err.message,
        "Membership vector size differs from number of vertices."
    );

    // Negative resolution.
    let err = modularity(&graph, &[0, 1, 1], None, -0.5, false).unwrap_err();
    assert_eq!(
        err.message,
        "The resolution parameter must not be negative."
    );

    // Negative weights.
    let err = modularity(&graph, &[0, 1, 1], Some(&[-1.0]), 1.0, false).unwrap_err();
    assert_eq!(err.message, "Negative weight in weight vector.");

    // Wrong weight vector length.
    let err = modularity(&graph, &[0, 1, 1], Some(&[1.0, 2.0]), 1.0, false).unwrap_err();
    assert_eq!(
        err.message,
        "Weight vector size differs from number of edges."
    );

    // No edges: modularity is NaN.
    let empty = Graph::new(2, false, &[]).unwrap();
    let q = modularity(&empty, &[0, 1], None, 1.0, false).unwrap();
    assert!(q.is_nan());
}
