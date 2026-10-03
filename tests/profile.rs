//! Integration tests for the resolution profile (public API).
//!
//! - Linear scan: entries sorted, valid memberships, deterministic.
//! - Bisection sweep: finds the partition transitions over a wide γ range
//!   with far fewer probes than a dense grid.
//! - Cross-check: for the modularity objective, entry quality matches the
//!   independent `modularity()` port at the same resolution.

use leiden_rs::{
    modularity, resolution_profile, resolution_profile_bisect, total_internal_edges, Graph,
    Objective, Pcg32,
};

/// Karate-club-like test graph: two dense groups with a few bridges.
fn two_block_graph() -> Graph {
    // Group A: 0..5, group B: 5..10; dense within, one bridge (4, 5).
    let mut edges = Vec::new();
    for a in 0..5 {
        for b in (a + 1)..5 {
            edges.push((a, b));
        }
    }
    for a in 5..10 {
        for b in (a + 1)..10 {
            edges.push((a, b));
        }
    }
    edges.push((4, 5));
    Graph::new(10, false, &edges).unwrap()
}

#[test]
fn linear_profile_sweeps_ascending() {
    let graph = two_block_graph();
    let resolutions = [4.0, 0.01, 0.5, 2.0];

    let mut rng = Pcg32::seeded(1234);
    let profile =
        resolution_profile(&graph, None, Objective::Cpm, &resolutions, 0.01, -1, &mut rng).unwrap();

    assert_eq!(profile.len(), resolutions.len());
    let got: Vec<f64> = profile.iter().map(|e| e.resolution).collect();
    assert_eq!(got, vec![0.01, 0.5, 2.0, 4.0]);

    // Valid memberships: consecutive cluster indices, and the entry's
    // internal edge count matches an independent recomputation.
    for entry in &profile {
        assert_eq!(entry.membership.len(), 10);
        let max = *entry.membership.iter().max().unwrap();
        assert_eq!(max + 1, entry.num_communities);
        assert_eq!(
            entry.internal_edges,
            total_internal_edges(&graph, &entry.membership, None)
        );
    }

    // Low resolution merges the two blocks; high resolution splits them.
    assert!(profile[0].num_communities < profile[3].num_communities);
}

#[test]
fn bisect_profile_covers_range_efficiently() {
    let graph = two_block_graph();

    let mut rng = Pcg32::seeded(42);
    let profile = resolution_profile_bisect(
        &graph,
        None,
        Objective::Cpm,
        (0.01, 10.0),
        0.01,
        -1,
        1e-3, // min_diff_resolution
        1.0,  // min_diff_bisect_value
        false,
        &mut rng,
    )
    .unwrap();

    // Endpoints included, strictly ascending, few probes relative to the
    // ~10^4 resolutions a dense grid at the same precision would scan.
    assert_eq!(profile[0].resolution, 0.01);
    assert_eq!(profile.last().unwrap().resolution, 10.0);
    assert!(profile.windows(2).all(|w| w[0].resolution < w[1].resolution));
    assert!(profile.len() < 30);

    // Monotone refinement across the sweep: communities only increase.
    assert!(profile
        .windows(2)
        .all(|w| w[0].num_communities <= w[1].num_communities));
}

#[test]
fn modularity_profile_quality_matches_modularity_port() {
    let graph = two_block_graph();

    let mut rng = Pcg32::seeded(7);
    let profile =
        resolution_profile(&graph, None, Objective::Modularity, &[0.5, 1.0], 0.01, -1, &mut rng)
            .unwrap();

    for entry in &profile {
        let q = modularity(&graph, &entry.membership, None, entry.resolution, false).unwrap();
        // Same tolerance as tests/properties.rs: the quality is accumulated
        // in a different order inside the core loop than by the standalone
        // modularity() port.
        assert!((entry.quality - q).abs() <= 1e-15 * f64::max(q.abs(), 1.0));
    }
}
