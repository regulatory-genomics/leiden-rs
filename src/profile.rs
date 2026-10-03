//! Resolution profile: sweeping the resolution parameter over a range and
//! collecting the optimal Leiden partition at each resolution.
//!
//! The resolution parameter γ controls the granularity of the detected
//! communities: lower resolutions merge communities (fewer, larger
//! clusters), higher resolutions split them (more, smaller clusters).
//! Because γ changes the objective function itself, each resolution value
//! requires its own Leiden run — but a sweep does not have to run the
//! algorithm from scratch every time:
//!
//! - **Linear scan** ([`resolution_profile`]): evaluates the partition at
//!   each requested resolution, sweeping from low to high and *warm-starting*
//!   each run from the previous partition (the `start` mechanism of
//!   [`leiden_simple`]). Since a higher resolution favors finer partitions,
//!   the previous partition is already close to the optimum, so subsequent
//!   runs typically converge in far fewer iterations.
//! - **Bisection sweep** ([`resolution_profile_bisect`]): mirrors
//!   leidenalg's `Optimiser.resolution_profile()`. The optimal partition is
//!   piecewise-constant in γ, changing only at a discrete set of jump
//!   points. Instead of scanning a dense grid, the bisection sweep binary-
//!   searches for the resolutions where the partition actually changes,
//!   evaluating only `O(#jump points × log(range/precision))` resolutions.
//!   An interval stops being subdivided when the bisect function (the total
//!   internal edge weight, [`total_internal_edges`]) differs by no more than
//!   `min_diff_bisect_value` between its endpoints, or when the resolution
//!   gap falls to `min_diff_resolution` (logarithmic gap by default, since
//!   jump points cluster near 0 and become sparse at high γ).
//!
//! Note that quality values are *not* comparable across entries: the
//! objective changes with γ (for modularity, the resolution is even
//! internally normalized by the total edge weight). Use
//! [`ProfileEntry::num_communities`], [`ProfileEntry::internal_edges`] or a
//! fixed-γ recomputation (e.g. the [`crate::modularity`] port) to compare
//! partitions across the profile.

use crate::error::{LeidenError, Result};
use crate::graph::Graph;
use crate::leiden::{leiden_simple, Objective};
use crate::rng::Rng;

/// One entry of a resolution profile: the optimal partition at a single
/// resolution value.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileEntry {
    /// The resolution parameter γ at which the partition was optimized.
    pub resolution: f64,
    /// Number of communities in the partition.
    pub num_communities: i64,
    /// Quality of the partition (objective function value at this γ). Not
    /// comparable across entries with different resolutions.
    pub quality: f64,
    /// Total internal edge weight of the partition (the default bisect
    /// function, as in leidenalg's `resolution_profile`).
    pub internal_edges: f64,
    /// Membership vector: community index per vertex.
    pub membership: Vec<i64>,
}

/// Total internal edge weight of a partition: the sum of the weights of the
/// edges whose endpoints lie in the same community. Self-loops are always
/// internal; undirected edges are counted once. This is the default bisect
/// function of leidenalg's `resolution_profile`.
pub fn total_internal_edges(graph: &Graph, membership: &[i64], weights: Option<&[f64]>) -> f64 {
    if let Some(w) = weights {
        debug_assert_eq!(w.len(), graph.ecount() as usize);
    }
    let mut total = 0.0;
    for e in 0..graph.ecount() {
        if membership[graph.from(e) as usize] == membership[graph.to(e) as usize] {
            total += weights.map_or(1.0, |w| w[e as usize]);
        }
    }
    total
}

/// Validate a resolution value, mirroring igraph's error message for
/// negative resolutions (the Leiden code in this crate does not itself
/// reject them for the CPM/ER objectives, so the profile checks upfront).
fn validate_resolution(resolution: f64) -> Result<()> {
    if resolution.is_nan() {
        return Err(LeidenError::new(
            "The resolution parameter must not be NaN.",
        ));
    }
    if resolution < 0.0 {
        return Err(LeidenError::new(
            "The resolution parameter must not be negative.",
        ));
    }
    Ok(())
}

/// Run one profile probe: optimize the partition at `resolution`, warm-
/// starting from `start` when given, and record the entry.
#[allow(clippy::too_many_arguments)]
fn run_at(
    graph: &Graph,
    weights: Option<&[f64]>,
    objective: Objective,
    resolution: f64,
    beta: f64,
    n_iterations: i64,
    start: Option<&[i64]>,
    rng: &mut dyn Rng,
) -> Result<ProfileEntry> {
    let mut membership: Vec<i64> = start.unwrap_or(&[]).to_vec();
    let outcome = leiden_simple(
        graph,
        weights,
        objective,
        resolution,
        beta,
        start.is_some(),
        n_iterations,
        &mut membership,
        rng,
    )?;
    let internal_edges = total_internal_edges(graph, &membership, weights);
    Ok(ProfileEntry {
        resolution,
        num_communities: outcome.nb_clusters,
        quality: outcome.quality,
        internal_edges,
        membership,
    })
}

/// Linear-scan resolution profile.
///
/// Evaluates the optimal Leiden partition at every resolution in
/// `resolutions`, sweeping from low to high resolution and warm-starting
/// each run from the previous partition (so an ascending sweep converges
/// much faster than independent cold starts). Entries are returned in
/// ascending resolution order; duplicate resolutions are kept.
///
/// `n_iterations`: passed through to [`leiden_simple`]; a negative value
/// iterates until the clustering stops changing.
pub fn resolution_profile(
    graph: &Graph,
    weights: Option<&[f64]>,
    objective: Objective,
    resolutions: &[f64],
    beta: f64,
    n_iterations: i64,
    rng: &mut dyn Rng,
) -> Result<Vec<ProfileEntry>> {
    for &r in resolutions {
        validate_resolution(r)?;
    }
    let mut sorted = resolutions.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("resolution validated to be non-NaN"));

    let mut entries = Vec::with_capacity(sorted.len());
    let mut prev: Option<Vec<i64>> = None;
    for &r in &sorted {
        let entry = run_at(
            graph,
            weights,
            objective,
            r,
            beta,
            n_iterations,
            prev.as_deref(),
            rng,
        )?;
        prev = Some(entry.membership.clone());
        entries.push(entry);
    }
    Ok(entries)
}

/// Bisection-based resolution profile, mirroring leidenalg's
/// `Optimiser.resolution_profile()`.
///
/// The optimal partition is piecewise-constant in the resolution γ, so
/// instead of scanning a dense grid this sweep binary-searches the range
/// `(lo, hi)` for the resolutions where the partition actually changes. An
/// interval between two evaluated resolutions is subdivided further only
/// while
///
/// - the bisect function (total internal edge weight,
///   [`total_internal_edges`]) differs between the interval's endpoints by
///   more than `min_diff_bisect_value` (by default a difference of a single
///   edge does not trigger further bisectioning), and
/// - the resolution gap exceeds `min_diff_resolution` (the logarithmic gap
///   `log(hi) - log(lo)` by default, which concentrates the search near 0
///   where jump points cluster; set `linear_bisection` to use the plain
///   difference).
///
/// Every evaluated resolution is returned as a [`ProfileEntry`], sorted by
/// resolution (the endpoints `lo` and `hi` are always included). The number
/// of Leiden runs is bounded by the number of partition changes, so the
/// sweep finds every distinct partition over a wide range at a fraction of
/// the cost of a dense grid — which can miss jump points entirely.
///
/// # Sensible defaults
///
/// | Parameter | Default | Rationale |
/// |---|---|---|
/// | `beta` | `0.01` | igraph's own default for `igraph_community_leiden()`. Higher values make the refinement's split decisions noisy, so consecutive probes of the *same* partition can land in different local optima — the bisection then sees phantom partition "changes" and subdivides far more than necessary. |
/// | `n_iterations` | `-1` | Iterate until the clustering stops changing. A profile is only meaningful if every probe is a *converged* partition; a small fixed count (igraph's example value 2) can leave probes unconverged, blurring or shifting the jump points. |
/// | `min_diff_resolution` | `1e-3` | leidenalg's default. This is the stopping precision on the (logarithmic) gap, so every jump point is localized to within roughly `min_diff_resolution` in relative γ terms. |
/// | `min_diff_bisect_value` | `1.0` | leidenalg's default — a difference of a single internal edge does not trigger further bisectioning. |
/// | `linear_bisection` | `false` | Logarithmic spacing. Jump points cluster near 0 and thin out at high γ, so log spacing places probes where transitions actually occur; linear spacing wastes most of its precision at large γ. |
///
/// # Parameters
///
/// - `graph`, `weights`, `objective`: passed through to [`leiden_simple`]
///   for every probe; `weights` must have length `graph.ecount()`.
/// - `resolution_range = (lo, hi)`: the swept range, with `0 <= lo < hi`.
///   Wide log-spaced ranges such as `(0.01, 10.0)` are cheap — the number
///   of Leiden runs grows like `O(#jump points × log(range /
///   min_diff_resolution))`, so tightening `min_diff_resolution` by an
///   order of magnitude costs only a few extra runs per jump point.
/// - `beta`: refinement randomness, see the defaults table above.
/// - `n_iterations`: passed through to [`leiden_simple`]; negative means
///   iterate until stable.
/// - `min_diff_resolution`: an interval is no longer subdivided once its
///   resolution gap (logarithmic, or linear with `linear_bisection`) falls
///   to or below this value. Must be positive and finite.
/// - `min_diff_bisect_value`: an interval is no longer subdivided once the
///   bisect function differs by at most this much between its endpoints.
///   Tune it for the sharpness/noise trade-off you want: `0.0` subdivides
///   until the internal edge count is identical everywhere (sharpest
///   profile, most probes), while a larger value such as √(total edge
///   weight) treats small partition tweaks as noise and terminates sooner.
///   Must not be negative.
/// - `linear_bisection`: `false` (default) bisects on the logarithmic gap
///   and uses geometric midpoints; `true` bisects on the plain difference
///   with arithmetic midpoints. The lower bound `lo = 0` has no geometric
///   midpoint, so the arithmetic midpoint is used for the first split of
///   the interval touching 0 regardless of this flag.
///
/// # Determinism
///
/// The sweep is fully deterministic: for the same seed and inputs, the
/// probe sequence and hence the entire profile (resolutions, memberships,
/// qualities) are identical run-to-run. Changing the seed can move a jump
/// point within `min_diff_resolution` or change the partition found at
/// probes very close to a jump point — the usual Leiden local-optimum
/// sensitivity, not a profile-specific effect.
///
/// # Examples
///
/// ```
/// use leiden_rs::{resolution_profile_bisect, Graph, Objective, Pcg32};
///
/// // Two triangles joined by a bridge edge.
/// let graph = Graph::new(
///     6,
///     false,
///     &[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (0, 3)],
/// )
/// .unwrap();
///
/// let mut rng = Pcg32::seeded(42);
/// let profile = resolution_profile_bisect(
///     &graph,
///     None,           // edge weights (None = unweighted)
///     Objective::Cpm, // objective function
///     (0.01, 10.0),   // resolution range (low, high)
///     0.01,           // beta: igraph's default
///     -1,             // n_iterations: iterate until stable
///     1e-3,           // min_diff_resolution: bisection precision
///     1.0,            // min_diff_bisect_value: one edge does not trigger
///     false,          // linear_bisection: false = logarithmic
///     &mut rng,
/// )
/// .unwrap();
///
/// // One merged community at low resolution, the two triangles apart in
/// // the middle of the range, and singletons at high resolution.
/// assert_eq!(profile[0].num_communities, 1);
/// assert!(profile.last().unwrap().num_communities > 1);
///
/// for entry in &profile {
///     println!("γ={:.3}: {} communities", entry.resolution, entry.num_communities);
/// }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn resolution_profile_bisect(
    graph: &Graph,
    weights: Option<&[f64]>,
    objective: Objective,
    resolution_range: (f64, f64),
    beta: f64,
    n_iterations: i64,
    min_diff_resolution: f64,
    min_diff_bisect_value: f64,
    linear_bisection: bool,
    rng: &mut dyn Rng,
) -> Result<Vec<ProfileEntry>> {
    let (lo, hi) = resolution_range;
    validate_resolution(lo)?;
    validate_resolution(hi)?;
    if lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) {
        return Err(LeidenError::new(
            "Resolution range should be provided from low to high.",
        ));
    }
    if !min_diff_resolution.is_finite() || min_diff_resolution <= 0.0 {
        return Err(LeidenError::new(
            "The minimum resolution difference must be positive and finite.",
        ));
    }
    if !min_diff_bisect_value.is_finite() || min_diff_bisect_value < 0.0 {
        return Err(LeidenError::new(
            "The minimum bisect value difference must not be negative and must be finite.",
        ));
    }

    // Endpoints first; the high-resolution run warm-starts from the
    // low-resolution partition.
    let first = run_at(graph, weights, objective, lo, beta, n_iterations, None, rng)?;
    let last = run_at(
        graph,
        weights,
        objective,
        hi,
        beta,
        n_iterations,
        Some(&first.membership),
        rng,
    )?;
    let mut entries: Vec<ProfileEntry> = vec![first, last];

    // Safety cap: a profile can never need more probes than one per
    // possible partition change; 10 000 is far beyond any practical sweep
    // and guards against pathological stopping criteria.
    const MAX_ENTRIES: usize = 10_000;

    loop {
        if entries.len() >= MAX_ENTRIES {
            return Err(LeidenError::new(
                "Resolution profile exceeded the maximum number of partitions; increase min_diff_resolution or min_diff_bisect_value.",
            ));
        }

        // Find the qualifying adjacent pair with the largest resolution gap.
        let mut best: Option<(usize, f64)> = None;
        for i in 0..entries.len() - 1 {
            let a = &entries[i];
            let b = &entries[i + 1];
            // A difference within min_diff_bisect_value does not trigger
            // further bisectioning (leidenalg: a single edge does not, by
            // default).
            if (a.internal_edges - b.internal_edges).abs() <= min_diff_bisect_value {
                continue;
            }
            let gap = if linear_bisection {
                b.resolution - a.resolution
            } else if a.resolution > 0.0 {
                b.resolution.ln() - a.resolution.ln()
            } else {
                // log(0) is -inf: treat the gap as unbounded so the interval
                // still gets subdivided (from its arithmetic midpoint, see
                // bisect_midpoint).
                f64::INFINITY
            };
            if gap <= min_diff_resolution {
                continue;
            }
            if best.is_none_or(|(_, g)| gap > g) {
                best = Some((i, gap));
            }
        }

        let Some((i, _)) = best else {
            break;
        };
        let (lo_r, hi_r) = (entries[i].resolution, entries[i + 1].resolution);
        let mid = bisect_midpoint(lo_r, hi_r, linear_bisection);
        if mid <= lo_r || mid >= hi_r {
            // The interval cannot be subdivided further in floating point;
            // this cannot happen for a sane min_diff_resolution, but guard
            // against stalling all the same.
            break;
        }
        let entry = run_at(
            graph,
            weights,
            objective,
            mid,
            beta,
            n_iterations,
            Some(&entries[i].membership),
            rng,
        )?;
        entries.insert(i + 1, entry);
    }

    Ok(entries)
}

/// Midpoint of an interval: arithmetic for linear bisection, geometric for
/// logarithmic bisection (falling back to arithmetic when the lower bound
/// is 0, where a geometric mean is undefined).
fn bisect_midpoint(lo: f64, hi: f64, linear_bisection: bool) -> f64 {
    if linear_bisection || lo <= 0.0 {
        (lo + hi) / 2.0
    } else {
        (lo * hi).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pcg32;

    /// Two triangles joined by a single bridge edge.
    fn bridge_graph() -> Graph {
        Graph::new(
            6,
            false,
            &[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (0, 3)],
        )
        .unwrap()
    }

    #[test]
    fn total_internal_edges_hand_computed() {
        let graph = bridge_graph();
        let membership = [0, 0, 0, 1, 1, 1];
        assert_eq!(total_internal_edges(&graph, &membership, None), 6.0);

        // The singleton partition has no internal edges.
        assert_eq!(
            total_internal_edges(&graph, &[0, 1, 2, 3, 4, 5], None),
            0.0
        );

        // The everything-in-one-community partition is all 7 edges.
        assert_eq!(total_internal_edges(&graph, &[0; 6], None), 7.0);

        // Weighted: the bridge edge has weight 10, the rest 1.
        let weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0];
        assert_eq!(total_internal_edges(&graph, &membership, Some(&weights)), 6.0);
        assert_eq!(
            total_internal_edges(&graph, &[0; 6], Some(&weights)),
            16.0
        );
    }

    #[test]
    fn linear_scan_matches_manual_warm_started_calls() {
        let graph = bridge_graph();

        // Profile of two resolutions ...
        let mut rng = Pcg32::seeded(123);
        let profile = resolution_profile(
            &graph,
            None,
            Objective::Cpm,
            &[0.2, 1.0],
            0.01,
            -1,
            &mut rng,
        )
        .unwrap();

        // ... must match manual leiden_simple calls with the same warm-start
        // chaining and the same RNG stream.
        let mut membership: Vec<i64> = Vec::new();
        let mut rng = Pcg32::seeded(123);
        let first = leiden_simple(
            &graph, None, Objective::Cpm, 0.2, 0.01, false, -1, &mut membership, &mut rng,
        )
        .unwrap();
        let second = leiden_simple(
            &graph, None, Objective::Cpm, 1.0, 0.01, true, -1, &mut membership, &mut rng,
        )
        .unwrap();

        assert_eq!(profile.len(), 2);
        assert_eq!(profile[0].resolution, 0.2);
        assert_eq!(profile[0].num_communities, first.nb_clusters);
        assert_eq!(profile[0].quality, first.quality);
        assert_eq!(profile[0].membership, membership[..].to_vec());
        assert_eq!(profile[1].num_communities, second.nb_clusters);
        assert_eq!(profile[1].quality, second.quality);
    }

    #[test]
    fn linear_scan_is_deterministic_and_sorted() {
        let graph = bridge_graph();
        let resolutions = [2.0, 0.1, 0.5, 0.1, 1.0];

        let mut rng = Pcg32::seeded(42);
        let a = resolution_profile(&graph, None, Objective::Cpm, &resolutions, 0.01, -1, &mut rng)
            .unwrap();
        let mut rng = Pcg32::seeded(42);
        let b = resolution_profile(&graph, None, Objective::Cpm, &resolutions, 0.01, -1, &mut rng)
            .unwrap();

        assert_eq!(a, b);

        // Ascending order, duplicates kept.
        let got: Vec<f64> = a.iter().map(|e| e.resolution).collect();
        assert_eq!(got, vec![0.1, 0.1, 0.5, 1.0, 2.0]);

        // Membership validity: consecutive cluster indices.
        for entry in &a {
            let max = *entry.membership.iter().max().unwrap();
            assert_eq!(max + 1, entry.num_communities);
            assert!(entry.membership.iter().all(|&m| m >= 0));
        }
    }

    #[test]
    fn cpm_sweep_refines_with_resolution() {
        let graph = bridge_graph();

        // At very low resolution the bridge pulls both triangles together
        // (the CPM switches to two communities only above γ = 1/9); at very
        // high resolution everything splits apart into singletons.
        let mut rng = Pcg32::seeded(7);
        let profile = resolution_profile(
            &graph,
            None,
            Objective::Cpm,
            &[0.001, 0.5, 1.0, 10.0],
            0.01,
            -1,
            &mut rng,
        )
        .unwrap();

        assert!(profile[0].num_communities < profile[3].num_communities);
        // The intermediate resolutions keep the two triangles separate.
        assert_eq!(profile[1].num_communities, 2);
        assert_eq!(profile[2].num_communities, 2);
        assert_eq!(profile[3].num_communities, 6);
    }

    #[test]
    fn bisect_finds_transition_with_few_probes() {
        let graph = bridge_graph();

        let mut rng = Pcg32::seeded(99);
        let profile = resolution_profile_bisect(
            &graph,
            None,
            Objective::Cpm,
            (0.001, 10.0),
            0.01,
            -1,
            1e-3, // min_diff_resolution
            1.0,  // min_diff_bisect_value: a single edge does not trigger
            false,
            &mut rng,
        )
        .unwrap();

        // Endpoints included, strictly ascending resolutions.
        assert_eq!(profile.first().unwrap().resolution, 0.001);
        assert_eq!(profile.last().unwrap().resolution, 10.0);
        assert!(profile.windows(2).all(|w| w[0].resolution < w[1].resolution));

        // A dense grid over this range at the same precision would need
        // ~10^4 probes; the bisection needs only a handful.
        assert!(profile.len() < 20);

        // The transition from one community (low γ) to more communities
        // (high γ) is captured.
        assert!(profile[0].num_communities < profile.last().unwrap().num_communities);

        // Internal edges are recomputed consistently.
        for entry in &profile {
            assert_eq!(
                entry.internal_edges,
                total_internal_edges(&graph, &entry.membership, None)
            );
        }
    }

    #[test]
    fn bisect_stops_when_partition_is_constant() {
        // Two clearly separated triangles: over a range where the partition
        // never changes, the bisection must terminate after the two
        // endpoint probes only.
        let graph = Graph::new(
            6,
            false,
            &[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)],
        )
        .unwrap();

        let mut rng = Pcg32::seeded(5);
        let profile = resolution_profile_bisect(
            &graph,
            None,
            Objective::Cpm,
            (0.5, 1.0),
            0.01,
            -1,
            1e-3,
            1.0,
            false,
            &mut rng,
        )
        .unwrap();

        assert_eq!(profile.len(), 2);
        for entry in &profile {
            assert_eq!(entry.num_communities, 2);
        }
    }

    #[test]
    fn bisect_linear_mode_terminates() {
        let graph = bridge_graph();

        let mut rng = Pcg32::seeded(11);
        let profile = resolution_profile_bisect(
            &graph,
            None,
            Objective::Cpm,
            (0.0, 2.0),
            0.01,
            -1,
            1e-3,
            0.0, // keep bisecting while the bisect value differs at all
            true, // linear bisection (also covers the lo = 0 edge case)
            &mut rng,
        )
        .unwrap();

        assert!(profile.len() >= 2);
        assert!(profile.windows(2).all(|w| w[0].resolution < w[1].resolution));
        assert_eq!(profile[0].resolution, 0.0);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let graph = bridge_graph();
        let mut rng = Pcg32::seeded(1);

        // Negative resolution.
        let err = resolution_profile(&graph, None, Objective::Cpm, &[-0.5], 0.01, -1, &mut rng)
            .unwrap_err();
        assert_eq!(err.message, "The resolution parameter must not be negative.");

        // NaN resolution.
        let err = resolution_profile(&graph, None, Objective::Cpm, &[f64::NAN], 0.01, -1, &mut rng)
            .unwrap_err();
        assert_eq!(err.message, "The resolution parameter must not be NaN.");

        // Reversed range.
        let err = resolution_profile_bisect(
            &graph, None, Objective::Cpm, (1.0, 0.5), 0.01, -1, 1e-3, 1.0, false, &mut rng,
        )
        .unwrap_err();
        assert_eq!(err.message, "Resolution range should be provided from low to high.");

        // Empty range.
        let err = resolution_profile_bisect(
            &graph, None, Objective::Cpm, (0.5, 0.5), 0.01, -1, 1e-3, 1.0, false, &mut rng,
        )
        .unwrap_err();
        assert_eq!(err.message, "Resolution range should be provided from low to high.");

        // Non-positive min_diff_resolution.
        let err = resolution_profile_bisect(
            &graph, None, Objective::Cpm, (0.5, 1.0), 0.01, -1, 0.0, 1.0, false, &mut rng,
        )
        .unwrap_err();
        assert_eq!(
            err.message,
            "The minimum resolution difference must be positive and finite."
        );

        // Negative min_diff_bisect_value.
        let err = resolution_profile_bisect(
            &graph, None, Objective::Cpm, (0.5, 1.0), 0.01, -1, 1e-3, -1.0, false, &mut rng,
        )
        .unwrap_err();
        assert_eq!(
            err.message,
            "The minimum bisect value difference must not be negative and must be finite."
        );
    }
}
