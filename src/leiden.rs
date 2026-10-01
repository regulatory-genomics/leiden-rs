//! The Leiden algorithm, ported from igraph's `src/community/leiden.c`
//! (commit `8225c3a4b`, which includes the fix for keeping the moves of the
//! last aggregation level).
//!
//! Every function is a faithful translation of its C counterpart; comments
//! give the C source line ranges. The random number stream matches C igraph
//! exactly (see [`crate::rng`]), so identical seeds and inputs produce
//! identical memberships and quality values.

use crate::error::{format_g, LeidenError, Result};
use crate::graph::{Graph, Inclist};
use crate::rng::{Pcg32, Rng};
use crate::Outcome;

use rayon::prelude::*;

/// Objective function for [`leiden_simple`], mirroring
/// `igraph_leiden_objective_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Generalized modularity with a multigraph configuration model null
    /// model. Edge weights must not be negative.
    Modularity,
    /// Constant Potts Model. Edge weights may be negative; edge directions
    /// have no impact on the result.
    Cpm,
    /// Erdős–Rényi G(n, p) null model based on the weighted density. Edge
    /// weights must not be negative; edge directions have no impact.
    Er,
}

/// Port of `leiden_fastmove_vertices()` (`leiden.c` lines 50-240):
/// move vertices in order to improve the quality of a partition.
///
/// Each vertex is greedily moved to a neighboring community that maximizes
/// the improvement in the quality of the partition; only strictly improving
/// moves are considered. Vertices are examined via a queue, initialized in a
/// random order; only neighbors of moved vertices are re-queued. The
/// membership vector is updated in place.
#[allow(clippy::too_many_arguments)]
/// Reusable scratch buffers for the Leiden core loop, passed down to the
/// hot refinement functions so that per-call allocations become per-run
/// allocations (mirrors the scratch reuse of igraph's C code, which keeps
/// its vectors alive across iterations).
#[derive(Default)]
struct Workspace {
    // Refined membership, written by merge_vertices via the workspace (all
    // entries are rewritten by the singleton initialization in each call).
    mg_refined_membership: Vec<i64>,
    // fastmove_vertices scratch (indexed by vertex ID).
    fm_vertex_is_stable: Vec<bool>,
    fm_queue: std::collections::VecDeque<i64>,
    fm_vertex_order: Vec<i64>,
    fm_cluster_out_weights: Vec<f64>,
    fm_cluster_in_weights: Vec<f64>,
    fm_nb_vertices_per_cluster: Vec<i64>,
    fm_empty_clusters: Vec<i64>,
    fm_edge_weights_per_cluster: Vec<f64>,
    fm_neighbor_cluster_added: Vec<bool>,
    fm_neighbor_clusters: Vec<i64>,
    // merge_vertices scratch (indexed by position in the vertex subset).
    mg_vertex_order: Vec<i64>,
    mg_cluster_out_weights: Vec<f64>,
    mg_cluster_in_weights: Vec<f64>,
    mg_nb_vertices_per_cluster: Vec<i64>,
    mg_external_edge_weight: Vec<f64>,
    mg_non_singleton_cluster: Vec<bool>,
    mg_edge_weights_per_cluster: Vec<f64>,
    mg_neighbor_cluster_added: Vec<bool>,
    mg_neighbor_clusters: Vec<i64>,
    mg_cum_trans_diff: Vec<f64>,
    mg_new_cluster: Vec<i64>,
}

/// Ensure `v` has at least `n` elements and zero the first `n`.
fn zero_f64(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    v[..n].fill(0.0);
}

/// Ensure `v` has at least `n` elements and zero the first `n`.
fn zero_i64(v: &mut Vec<i64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0);
    }
    v[..n].fill(0);
}

/// Ensure `v` has at least `n` elements and reset the first `n` to `false`.
fn zero_bool(v: &mut Vec<bool>, n: usize) {
    if v.len() < n {
        v.resize(n, false);
    }
    v[..n].fill(false);
}

#[allow(clippy::too_many_arguments)]
fn fastmove_vertices(
    graph: &Graph,
    edges_per_vertex: &Inclist,
    edge_weights: &[f64],
    vertex_out_weights: &[f64],
    vertex_in_weights: Option<&[f64]>,
    resolution: f64,
    nb_clusters: &mut i64,
    membership: &mut [i64],
    changed: &mut bool,
    rng: &mut dyn Rng,
    ws: &mut Workspace,
) -> Result<()> {
    let n = graph.vcount() as usize;
    let directed = vertex_in_weights.is_some();
    let vertex_in_weights = vertex_in_weights.unwrap_or(vertex_out_weights);

    // Queue of unstable vertices and whether a vertex is stable. Only
    // unstable vertices are in the queue.
    zero_bool(&mut ws.fm_vertex_is_stable, n);
    ws.fm_queue.clear();
    let unstable_vertices = &mut ws.fm_queue;

    // Shuffle vertices and add them to the queue.
    zero_i64(&mut ws.fm_vertex_order, n);
    for (i, item) in ws.fm_vertex_order[..n].iter_mut().enumerate() {
        *item = i as i64;
    }
    crate::rng::shuffle(rng, &mut ws.fm_vertex_order[..n]);
    for &v in &ws.fm_vertex_order[..n] {
        unstable_vertices.push_back(v);
    }

    // Initialize cluster weights and nb vertices.
    zero_f64(&mut ws.fm_cluster_out_weights, n);
    let cluster_out_weights: &mut [f64] = &mut ws.fm_cluster_out_weights[..n];
    if directed {
        zero_f64(&mut ws.fm_cluster_in_weights, n);
    }
    let cluster_in_weights: &mut [f64] = if directed {
        &mut ws.fm_cluster_in_weights[..n]
    } else {
        &mut []
    };
    zero_i64(&mut ws.fm_nb_vertices_per_cluster, n);
    let nb_vertices_per_cluster: &mut [i64] = &mut ws.fm_nb_vertices_per_cluster[..n];
    for i in 0..n {
        let c = membership[i] as usize;
        cluster_out_weights[c] += vertex_out_weights[i];
        if directed {
            cluster_in_weights[c] += vertex_in_weights[i];
        }
        nb_vertices_per_cluster[c] += 1;
    }

    // Initialize empty clusters.
    zero_i64(&mut ws.fm_empty_clusters, n);
    let mut nb_empty = 0_usize;
    for (c, count) in nb_vertices_per_cluster.iter().enumerate() {
        if *count == 0 {
            ws.fm_empty_clusters[nb_empty] = c as i64;
            nb_empty += 1;
        }
    }

    // Vectors used in calculating differences.
    zero_f64(&mut ws.fm_edge_weights_per_cluster, n);
    let edge_weights_per_cluster: &mut [f64] = &mut ws.fm_edge_weights_per_cluster[..n];
    zero_bool(&mut ws.fm_neighbor_cluster_added, n);
    let neighbor_cluster_added: &mut [bool] = &mut ws.fm_neighbor_cluster_added[..n];
    zero_i64(&mut ws.fm_neighbor_clusters, n);
    let neighbor_clusters: &mut [i64] = &mut ws.fm_neighbor_clusters[..n];
    let vertex_is_stable: &mut [bool] = &mut ws.fm_vertex_is_stable[..n];

    // Iterate while the queue is not empty.
    while let Some(v) = unstable_vertices.pop_front() {
        let v = v as usize;
        let current_cluster = membership[v] as usize;

        // Remove vertex from current cluster.
        cluster_out_weights[current_cluster] -= vertex_out_weights[v];
        if directed {
            cluster_in_weights[current_cluster] -= vertex_in_weights[v];
        }
        nb_vertices_per_cluster[current_cluster] -= 1;
        if nb_vertices_per_cluster[current_cluster] == 0 {
            ws.fm_empty_clusters[nb_empty] = current_cluster as i64;
            nb_empty += 1;
        }

        // Find out neighboring clusters.
        let mut c = ws.fm_empty_clusters[nb_empty - 1] as usize;
        neighbor_clusters[0] = c as i64;
        neighbor_cluster_added[c] = true;
        let mut nb_neigh_clusters = 1_usize;

        // Determine the edge weight to each neighboring cluster.
        let (edges, neighbors) = edges_per_vertex.edges_with_neighbors(v as i64);
        for (&e, &u) in edges.iter().zip(neighbors) {
            let u = u as usize;
            if u != v {
                c = membership[u] as usize;
                if !neighbor_cluster_added[c] {
                    neighbor_cluster_added[c] = true;
                    neighbor_clusters[nb_neigh_clusters] = c as i64;
                    nb_neigh_clusters += 1;
                }
                edge_weights_per_cluster[c] += edge_weights[e as usize];
            }
        }

        // Calculate maximum diff.
        let mut best_cluster = current_cluster;
        let mut max_diff = edge_weights_per_cluster[current_cluster];
        if directed {
            max_diff -= (vertex_in_weights[v] * cluster_out_weights[current_cluster]
                + vertex_out_weights[v] * cluster_in_weights[current_cluster])
                * resolution;
        } else {
            max_diff -= vertex_out_weights[v] * cluster_out_weights[current_cluster] * resolution;
        }
        for &nc in &neighbor_clusters[..nb_neigh_clusters] {
            c = nc as usize;
            let mut diff = edge_weights_per_cluster[c];
            if directed {
                diff -= (vertex_out_weights[v] * cluster_in_weights[c]
                    + vertex_in_weights[v] * cluster_out_weights[c])
                    * resolution;
            } else {
                diff -= vertex_out_weights[v] * cluster_out_weights[c] * resolution;
            }
            // Only consider strictly improving moves. Note that this is
            // important in considering convergence.
            if diff > max_diff {
                best_cluster = c;
                max_diff = diff;
            }
            edge_weights_per_cluster[c] = 0.0;
            neighbor_cluster_added[c] = false;
        }

        // Move vertex to best cluster.
        cluster_out_weights[best_cluster] += vertex_out_weights[v];
        if directed {
            cluster_in_weights[best_cluster] += vertex_in_weights[v];
        }
        nb_vertices_per_cluster[best_cluster] += 1;
        if best_cluster as i64 == ws.fm_empty_clusters[nb_empty - 1] {
            nb_empty -= 1;
        }

        // Mark vertex as stable.
        vertex_is_stable[v] = true;

        // Add stable neighbours that are not part of the new cluster to the queue.
        if best_cluster != current_cluster {
            *changed = true;
            membership[v] = best_cluster as i64;

            for &u in neighbors {
                let u = u as usize;
                if vertex_is_stable[u] && membership[u] != best_cluster as i64 {
                    unstable_vertices.push_back(u as i64);
                    vertex_is_stable[u] = false;
                }
            }
        }
    }

    *nb_clusters = reindex_membership(membership, None);

    Ok(())
}

/// Port of `leiden_clean_refined_membership()` (`leiden.c` lines 251-286):
/// renumber the clusters of the vertices in `vertex_subset` consecutively,
/// starting from `nb_refined_clusters`.
fn clean_refined_membership(
    vertex_subset: &[i64],
    refined_membership: &mut [i64],
    nb_refined_clusters: &mut i64,
    new_cluster: &mut [i64],
) {
    let n = vertex_subset.len();
    debug_assert!(new_cluster.len() >= n);

    *nb_refined_clusters += 1;
    for &v in vertex_subset {
        let c = refined_membership[v as usize] as usize;
        if new_cluster[c] == 0 {
            new_cluster[c] = *nb_refined_clusters;
            *nb_refined_clusters += 1;
        }
    }

    // Assign new cluster.
    for &v in vertex_subset {
        let c = refined_membership[v as usize] as usize;
        refined_membership[v as usize] = new_cluster[c] - 1;
    }
    *nb_refined_clusters -= 1;
}

/// Port of `leiden_merge_vertices()` (`leiden.c` lines 318-544): merge
/// vertices for a subset of the vertices, used to refine a partition.
///
/// All vertices in `vertex_subset` are initialized to a singleton partition
/// in `refined_membership`. Only singleton clusters can be merged if they
/// are sufficiently well connected to the current subgraph induced by
/// `vertex_subset`. The cluster to merge with is chosen randomly among all
/// possibilities that do not decrease the quality, with probability
/// proportional to exp(diff/beta).
#[allow(clippy::too_many_arguments)]
fn merge_vertices(
    edges_per_vertex: &Inclist,
    edge_weights: &[f64],
    vertex_out_weights: &[f64],
    vertex_in_weights: Option<&[f64]>,
    vertex_subset: &[i64],
    membership: &[i64],
    cluster_subset: i64,
    resolution: f64,
    beta: f64,
    nb_refined_clusters: &mut i64,
    rng: &mut dyn Rng,
    ws: &mut Workspace,
) -> Result<()> {
    let directed = vertex_in_weights.is_some();
    let vertex_in_weights = vertex_in_weights.unwrap_or(vertex_out_weights);
    let n = vertex_subset.len();
    let refined_membership: &mut [i64] = &mut ws.mg_refined_membership[..];

    // Initialize cluster weights.
    zero_f64(&mut ws.mg_cluster_out_weights, n);
    let cluster_out_weights: &mut [f64] = &mut ws.mg_cluster_out_weights[..n];
    if directed {
        zero_f64(&mut ws.mg_cluster_in_weights, n);
    }
    let cluster_in_weights: &mut [f64] = if directed {
        &mut ws.mg_cluster_in_weights[..n]
    } else {
        &mut []
    };

    // Initialize number of vertices per cluster.
    zero_i64(&mut ws.mg_nb_vertices_per_cluster, n);
    let nb_vertices_per_cluster: &mut [i64] = &mut ws.mg_nb_vertices_per_cluster[..n];

    // Initialize external edge weight per cluster in subset.
    zero_f64(&mut ws.mg_external_edge_weight, n);
    let external_edge_weight_per_cluster_in_subset: &mut [f64] =
        &mut ws.mg_external_edge_weight[..n];

    // Initialize administration for a singleton partition.
    let mut total_vertex_out_weight = 0.0_f64;
    let mut total_vertex_in_weight = 0.0_f64;
    for i in 0..n {
        let v = vertex_subset[i] as usize;
        refined_membership[v] = i as i64;
        cluster_out_weights[i] += vertex_out_weights[v];
        total_vertex_out_weight += vertex_out_weights[v];
        if directed {
            cluster_in_weights[i] += vertex_in_weights[v];
            total_vertex_in_weight += vertex_in_weights[v];
        }
        nb_vertices_per_cluster[i] += 1;

        // Find out neighboring clusters.
        let (edges, neighbors) = edges_per_vertex.edges_with_neighbors(v as i64);
        for (&e, &u) in edges.iter().zip(neighbors) {
            let u = u as usize;
            if u != v && membership[u] == cluster_subset {
                external_edge_weight_per_cluster_in_subset[i] += edge_weights[e as usize];
            }
        }
    }

    // Shuffle vertices.
    if ws.mg_vertex_order.len() < n {
        ws.mg_vertex_order.resize(n, 0);
    }
    ws.mg_vertex_order[..n].copy_from_slice(vertex_subset);
    crate::rng::shuffle(rng, &mut ws.mg_vertex_order[..n]);
    let vertex_order: &[i64] = &ws.mg_vertex_order[..n];

    // Initialize non singleton clusters.
    zero_bool(&mut ws.mg_non_singleton_cluster, n);
    let non_singleton_cluster: &mut [bool] = &mut ws.mg_non_singleton_cluster[..n];

    // Vectors used in calculating differences.
    zero_f64(&mut ws.mg_edge_weights_per_cluster, n);
    let edge_weights_per_cluster: &mut [f64] = &mut ws.mg_edge_weights_per_cluster[..n];
    zero_bool(&mut ws.mg_neighbor_cluster_added, n);
    let neighbor_cluster_added: &mut [bool] = &mut ws.mg_neighbor_cluster_added[..n];
    zero_i64(&mut ws.mg_neighbor_clusters, n);
    let neighbor_clusters: &mut [i64] = &mut ws.mg_neighbor_clusters[..n];

    // Initialize cumulative transformed difference.
    zero_f64(&mut ws.mg_cum_trans_diff, n);
    let cum_trans_diff: &mut [f64] = &mut ws.mg_cum_trans_diff[..n];

    for &vi in vertex_order.iter() {
        let v = vi as usize;
        let current_cluster = refined_membership[v] as usize;

        let mut vertex_weight_prod = if directed {
            cluster_out_weights[current_cluster]
                * (total_vertex_in_weight - cluster_in_weights[current_cluster])
                + cluster_in_weights[current_cluster]
                    * (total_vertex_out_weight - cluster_out_weights[current_cluster])
        } else {
            cluster_out_weights[current_cluster]
                * (total_vertex_out_weight - cluster_out_weights[current_cluster])
        };

        if !non_singleton_cluster[current_cluster]
            && external_edge_weight_per_cluster_in_subset[current_cluster]
                >= vertex_weight_prod * resolution
        {
            // Remove vertex from current cluster, which is then a singleton
            // by definition.
            cluster_out_weights[current_cluster] = 0.0;
            if directed {
                cluster_in_weights[current_cluster] = 0.0;
            }
            nb_vertices_per_cluster[current_cluster] = 0;

            // Find out neighboring clusters.
            let (edges, neighbors) = edges_per_vertex.edges_with_neighbors(v as i64);

            // Also add current cluster to ensure it can be chosen.
            neighbor_clusters[0] = current_cluster as i64;
            neighbor_cluster_added[current_cluster] = true;
            let mut nb_neigh_clusters = 1_usize;
            for (&e, &u) in edges.iter().zip(neighbors) {
                let u = u as usize;
                if u != v && membership[u] == cluster_subset {
                    let c = refined_membership[u] as usize;
                    if !neighbor_cluster_added[c] {
                        neighbor_cluster_added[c] = true;
                        neighbor_clusters[nb_neigh_clusters] = c as i64;
                        nb_neigh_clusters += 1;
                    }
                    edge_weights_per_cluster[c] += edge_weights[e as usize];
                }
            }

            // Calculate diffs.
            let mut best_cluster = current_cluster;
            let mut max_diff = 0.0_f64;
            let mut total_cum_trans_diff = 0.0_f64;
            for j in 0..nb_neigh_clusters {
                let c = neighbor_clusters[j] as usize;

                vertex_weight_prod = if directed {
                    cluster_out_weights[c] * (total_vertex_in_weight - cluster_in_weights[c])
                        + cluster_in_weights[c] * (total_vertex_out_weight - cluster_out_weights[c])
                } else {
                    cluster_out_weights[c] * (total_vertex_out_weight - cluster_out_weights[c])
                };

                if external_edge_weight_per_cluster_in_subset[c] >= vertex_weight_prod * resolution
                {
                    let mut diff = edge_weights_per_cluster[c];
                    if directed {
                        diff -= (vertex_out_weights[v] * cluster_in_weights[c]
                            + vertex_in_weights[v] * cluster_out_weights[c])
                            * resolution;
                    } else {
                        diff -= vertex_out_weights[v] * cluster_out_weights[c] * resolution;
                    }

                    if diff > max_diff {
                        best_cluster = c;
                        max_diff = diff;
                    }

                    // Calculate the transformed difference for sampling.
                    if diff >= 0.0 {
                        total_cum_trans_diff += (diff / beta).exp();
                    }
                }

                cum_trans_diff[j] = total_cum_trans_diff;
                edge_weights_per_cluster[c] = 0.0;
                neighbor_cluster_added[c] = false;
            }

            // Determine the neighboring cluster to which the currently
            // selected vertex will be moved.
            let chosen_cluster: usize = if total_cum_trans_diff < f64::INFINITY {
                let r = rng.unif(0.0, total_cum_trans_diff);
                let chosen_idx = binsearch_slice(cum_trans_diff, r, 0, nb_neigh_clusters);
                neighbor_clusters[chosen_idx] as usize
            } else {
                best_cluster
            };

            // Move vertex to randomly chosen cluster.
            cluster_out_weights[chosen_cluster] += vertex_out_weights[v];
            if directed {
                cluster_in_weights[chosen_cluster] += vertex_in_weights[v];
            }
            nb_vertices_per_cluster[chosen_cluster] += 1;

            for (&e, &u) in edges.iter().zip(neighbors) {
                let u = u as usize;
                if membership[u] == cluster_subset {
                    if refined_membership[u] == chosen_cluster as i64 {
                        external_edge_weight_per_cluster_in_subset[chosen_cluster] -=
                            edge_weights[e as usize];
                    } else {
                        external_edge_weight_per_cluster_in_subset[chosen_cluster] +=
                            edge_weights[e as usize];
                    }
                }
            }

            // Set cluster.
            if chosen_cluster != current_cluster {
                refined_membership[v] = chosen_cluster as i64;
                non_singleton_cluster[chosen_cluster] = true;
            }
        }
    }

    // Store the new cluster + 1 so that 0 indicates that no membership was
    // assigned yet (zeroed per call).
    zero_i64(&mut ws.mg_new_cluster, n);
    clean_refined_membership(
        vertex_subset,
        refined_membership,
        nb_refined_clusters,
        &mut ws.mg_new_cluster[..n],
    );

    Ok(())
}

/// Port of `leiden_get_clusters()` (`leiden.c` lines 553-568): create
/// clusters out of a membership vector. `clusters` must already have at
/// least as many items as the number of clusters in the membership vector,
/// and each item must be empty.
/// Parallel variant of the cluster refinement phase: refines the clusters in
/// parallel chunks, each chunk with its own RNG stream seeded from
/// `chunk_seeds[ci]`, then renumbers the per-chunk local clusters
/// consecutively in cluster order.
///
/// Deterministic regardless of thread scheduling: the chunk boundaries, the
/// seed assignment and the renumbering all depend only on the cluster order.
/// Each chunk task owns a workspace from `pool` (created once and reused
/// across levels), so there is no allocation churn and no shared mutable
/// state; `membership` and the weight slices are only read.
#[allow(clippy::too_many_arguments)]
fn refine_clusters_parallel(
    edges_per_vertex: &Inclist,
    edge_weights: &[f64],
    vertex_out_weights: &[f64],
    vertex_in_weights: Option<&[f64]>,
    clusters: &[Vec<i64>],
    membership: &[i64],
    resolution: f64,
    beta: f64,
    chunk_seeds: &[u64],
    pool: &mut Vec<Workspace>,
    vcount: usize,
) -> Result<(Vec<i64>, i64)> {
    let n_chunks = chunk_seeds.len();
    let chunk_len = if n_chunks == 0 {
        0
    } else {
        clusters.len().div_ceil(n_chunks)
    };

    // Grow the workspace pool to n_chunks entries. Workspaces are created
    // once and reused across levels and iterations.
    if pool.len() < n_chunks {
        pool.resize_with(n_chunks, Workspace::default);
    }

    // Refine each chunk (in parallel). Each chunk writes the refined
    // membership of its clusters into its own workspace buffer (local
    // cluster IDs, numbered from 0 within the chunk) and reports the number
    // of local clusters it created.
    let counts: Vec<Result<i64>> = (0..n_chunks)
        .into_par_iter()
        .zip(pool.par_iter_mut())
        .map(|(ci, ws)| {
            let s = (ci * chunk_len).min(clusters.len());
            let e = (s + chunk_len).min(clusters.len());
            // The refined buffer is indexed by vertex ID. Grow it if needed,
            // but never shrink it: entries beyond vcount are never read.
            if ws.mg_refined_membership.len() < vcount {
                ws.mg_refined_membership.resize(vcount, 0);
            }
            let mut rng = Pcg32::seeded(chunk_seeds[ci]);
            let mut count: i64 = 0;
            for (li, cluster) in clusters[s..e].iter().enumerate() {
                merge_vertices(
                    edges_per_vertex,
                    edge_weights,
                    vertex_out_weights,
                    vertex_in_weights,
                    cluster,
                    membership,
                    (s + li) as i64,
                    resolution,
                    beta,
                    &mut count,
                    &mut rng,
                    ws,
                )?;
            }
            Ok(count)
        })
        .collect();

    // Renumber the per-chunk local clusters consecutively in cluster order,
    // reproducing the sequential numbering semantics.
    let mut refined_out = vec![0_i64; vcount];
    let mut offset: i64 = 0;
    for (ci, count) in counts.into_iter().enumerate() {
        let count = count?;
        let s = (ci * chunk_len).min(clusters.len());
        let e = (s + chunk_len).min(clusters.len());
        let chunk_refined = &pool[ci].mg_refined_membership;
        for cluster in &clusters[s..e] {
            for &v in cluster {
                refined_out[v as usize] = offset + chunk_refined[v as usize];
            }
        }
        offset += count;
    }
    Ok((refined_out, offset))
}

fn get_clusters(membership: &[i64], clusters: &mut [Vec<i64>]) {
    for (i, &m) in membership.iter().enumerate() {
        clusters[m as usize].push(i as i64);
    }
}

/// The aggregated graph and its data returned by [`aggregate`].
type Aggregated = (Graph, Vec<f64>, Vec<f64>, Option<Vec<f64>>, Vec<i64>);
/// Port of `leiden_aggregate()` (`leiden.c` lines 584-700): aggregate the
/// graph based on the refined membership while setting the membership of
/// each aggregated vertex according to `membership`.
#[allow(clippy::too_many_arguments)]
fn aggregate(
    edges_per_vertex: &Inclist,
    edge_weights: &[f64],
    vertex_out_weights: &[f64],
    vertex_in_weights: Option<&[f64]>,
    membership: &[i64],
    refined_membership: &[i64],
    nb_refined_clusters: i64,
) -> Result<Aggregated> {
    let directed = vertex_in_weights.is_some();
    let vertex_in_weights = vertex_in_weights.unwrap_or(vertex_out_weights);

    // Get refined clusters.
    let mut refined_clusters: Vec<Vec<i64>> = vec![Vec::new(); nb_refined_clusters as usize];
    get_clusters(refined_membership, &mut refined_clusters);

    // New edges and their weights.
    let mut aggregated_edges: Vec<i64> = Vec::new();
    let mut aggregated_edge_weights: Vec<f64> = Vec::new();

    let mut aggregated_vertex_out_weights = vec![0.0_f64; nb_refined_clusters as usize];
    let mut aggregated_vertex_in_weights = if directed {
        vec![0.0_f64; nb_refined_clusters as usize]
    } else {
        Vec::new()
    };
    let mut aggregated_membership = vec![0_i64; nb_refined_clusters as usize];

    // Total edge weight to other clusters.
    let mut edge_weight_to_cluster = vec![0.0_f64; nb_refined_clusters as usize];
    let mut neighbor_cluster_added = vec![false; nb_refined_clusters as usize];
    let mut neighbor_clusters: Vec<i64> = Vec::new();

    // Check per cluster.
    for c in 0..nb_refined_clusters as usize {
        let refined_cluster = &refined_clusters[c];

        // Calculate the total edge weight to other clusters.
        aggregated_vertex_out_weights[c] = 0.0;
        if directed {
            aggregated_vertex_in_weights[c] = 0.0;
        }
        neighbor_clusters.clear();
        let mut v: i64 = -1;
        for &vi in refined_cluster.iter() {
            v = vi;
            let (edges, neighbors) = edges_per_vertex.edges_with_neighbors(v);
            for (&e, &u) in edges.iter().zip(neighbors) {
                let u = u as usize;
                let c2 = refined_membership[u] as usize;

                if c2 > c {
                    if !neighbor_cluster_added[c2] {
                        neighbor_cluster_added[c2] = true;
                        neighbor_clusters.push(c2 as i64);
                    }
                    edge_weight_to_cluster[c2] += edge_weights[e as usize];
                }
            }

            aggregated_vertex_out_weights[c] += vertex_out_weights[v as usize];
            if directed {
                aggregated_vertex_in_weights[c] += vertex_in_weights[v as usize];
            }
        }

        // Add actual edges from this cluster to the other clusters.
        for &c2 in &neighbor_clusters {
            aggregated_edges.push(c as i64);
            aggregated_edges.push(c2);

            aggregated_edge_weights.push(edge_weight_to_cluster[c2 as usize]);

            edge_weight_to_cluster[c2 as usize] = 0.0;
            neighbor_cluster_added[c2 as usize] = false;
        }

        aggregated_membership[c] = membership[v as usize];
    }

    // Mirrors igraph_create(aggregated_graph, &aggregated_edges,
    //                        nb_refined_clusters, directed);
    let aggregated_graph = Graph::new(
        nb_refined_clusters as usize,
        directed,
        &as_pairs(&aggregated_edges),
    )?;

    Ok((
        aggregated_graph,
        aggregated_edge_weights,
        aggregated_vertex_out_weights,
        if directed {
            Some(aggregated_vertex_in_weights)
        } else {
            None
        },
        aggregated_membership,
    ))
}

fn as_pairs(edges: &[i64]) -> Vec<(i64, i64)> {
    edges
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| (c[0], c[1]))
        .collect()
}

/// Port of `leiden_quality()` (`leiden.c` lines 729-795): calculate the
/// quality of the partition.
///
/// The quality is defined as
/// `1 / 2m sum_ij (A_ij - gamma n_i n_j) d(s_i, s_j)` for undirected graphs
/// and as `1 / m sum_ij (A_ij - gamma n^out_i n^in_j) d(s_i, s_j)` for
/// directed graphs.
fn quality(
    graph: &Graph,
    edge_weights: &[f64],
    vertex_out_weights: &[f64],
    vertex_in_weights: Option<&[f64]>,
    membership: &[i64],
    nb_clusters: i64,
    resolution: f64,
) -> f64 {
    let vcount = graph.vcount() as usize;
    let directed = vertex_in_weights.is_some();
    let vertex_in_weights = vertex_in_weights.unwrap_or(vertex_out_weights);
    let directed_multiplier: f64 = if directed { 1.0 } else { 2.0 };

    let mut quality = 0.0_f64;
    let mut total_edge_weight = 0.0_f64;

    for (e, &w) in edge_weights.iter().enumerate() {
        let from = graph.from(e as i64) as usize;
        let to = graph.to(e as i64) as usize;
        total_edge_weight += w;

        // We add the internal edge weights.
        if membership[from] == membership[to] {
            quality += directed_multiplier * w;
        }
    }

    // Initialize and compute cluster weights.
    let mut cluster_out_weights = vec![0.0_f64; vcount];
    let mut cluster_in_weights = if directed {
        vec![0.0_f64; vcount]
    } else {
        Vec::new()
    };

    for i in 0..vcount {
        let c = membership[i] as usize;
        cluster_out_weights[c] += vertex_out_weights[i];
        if directed {
            cluster_in_weights[c] += vertex_in_weights[i];
        }
    }

    // We subtract gamma * N^out_c * N^in_c.
    for c in 0..nb_clusters as usize {
        if directed {
            quality -= resolution * cluster_out_weights[c] * cluster_in_weights[c];
        } else {
            quality -= resolution * cluster_out_weights[c] * cluster_out_weights[c];
        }
    }

    // We normalise by m or 2m depending on directedness.
    quality / (directed_multiplier * total_edge_weight)
}

/// The core of the Leiden algorithm, port of the static `community_leiden()`
/// (`leiden.c` lines 803-1023). Relies on subroutines performing the three
/// phases: (1) local moving of vertices, (2) refinement of the partition and
/// (3) aggregation of the network based on the refined partition, using the
/// non-refined partition to create an initial partition for the aggregate
/// network.
///
/// `membership` is updated in place (level 0 semantics are reproduced by
/// writing back at the end, see the module docs of the port notes in
/// README). `graph` is the current working graph; all weight vectors and
/// `membership` must have lengths matching `graph`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::ptr_arg)]
fn community_leiden_core(
    graph: &Graph,
    edge_weights: &[f64],
    vertex_out_weights: &[f64],
    vertex_in_weights: Option<&[f64]>,
    resolution: f64,
    beta: f64,
    membership: &mut Vec<i64>,
    nb_clusters: &mut i64,
    changed: &mut bool,
    rng: &mut dyn Rng,
    level0_eps: &Inclist,
    ws: &mut Workspace,
    pool: &mut Vec<Workspace>,
    parallel: bool,
) -> Result<f64> {
    let n = graph.vcount() as usize;
    let directed = vertex_in_weights.is_some();

    // Working copy of the membership (the C code switches between the
    // user's membership at level 0 and the aggregated membership at higher
    // levels; using an owned copy is equivalent and avoids aliasing).
    let mut mem_working: Vec<i64> = membership.clone();

    // Clean membership: ensure that cluster indices are 0 <= c < n.
    *nb_clusters = reindex_membership(&mut mem_working, None);

    // We start out with no changes; whenever a vertex is moved, this is set
    // to true.
    *changed = false;

    // The aggregate graph and its data (None at level 0).
    let mut agg: Option<Level> = None;

    // Keep track of the aggregate vertex: the C code keeps a single vector,
    // initialized to 0..n and updated at each level.
    let mut aggregate_vertex: Vec<i64> = (0..n as i64).collect();

    loop {
        let (cur_graph, cur_ew, cur_vow, cur_viw, cur_mem, level) = match &mut agg {
            None => (
                graph,
                edge_weights,
                vertex_out_weights,
                vertex_in_weights,
                &mut mem_working[..],
                0_usize,
            ),
            Some(a) => (
                &a.graph,
                &a.ew[..],
                &a.vow[..],
                a.viw.as_deref(),
                &mut a.mem[..],
                a.depth,
            ),
        };

        // Get incidence list for fast iteration. At level 0 the incidence
        // list only depends on the (immutable) input graph, so it is built
        // once per run and reused across iterations.
        let _level_eps;
        let edges_per_vertex: &Inclist = if level == 0 {
            level0_eps
        } else {
            _level_eps = cur_graph.inclist();
            &_level_eps
        };

        // Move around the vertices in order to increase the quality.
        fastmove_vertices(
            cur_graph,
            edges_per_vertex,
            cur_ew,
            cur_vow,
            cur_viw,
            resolution,
            nb_clusters,
            cur_mem,
            changed,
            rng,
            ws,
        )?;

        // We only continue clustering if not all clusters are represented by
        // a single vertex yet.
        let continue_clustering = *nb_clusters < cur_graph.vcount();

        if continue_clustering {
            // Get vertex sets for each cluster.
            let mut clusters: Vec<Vec<i64>> = vec![Vec::new(); *nb_clusters as usize];
            get_clusters(cur_mem, &mut clusters);

            // Refine each cluster.
            let mut nb_refined_clusters: i64 = 0;
            if ws.mg_refined_membership.len() != cur_graph.vcount() as usize {
                ws.mg_refined_membership.clear();
                ws.mg_refined_membership
                    .resize(cur_graph.vcount() as usize, 0);
            }
            if parallel {
                // Draw one seed per chunk from the master RNG stream, in
                // chunk order, so the result is deterministic regardless of
                // thread scheduling. Each chunk refines its clusters with
                // its own sub-RNG seeded from that seed.
                //
                // Chunks are kept at least MIN_CHUNK clusters large so that
                // the per-task overhead is amortized; smaller cluster counts
                // are refined sequentially.
                const MIN_CHUNK: usize = 32;
                let n_threads = rayon::current_num_threads();
                let nb_clusters_usize = *nb_clusters as usize;
                let n_chunks = if nb_clusters_usize >= 2 * MIN_CHUNK {
                    n_threads.min(nb_clusters_usize / MIN_CHUNK)
                } else {
                    1
                };
                let mut chunk_seeds = Vec::with_capacity(n_chunks);
                for _ in 0..n_chunks {
                    chunk_seeds.push(rng.random_bits_u64(64));
                }
                let (refined, count) = refine_clusters_parallel(
                    edges_per_vertex,
                    cur_ew,
                    cur_vow,
                    cur_viw,
                    &clusters,
                    cur_mem,
                    resolution,
                    beta,
                    &chunk_seeds,
                    pool,
                    cur_graph.vcount() as usize,
                )?;
                if count >= cur_graph.vcount() {
                    // Refinement didn't aggregate anything: aggregate on the
                    // basis of the actual clustering (same as sequential).
                    ws.mg_refined_membership.copy_from_slice(cur_mem);
                    nb_refined_clusters = *nb_clusters;
                } else {
                    ws.mg_refined_membership.copy_from_slice(&refined);
                    nb_refined_clusters = count;
                }
            } else {
                ws.mg_refined_membership.fill(0);
                for (c, cluster) in clusters.iter().enumerate() {
                    merge_vertices(
                        edges_per_vertex,
                        cur_ew,
                        cur_vow,
                        cur_viw,
                        cluster,
                        cur_mem,
                        c as i64,
                        resolution,
                        beta,
                        &mut nb_refined_clusters,
                        rng,
                        ws,
                    )?;
                }
            }

            // If refinement didn't aggregate anything, we aggregate on the
            // basis of the actual clustering.
            if nb_refined_clusters >= cur_graph.vcount() {
                ws.mg_refined_membership.copy_from_slice(cur_mem);
                nb_refined_clusters = *nb_clusters;
            }

            // Keep track of the aggregate vertex.
            for item in aggregate_vertex.iter_mut() {
                *item = ws.mg_refined_membership[*item as usize];
            }

            let (new_graph, new_ew, new_vow, new_viw, new_mem) = aggregate(
                edges_per_vertex,
                cur_ew,
                cur_vow,
                cur_viw,
                cur_mem,
                &ws.mg_refined_membership,
                nb_refined_clusters,
            )?;

            agg = Some(Level {
                graph: new_graph,
                ew: new_ew,
                vow: new_vow,
                viw: new_viw,
                mem: new_mem,
                depth: level + 1,
            });
        } else {
            break;
        }
    }

    // Map the cluster membership in the aggregate network (level > 0) back to
    // the clusters in the original network (level == 0).
    match &agg {
        None => {
            membership.copy_from_slice(&mem_working);
        }
        Some(a) => {
            for i in 0..n {
                membership[i] = a.mem[aggregate_vertex[i] as usize];
            }
        }
    }
    let _ = directed;

    // Calculate quality (on the original graph and weights).
    Ok(quality(
        graph,
        edge_weights,
        vertex_out_weights,
        vertex_in_weights,
        membership,
        *nb_clusters,
        resolution,
    ))
}

/// One aggregation level of the core loop.
struct Level {
    graph: Graph,
    ew: Vec<f64>,
    vow: Vec<f64>,
    viw: Option<Vec<f64>>,
    mem: Vec<i64>,
    /// Aggregation depth (0 is the input graph level).
    depth: usize,
}

/// Port of `igraph_reindex_membership()` (`community_misc.c`): renumber
/// cluster IDs consecutively from 0. The `new_to_old` mapping (used only by
/// other igraph functions) is not needed by the Leiden code and is omitted.
/// The range-check fallback for out-of-range IDs is not needed either, since
/// the Leiden code always calls it on already-cleaned membership vectors
/// (the fast path assumes IDs in `0..vcount`).
///
/// Returns the number of clusters.
pub(crate) fn reindex_membership(membership: &mut [i64], new_to_old: Option<&mut Vec<i64>>) -> i64 {
    let vcount = membership.len();
    let mut new_cluster = vec![0_i64; vcount];
    let mut new_to_old = new_to_old;

    if let Some(nto) = &mut new_to_old {
        nto.clear();
    }

    // Clean clusters. Store the new cluster + 1 so that membership == 0
    // indicates that no cluster was assigned yet.
    let mut nb_clusters: i64 = 1;
    for m in membership.iter() {
        let c = *m as usize;
        if new_cluster[c] == 0 {
            new_cluster[c] = nb_clusters;
            nb_clusters += 1;
            if let Some(nto) = &mut new_to_old {
                nto.push(c as i64);
            }
        }
    }

    // Assign new membership.
    for m in membership.iter_mut() {
        *m = new_cluster[*m as usize] - 1;
    }

    nb_clusters - 1
}

/// Port of `igraph_community_leiden()` (`leiden.c` lines 1152-1277): the
/// public interface with raw vertex weights.
///
/// Finds community structure using the Leiden algorithm. The objective
/// function being optimized is
/// `1 / 2m sum_ij (A_ij - gamma n_i n_j) d(s_i, s_j)` in the undirected case
/// and `1 / m sum_ij (A_ij - gamma n^out_i n^in_j) d(s_i, s_j)` in the
/// directed case.
///
/// - `edge_weights`: `None` means every edge has weight 1. Weights need not
///   be non-negative.
/// - `vertex_out_weights` / `vertex_in_weights`: `None` means every vertex
///   has weight 1. `vertex_in_weights` must be `None` for undirected graphs;
///   for directed graphs, `None` means in-weights equal out-weights (which
///   effectively ignores edge directions).
/// - `start`: start from the given membership vector (`true`) or from a
///   singleton partition (`false`).
/// - `n_iterations`: number of times to run the core Leiden algorithm. A
///   negative value keeps iterating until an iteration does not change the
///   clustering.
///
/// The membership vector is both used as the initial membership and updated
/// in place; it must have length `vcount` (or be resized appropriately when
/// `start` is `false`).
#[allow(clippy::too_many_arguments)]
fn leiden_impl(
    graph: &Graph,
    edge_weights: Option<&[f64]>,
    vertex_out_weights: Option<&[f64]>,
    vertex_in_weights: Option<&[f64]>,
    resolution: f64,
    beta: f64,
    start: bool,
    n_iterations: i64,
    membership: &mut Vec<i64>,
    rng: &mut dyn Rng,
    parallel: bool,
) -> Result<Outcome> {
    let vcount = graph.vcount();
    let ecount = graph.ecount();
    let directed = graph.is_directed();

    if start && membership.len() != vcount as usize {
        return Err(LeidenError::new(
            "Membership vector length does not equal the number of vertices.",
        ));
    }
    if !start {
        membership.clear();
        membership.extend(0..vcount);
    }

    // Check edge weights to possibly use default.
    let default_ew: Vec<f64>;
    let edge_weights: &[f64] = match edge_weights {
        None => {
            default_ew = vec![1.0; ecount as usize];
            &default_ew
        }
        Some(w) => {
            if w.len() != ecount as usize {
                return Err(LeidenError::new(format!(
                    "Edge weight vector length ({}) does not match number of edges ({}).",
                    w.len(),
                    ecount
                )));
            }
            w
        }
    };

    // Check vertex out-weights to possibly use default.
    let default_vow: Vec<f64>;
    let vertex_out_weights: &[f64] = match vertex_out_weights {
        None => {
            default_vow = vec![1.0; vcount as usize];
            &default_vow
        }
        Some(w) => {
            if w.len() != vcount as usize {
                let kind = if directed { "out-" } else { "" };
                return Err(LeidenError::new(format!(
                    "Vertex {}weight vector length ({}) does not match number of vertices ({}).",
                    kind,
                    w.len(),
                    vcount
                )));
            }
            w
        }
    };

    let explicit_viw: Vec<f64>;
    let vertex_in_weights: Option<&[f64]> = if directed {
        match vertex_in_weights {
            // When in-weights are not given for a directed graph, assume
            // that they are the same as the out-weights. This effectively
            // ignores edge directions.
            None => Some(vertex_out_weights),
            Some(w) => {
                if w.len() != vcount as usize {
                    return Err(LeidenError::new(format!(
                        "Vertex in-weight vector length ({}) does not match number of vertices ({}).",
                        w.len(),
                        vcount
                    )));
                }
                explicit_viw = w.to_vec();
                Some(&explicit_viw)
            }
        }
    } else {
        // In-weights must be None in the undirected case.
        if vertex_in_weights.is_some() {
            return Err(LeidenError::new(
                "Vertex in-weights must not be given for undirected graphs.",
            ));
        }
        None
    };

    // Perform the actual Leiden algorithm iteratively. We either perform a
    // fixed number of iterations, or we perform iterations until the quality
    // remains unchanged. Even if a single iteration did not change anything,
    // a subsequent iteration may still find some improvement, because each
    // iteration explores different subsets of vertices.
    let mut changed = true;
    let mut nb_clusters: i64 = vcount;
    let mut quality = 0.0_f64;
    let mut itr: i64 = 0;

    // The incidence list of the level-0 graph only depends on the immutable
    // input graph, so it is built once and reused across iterations. The C
    // code rebuilds it on every iteration; reusing it is a performance-only
    // optimization that does not affect the results.
    let level0_eps = graph.inclist();
    let mut ws = Workspace::default();
    // Workspaces for the parallel refinement chunks (grown on demand,
    // reused across iterations and levels).
    let mut pool: Vec<Workspace> = Vec::new();

    while if n_iterations < 0 {
        changed
    } else {
        itr < n_iterations
    } {
        quality = community_leiden_core(
            graph,
            edge_weights,
            vertex_out_weights,
            vertex_in_weights,
            resolution,
            beta,
            membership,
            &mut nb_clusters,
            &mut changed,
            rng,
            &level0_eps,
            &mut ws,
            &mut pool,
            parallel,
        )?;
        itr += 1;
    }

    Ok(Outcome {
        nb_clusters,
        quality,
    })
}

/// Port of `igraph_community_leiden()` with a parallel variant of the
/// refinement phase.
///
/// Runs the same Leiden algorithm as [`leiden`], but refines the clusters in
/// parallel chunks, each with its own RNG stream. The result is a valid
/// high-quality Leiden partition with all structural guarantees (connected
/// clusters, non-decreasing per-iteration quality) and is deterministic
/// run-to-run, but it is *not* bit-identical to the sequential [`leiden`]
/// result, because the random decisions are drawn from per-chunk streams
/// instead of a single stream.
#[allow(clippy::too_many_arguments)]
pub fn leiden_parallel(
    graph: &Graph,
    edge_weights: Option<&[f64]>,
    vertex_out_weights: Option<&[f64]>,
    vertex_in_weights: Option<&[f64]>,
    resolution: f64,
    beta: f64,
    start: bool,
    n_iterations: i64,
    membership: &mut Vec<i64>,
    rng: &mut dyn Rng,
) -> Result<Outcome> {
    leiden_impl(
        graph,
        edge_weights,
        vertex_out_weights,
        vertex_in_weights,
        resolution,
        beta,
        start,
        n_iterations,
        membership,
        rng,
        true,
    )
}

/// Sequential, bit-exact port of `igraph_community_leiden()`.
#[allow(clippy::too_many_arguments)]
pub fn leiden(
    graph: &Graph,
    edge_weights: Option<&[f64]>,
    vertex_out_weights: Option<&[f64]>,
    vertex_in_weights: Option<&[f64]>,
    resolution: f64,
    beta: f64,
    start: bool,
    n_iterations: i64,
    membership: &mut Vec<i64>,
    rng: &mut dyn Rng,
) -> Result<Outcome> {
    leiden_impl(
        graph,
        edge_weights,
        vertex_out_weights,
        vertex_in_weights,
        resolution,
        beta,
        start,
        n_iterations,
        membership,
        rng,
        false,
    )
}

/// Port of `igraph_community_leiden_simple()` (`leiden.c` lines
/// 1353-1504): the simplified interface, choosing from a set of objective
/// functions instead of supplying vertex weights.
#[allow(clippy::too_many_arguments)]
fn leiden_simple_impl(
    graph: &Graph,
    weights: Option<&[f64]>,
    objective: Objective,
    resolution: f64,
    beta: f64,
    start: bool,
    n_iterations: i64,
    membership: &mut Vec<i64>,
    rng: &mut dyn Rng,
    parallel: bool,
) -> Result<Outcome> {
    let vcount = graph.vcount();
    let ecount = graph.ecount();
    let directed = graph.is_directed();

    // Basic weight vector validation; calculate properties used for the
    // validation steps specific to different objective functions.
    let mut min_weight = f64::INFINITY;
    if let Some(w) = weights {
        if w.len() != ecount as usize {
            return Err(LeidenError::new(
                "Edge weight vector length does not match number of edges.",
            ));
        }
        for &wi in w {
            if wi < min_weight {
                min_weight = wi;
            }
            if !wi.is_finite() {
                return Err(LeidenError::new(format!(
                    "Edge weights must not be infinite or NaN, got {}.",
                    format_g(wi)
                )));
            }
        }
    }

    let mut vertex_out_weights = vec![0.0_f64; vcount as usize];
    let mut vertex_in_weights = if directed {
        vec![0.0_f64; vcount as usize]
    } else {
        Vec::new()
    };

    if start && membership.len() != vcount as usize {
        return Err(LeidenError::new(format!(
            "Requesting to start the computation from a specific community assignment, but the given membership vector has a different size ({} than the vertex count ({}).",
            membership.len(),
            vcount
        )));
    }
    if !start {
        membership.clear();
        membership.extend(0..vcount);
    }

    let mut resolution = resolution;
    match objective {
        Objective::Modularity => {
            if min_weight < 0.0 {
                return Err(LeidenError::new(format!(
                    "Edge weights must not be negative for Leiden community detection with modularity objective function, got {}.",
                    format_g(min_weight)
                )));
            }

            strength_into(graph, &mut vertex_out_weights, false, weights);
            if directed {
                strength_into(graph, &mut vertex_in_weights, true, weights);
            }

            // If directed, the sum of vertex_out_weights is the total edge
            // weight. If undirected, it is twice the total edge weight.
            resolution /= vertex_out_weights.iter().sum::<f64>();
        }
        Objective::Cpm => {
            vertex_out_weights.iter_mut().for_each(|w| *w = 1.0);
            if directed {
                vertex_in_weights.iter_mut().for_each(|w| *w = 1.0);
            }
        }
        Objective::Er => {
            if min_weight < 0.0 {
                return Err(LeidenError::new(format!(
                    "Edge weights must not be negative for Leiden community detection with ER objective function, got {}.",
                    format_g(min_weight)
                )));
            }

            vertex_out_weights.iter_mut().for_each(|w| *w = 1.0);
            if directed {
                vertex_in_weights.iter_mut().for_each(|w| *w = 1.0);
            }

            // Note: Loops must be allowed, as the aggregation step of the
            // algorithm effectively creates them.
            let p = density(graph, weights, true)?;
            resolution *= p;
        }
    }

    let outcome = leiden_impl(
        graph,
        weights,
        Some(&vertex_out_weights),
        if directed {
            Some(&vertex_in_weights)
        } else {
            None
        },
        resolution,
        beta,
        start,
        n_iterations,
        membership,
        rng,
        parallel,
    )?;

    Ok(outcome)
}

/// Port of `igraph_community_leiden_simple()` with a parallel variant of the
/// refinement phase.
///
/// Runs the same algorithm as [`leiden_simple`], but refines the clusters in
/// parallel chunks. The result is a valid high-quality Leiden partition with
/// all structural guarantees and is deterministic run-to-run, but it is *not*
/// bit-identical to the sequential [`leiden_simple`] result.
#[allow(clippy::too_many_arguments)]
pub fn leiden_simple_parallel(
    graph: &Graph,
    weights: Option<&[f64]>,
    objective: Objective,
    resolution: f64,
    beta: f64,
    start: bool,
    n_iterations: i64,
    membership: &mut Vec<i64>,
    rng: &mut dyn Rng,
) -> Result<Outcome> {
    leiden_simple_impl(
        graph,
        weights,
        objective,
        resolution,
        beta,
        start,
        n_iterations,
        membership,
        rng,
        true,
    )
}

/// Sequential, bit-exact port of `igraph_community_leiden_simple()`.
#[allow(clippy::too_many_arguments)]
pub fn leiden_simple(
    graph: &Graph,
    weights: Option<&[f64]>,
    objective: Objective,
    resolution: f64,
    beta: f64,
    start: bool,
    n_iterations: i64,
    membership: &mut Vec<i64>,
    rng: &mut dyn Rng,
) -> Result<Outcome> {
    leiden_simple_impl(
        graph,
        weights,
        objective,
        resolution,
        beta,
        start,
        n_iterations,
        membership,
        rng,
        false,
    )
}

/// Port of `igraph_strength()` for all vertices
/// (`src/properties/degrees.c`, `strength_all()`): weighted degree.
///
/// `mode_in`: `false` for out-strength, `true` for in-strength. Loops are
/// counted (`IGRAPH_LOOPS`).
fn strength_into(graph: &Graph, res: &mut [f64], mode_in: bool, weights: Option<&[f64]>) {
    let no_of_edges = graph.ecount() as usize;
    res.iter_mut().for_each(|r| *r = 0.0);

    // Loops are counted (IGRAPH_LOOPS). For undirected graphs, mode is
    // forced to IGRAPH_ALL, so both passes run and each edge (including
    // loops) contributes to both endpoints. For directed graphs, out-
    // strength only accumulates into `from` and in-strength into `to`.
    let out_pass = !mode_in || !graph.is_directed();
    let in_pass = mode_in || !graph.is_directed();

    if out_pass {
        for edge in 0..no_of_edges {
            res[graph.from(edge as i64) as usize] += weights.map_or(1.0, |w| w[edge]);
        }
    }
    if in_pass {
        for edge in 0..no_of_edges {
            res[graph.to(edge as i64) as usize] += weights.map_or(1.0, |w| w[edge]);
        }
    }
}

/// Port of `igraph_strength()` for all vertices with `IGRAPH_LOOPS`
/// (`src/properties/degrees.c`, `strength_all()`): the weighted degree of
/// every vertex.
///
/// `mode_in`: `false` for out-strength (`IGRAPH_OUT`), `true` for
/// in-strength (`IGRAPH_IN`). For undirected graphs, out- and in-strength
/// are the same (total strengths).
pub fn strength(graph: &Graph, mode_in: bool, weights: Option<&[f64]>) -> Vec<f64> {
    let mut res = vec![0.0_f64; graph.vcount() as usize];
    strength_into(graph, &mut res, mode_in, weights);
    res
}

/// Port of `igraph_density()` (`src/properties/basic_properties.c`) with
/// `loops` handled explicitly.
fn density(graph: &Graph, weights: Option<&[f64]>, loops: bool) -> Result<f64> {
    let directed = graph.is_directed();
    let ecount = graph.ecount();
    let vcount = graph.vcount() as f64;

    if vcount == 0.0 {
        return Ok(f64::NAN);
    }

    let total_weight = match weights {
        Some(w) => {
            if w.len() != ecount as usize {
                return Err(LeidenError::new(
                    "Weight vector length does not match edge count.",
                ));
            }
            w.iter().sum::<f64>()
        }
        None => ecount as f64,
    };

    if !loops {
        if vcount == 1.0 {
            Ok(f64::NAN)
        } else if directed {
            Ok(total_weight / vcount / (vcount - 1.0))
        } else {
            Ok(total_weight / vcount * 2.0 / (vcount - 1.0))
        }
    } else if directed {
        Ok(total_weight / vcount / vcount)
    } else {
        Ok(total_weight / vcount * 2.0 / (vcount + 1.0))
    }
}

/// Port of `igraph_vector_binsearch_slice()`
/// (`src/core/vector.pmt`): binary search of a sorted slice, returning the
/// position of `what` if present, or the position where it should be
/// inserted otherwise (lower bound semantics).
fn binsearch_slice(v: &[f64], what: f64, start: usize, end: usize) -> usize {
    let mut left = start;
    let mut right = end as i64 - 1;

    while left as i64 <= right {
        let middle = left as i64 + ((right - left as i64) >> 1);
        if v[middle as usize] > what {
            right = middle - 1;
        } else if v[middle as usize] < what {
            left = middle as usize + 1;
        } else {
            return middle as usize;
        }
    }

    left
}
