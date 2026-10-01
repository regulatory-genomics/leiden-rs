//! Port of `igraph_modularity()` (`src/community/modularity.c`).

use crate::error::{LeidenError, Result};
use crate::graph::Graph;

/// Port of `igraph_modularity()`: the modularity of a graph with respect to
/// some clustering of the vertices.
///
/// `directed` selects the directed or undirected version of modularity
/// (ignored for undirected graphs); `resolution` is the resolution
/// parameter γ. Returns NaN for graphs with no edges, where modularity is
/// not well-defined.
pub fn modularity(
    graph: &Graph,
    membership: &[i64],
    weights: Option<&[f64]>,
    resolution: f64,
    directed: bool,
) -> Result<f64> {
    let vcount = graph.vcount();
    let ecount = graph.ecount();
    let use_directed = directed && graph.is_directed();
    let dm: f64 = if use_directed { 1.0 } else { 2.0 };

    if membership.len() != vcount as usize {
        return Err(LeidenError::new(
            "Membership vector size differs from number of vertices.",
        ));
    }
    if resolution < 0.0 {
        return Err(LeidenError::new(
            "The resolution parameter must not be negative.",
        ));
    }

    if ecount == 0 {
        // Special case: the modularity of graphs with no edges is not
        // well-defined.
        return Ok(f64::NAN);
    }

    // Community indices in this crate are always reindexed into the
    // standard range, so no reindexing step is needed here.
    let no_of_partitions = membership.iter().max().copied().unwrap_or(0) + 1;
    if no_of_partitions <= 0 || no_of_partitions > vcount {
        return Err(LeidenError::new(
            "Membership vector contains invalid community indices.",
        ));
    }

    let mut k_out = vec![0.0_f64; no_of_partitions as usize];
    let mut k_in = vec![0.0_f64; no_of_partitions as usize];

    let mut e = 0.0_f64;
    let m: f64;
    match weights {
        Some(w) => {
            if w.len() != ecount as usize {
                return Err(LeidenError::new(
                    "Weight vector size differs from number of edges.",
                ));
            }
            m = w.iter().sum();
            for i in 0..ecount {
                let wi = w[i as usize];
                if wi < 0.0 {
                    return Err(LeidenError::new("Negative weight in weight vector."));
                }
                let c1 = membership[graph.from(i) as usize];
                let c2 = membership[graph.to(i) as usize];
                if c1 == c2 {
                    e += dm * wi;
                }
                k_out[c1 as usize] += wi;
                k_in[c2 as usize] += wi;
            }
        }
        None => {
            m = ecount as f64;
            for i in 0..ecount {
                let c1 = membership[graph.from(i) as usize];
                let c2 = membership[graph.to(i) as usize];
                if c1 == c2 {
                    e += dm;
                }
                k_out[c1 as usize] += 1.0;
                k_in[c2 as usize] += 1.0;
            }
        }
    }

    if !use_directed {
        // Graph is undirected: k_in := k_out + k_in for both vectors.
        for i in 0..no_of_partitions as usize {
            k_out[i] += k_in[i];
        }
        k_in.copy_from_slice(&k_out);
    }

    // Divide all vectors by the total weight.
    let scale = 1.0 / (dm * m);
    for k in k_out.iter_mut().chain(k_in.iter_mut()) {
        *k *= scale;
    }
    e /= dm * m;

    if m > 0.0 {
        let mut q = e;
        for i in 0..no_of_partitions as usize {
            q -= resolution * k_out[i] * k_in[i];
        }
        Ok(q)
    } else {
        Ok(f64::NAN)
    }
}
