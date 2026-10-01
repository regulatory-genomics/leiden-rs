//! Rust port of the igraph Leiden community detection implementation.
//!
//! Ported from `igraph`'s `src/community/leiden.c` (commit `8225c3a4b`),
//! including its support routines: the PCG32 default RNG
//! (`src/random/rng_pcg32.c` + `vendor/pcg/pcg_variants.h`), graph
//! indexing and incidence lists (`src/graph/type_indexededgelist.c`,
//! `src/core/vector.c`), membership reindexing
//! (`src/community/community_misc.c`), strength and density
//! (`src/properties/`). The port is faithful to the C implementation,
//! including its random number stream, so results are bit-for-bit
//! identical to C igraph given the same seed and inputs.
//!
//! The algorithm implements the Leiden method for community detection
//! (Traag, Waltman & van Eck, 2019), optimizing either modularity, the
//! Constant Potts Model (CPM) or an Erdős–Rényi (ER) based objective.

mod error;
mod graph;
mod leiden;
mod modularity;
mod rng;

pub use error::LeidenError;
pub use graph::Graph;
pub use leiden::{leiden, leiden_simple, strength, Objective};
pub use modularity::modularity;
pub use rng::{shuffle, Pcg32, Rng};

/// Result of a Leiden community detection run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outcome {
    /// Number of clusters contained in the final membership vector.
    pub nb_clusters: i64,
    /// Quality of the partition (objective function value).
    pub quality: f64,
}
