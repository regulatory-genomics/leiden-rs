//! Minimal graph representation, ported from igraph's graph core.
//!
//! Reproduces the parts of igraph's `igraph_t` that the Leiden algorithm
//! depends on: the edge list (`from`/`to`, with edge IDs in insertion order
//! and `from <= to` for undirected graphs) and the sorted index structures
//! `os`/`is`/`oi`/`ii` built by `igraph_create()` via
//! `igraph_vector_int_pair_order()` and `igraph_i_create_start_vectors()`
//! (`src/graph/type_indexededgelist.c`, `src/core/vector.c`).
//!
//! The incidence list port (`Graph::incident`) is a literal translation of
//! `igraph_incident()` (`src/graph/type_indexededgelist.c`) with
//! `IGRAPH_LOOPS_TWICE` handling, including the exact ordering of the
//! returned edge IDs. This matters for parity: the order in which
//! neighboring clusters are discovered influences tie-breaking, which in
//! turn influences the random number stream.

use crate::error::{LeidenError, Result};

/// A graph with vertex IDs `0..n-1` and edge IDs in insertion order.
///
/// Mirrors the internal layout of igraph's `igraph_t`:
/// - `from`/`to`: edge endpoints, edge IDs in insertion order. For
///   undirected graphs, edges are stored with `from <= to`.
/// - `os`/`oi`: out-sequence and out-index; `is`/`ii`: in-sequence and
///   in-index, exactly as built by `igraph_create()`.
#[derive(Debug, Clone)]
pub struct Graph {
    n: usize,
    directed: bool,
    from: Vec<i64>,
    to: Vec<i64>,
    os: Vec<i64>,
    oi: Vec<i64>,
    is: Vec<i64>,
    ii: Vec<i64>,
}

impl Graph {
    /// Creates a graph from an edge list. Mirrors `igraph_create()`.
    ///
    /// For undirected graphs, edge endpoints are swapped so that
    /// `from <= to` (`igraph_add_edges()` semantics), keeping edge IDs in
    /// insertion order.
    pub fn new(n: usize, directed: bool, edges: &[(i64, i64)]) -> Result<Graph> {
        for &(from, to) in edges {
            if from < 0 || to < 0 || from >= n as i64 || to >= n as i64 {
                return Err(LeidenError::new(
                    "Out-of-range vertex IDs when adding edges.",
                ));
            }
        }

        let mut from = Vec::with_capacity(edges.len());
        let mut to = Vec::with_capacity(edges.len());
        for &(a, b) in edges {
            if directed || a > b {
                from.push(a);
                to.push(b);
            } else {
                // Undirected: store with from <= to.
                from.push(b);
                to.push(a);
            }
        }

        let mut graph = Graph {
            n,
            directed,
            from,
            to,
            os: Vec::new(),
            oi: Vec::new(),
            is: Vec::new(),
            ii: Vec::new(),
        };

        // oi & ii, mirroring igraph_add_edges()/igraph_create():
        // oi = pair_order(from, to), ii = pair_order(to, from).
        graph.oi = pair_order(&graph.from, &graph.to, n as i64);
        graph.ii = pair_order(&graph.to, &graph.from, n as i64);

        // os & is.
        graph.os = create_start_vectors(&graph.from, &graph.oi, n as i64);
        graph.is = create_start_vectors(&graph.to, &graph.ii, n as i64);

        Ok(graph)
    }

    /// Number of vertices. Mirrors `igraph_vcount()`.
    pub fn vcount(&self) -> i64 {
        self.n as i64
    }

    /// Number of edges. Mirrors `igraph_ecount()`.
    pub fn ecount(&self) -> i64 {
        self.from.len() as i64
    }

    /// Mirrors `igraph_is_directed()`.
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Mirrors `IGRAPH_FROM(graph, e)`.
    pub fn from(&self, e: i64) -> i64 {
        self.from[e as usize]
    }

    /// Mirrors `IGRAPH_TO(graph, e)`.
    pub fn to(&self, e: i64) -> i64 {
        self.to[e as usize]
    }

    /// Mirrors `IGRAPH_OTHER(graph, e, v)`.
    pub fn other(&self, e: i64, v: i64) -> i64 {
        if self.from(e) == v {
            self.to(e)
        } else {
            self.from(e)
        }
    }

    /// Mirrors `igraph_incident()` with `IGRAPH_LOOPS_TWICE`, returning
    /// incident edge IDs in the exact order produced by igraph.
    ///
    /// For undirected graphs (and directed graphs with `IGRAPH_ALL`),
    /// self-loops are included twice; for directed graphs the
    /// `IGRAPH_LOOPS_TWICE` request is demoted to `IGRAPH_LOOPS_ONCE`, so
    /// self-loops appear once.
    pub fn incident(&self, node: i64) -> Vec<i64> {
        let mut eids = vec![0_i64; self.incident_count(node) as usize];
        let mut others = vec![0_i64; eids.len()];
        self.incident_into(node, &mut eids, &mut others);
        eids
    }

    /// Number of entries produced by [`Graph::incident`] for `node`.
    fn incident_count(&self, node: i64) -> i64 {
        (self.os[node as usize + 1] - self.os[node as usize])
            + (self.is[node as usize + 1] - self.is[node as usize])
    }

    /// Write the incident edge IDs of `node` (in exactly the order
    /// [`Graph::incident`] would produce) into `out`, together with the
    /// "other endpoint" of each edge with respect to `node` into `others`
    /// (in matching order). Both slices must have been sized with
    /// [`Graph::incident_count`] entries.
    ///
    /// This is the shared fast path used both by [`Graph::incident`] and by
    /// the CSR incidence list, guaranteeing identical per-vertex edge order.
    /// Filling `others` here lets the Leiden hot loops replace the
    /// bounds-checked `graph.other(e, v)` calls with a branch-free lookup.
    fn incident_into(&self, node: i64, out: &mut [i64], others: &mut [i64]) {
        let mut w = 0_usize;
        let node_us = node as usize;
        if !self.directed {
            // Easy case with mode == IGRAPH_ALL.
            // mode & IGRAPH_OUT: iterate oi; keep loops (LOOPS_TWICE).
            // Entries from the out-index have from == node.
            for i in self.os[node_us]..self.os[node_us + 1] {
                let e = self.oi[i as usize];
                out[w] = e;
                others[w] = self.to[e as usize];
                w += 1;
            }
            // mode & IGRAPH_IN: iterate ii; with LOOPS_TWICE loops are kept,
            // so no filtering. Entries from the in-index have to == node.
            for i in self.is[node_us]..self.is[node_us + 1] {
                let e = self.ii[i as usize];
                out[w] = e;
                others[w] = self.from[e as usize];
                w += 1;
            }
        } else {
            // Merge case (directed, mode == IGRAPH_ALL). Note that the
            // demotion of IGRAPH_LOOPS_TWICE to IGRAPH_LOOPS_ONCE only
            // happens when mode != IGRAPH_ALL, so with mode == IGRAPH_ALL
            // (which is what the Leiden code uses) loops are kept twice:
            // for a self-loop, eid1 and eid2 are the same edge ID, pushed
            // both times.
            let mut i1 = self.os[node_us];
            let mut i2 = self.is[node_us];
            let j1 = self.os[node_us + 1];
            let j2 = self.is[node_us + 1];

            while i1 < j1 && i2 < j2 {
                let eid1 = self.oi[i1 as usize];
                let eid2 = self.ii[i2 as usize];
                let n1 = self.to(eid1);
                let n2 = self.from(eid2);
                if n1 < n2 {
                    i1 += 1;
                    out[w] = eid1;
                    others[w] = n1;
                    w += 1;
                } else if n1 > n2 {
                    i2 += 1;
                    out[w] = eid2;
                    others[w] = n2;
                    w += 1;
                } else {
                    // Multiple edges and self-loops: both eid1 and eid2
                    // (which are the same edge for a self-loop).
                    i1 += 1;
                    i2 += 1;
                    out[w] = eid1;
                    others[w] = n1;
                    w += 1;
                    out[w] = eid2;
                    others[w] = n2;
                    w += 1;
                }
            }

            while i1 < j1 {
                let e = self.oi[i1 as usize];
                out[w] = e;
                others[w] = self.to[e as usize];
                w += 1;
                i1 += 1;
            }
            while i2 < j2 {
                let e = self.ii[i2 as usize];
                out[w] = e;
                others[w] = self.from[e as usize];
                w += 1;
                i2 += 1;
            }
        }
        debug_assert_eq!(w, out.len());
    }

    /// CSR incidence list mirroring `igraph_inclist_init()` with
    /// `IGRAPH_ALL` and `IGRAPH_LOOPS_TWICE`: a flat edge-ID array with
    /// per-vertex start offsets, in the exact per-vertex order produced by
    /// [`Graph::incident`].
    ///
    /// This replaces one allocation of two flat arrays for what would
    /// otherwise be `vcount` separate `Vec` allocations.
    pub fn inclist(&self) -> Inclist {
        let n = self.n;
        // Per-vertex incident counts prefix sum: telescoping
        // starts[v] = sum_{u<v} (os[u+1]-os[u] + is[u+1]-is[u]) = os[v] + is[v],
        // so the start offsets can be computed elementwise.
        let mut starts = Vec::with_capacity(n + 1);
        for v in 0..=n {
            starts.push(self.os[v] + self.is[v]);
        }
        let total = starts[n] as usize;
        let mut eids = vec![0_i64; total];
        let mut others = vec![0_i64; total];
        for v in 0..n as i64 {
            let s = starts[v as usize] as usize;
            let e = starts[v as usize + 1] as usize;
            self.incident_into(v, &mut eids[s..e], &mut others[s..e]);
        }
        Inclist {
            starts,
            eids,
            others,
        }
    }
}

/// CSR incidence list returned by [`Graph::inclist`].
pub struct Inclist {
    starts: Vec<i64>,
    eids: Vec<i64>,
    /// "Other endpoint" of each stored edge ID with respect to its owner
    /// vertex, in matching order with `eids`.
    others: Vec<i64>,
}

impl Inclist {
    /// Incident edge IDs of `v`, in the exact order produced by
    /// [`Graph::incident`].
    #[inline]
    pub fn edges(&self, v: i64) -> &[i64] {
        let s = self.starts[v as usize] as usize;
        let e = self.starts[v as usize + 1] as usize;
        &self.eids[s..e]
    }

    /// Incident edge IDs of `v` together with the "other endpoint" of each
    /// edge with respect to `v`, in matching order.
    #[inline]
    pub fn edges_with_neighbors(&self, v: i64) -> (&[i64], &[i64]) {
        let s = self.starts[v as usize] as usize;
        let e = self.starts[v as usize + 1] as usize;
        (&self.eids[s..e], &self.others[s..e])
    }

    /// Total number of stored edge IDs.
    pub fn total(&self) -> usize {
        self.eids.len()
    }
}

/// Mirrors `igraph_vector_int_pair_order()` (`src/core/vector.c`): the
/// permutation that sorts edges lexicographically by `(v[i], v2[i])`,
/// computed with two counting-sort passes over linked-list buckets. Ported
/// literally to reproduce the tie ordering of multi-edges exactly.
fn pair_order(v: &[i64], v2: &[i64], nodes: i64) -> Vec<i64> {
    let edges = v.len();
    let mut ptr = vec![0_i64; nodes as usize + 1];
    let mut rad = vec![0_i64; edges];
    let mut res = vec![0_i64; edges];

    // First pass: counting sort by v2.
    for i in 0..edges {
        let radix = v2[i];
        if ptr[radix as usize] != 0 {
            rad[i] = ptr[radix as usize];
        }
        ptr[radix as usize] = i as i64 + 1;
    }

    let mut j = 0_usize;
    for p in ptr.iter().take(nodes as usize + 1) {
        if *p != 0 {
            let mut next = *p - 1;
            res[j] = next;
            j += 1;
            while rad[next as usize] != 0 {
                next = rad[next as usize] - 1;
                res[j] = next;
                j += 1;
            }
        }
    }

    // Second pass: stable counting sort of the above by v.
    ptr.iter_mut().for_each(|p| *p = 0);
    rad.iter_mut().for_each(|r| *r = 0);

    for i in 0..edges {
        let edge = res[edges - i - 1];
        let radix = v[edge as usize];
        if ptr[radix as usize] != 0 {
            rad[edge as usize] = ptr[radix as usize];
        }
        ptr[radix as usize] = edge + 1;
    }

    let mut j = 0_usize;
    for p in ptr.iter().take(nodes as usize + 1) {
        if *p != 0 {
            let mut next = *p - 1;
            res[j] = next;
            j += 1;
            while rad[next as usize] != 0 {
                next = rad[next as usize] - 1;
                res[j] = next;
                j += 1;
            }
        }
    }

    res
}

/// Mirrors `igraph_i_create_start_vectors()`
/// (`src/graph/type_indexededgelist.c`): builds the start-index vector
/// (`os`/`is`) from an edge list and its order index.
fn create_start_vectors(el: &[i64], iindex: &[i64], nodes: i64) -> Vec<i64> {
    let no_of_edges = el.len() as i64;
    let edge = |i: i64| el[iindex[i as usize] as usize];
    let mut res = vec![0_i64; nodes as usize + 1];

    if no_of_edges == 0 {
        // Empty graph: all zeros.
        return res;
    }

    let mut idx: i64 = -1;
    for _ in 0..=edge(0) {
        idx += 1;
        res[idx as usize] = 0;
    }
    for i in 1..no_of_edges {
        let n = edge(i) - edge(res[idx as usize]);
        for _ in 0..n {
            idx += 1;
            res[idx as usize] = i;
        }
    }
    let j = edge(res[idx as usize]);
    for _ in 0..(nodes - j) {
        idx += 1;
        res[idx as usize] = no_of_edges;
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_undirected_graph_incidence() {
        // Triangle plus a self-loop at vertex 0.
        let g = Graph::new(3, false, &[(0, 1), (1, 2), (2, 0), (0, 0)]).unwrap();
        assert_eq!(g.vcount(), 3);
        assert_eq!(g.ecount(), 4);

        let inc0 = g.incident(0);
        // All four edges are incident to vertex 0; the self-loop appears
        // twice. Verified against C igraph: [3, 3, 0, 2] — the loop first
        // (once from the out-index, once from the in-index), then the two
        // non-loop edges.
        assert_eq!(inc0, vec![3, 3, 0, 2]);
        assert_eq!(inc0.iter().filter(|&&e| e == 3).count(), 2);

        let inc1 = g.incident(1);
        assert_eq!(inc1.len(), 2);
        assert!(inc1.contains(&0));
        assert!(inc1.contains(&1));

        let inc2 = g.incident(2);
        assert_eq!(inc2.len(), 2);
        assert!(inc2.contains(&1));
        assert!(inc2.contains(&2));
    }

    #[test]
    fn directed_graph_incidence_loops_twice() {
        let g = Graph::new(3, true, &[(0, 1), (1, 2), (0, 0)]).unwrap();
        // With mode == IGRAPH_ALL and IGRAPH_LOOPS_TWICE (what the Leiden
        // code uses), self-loops appear twice even for directed graphs.
        // Verified against C igraph.
        let inc0 = g.incident(0);
        assert_eq!(inc0, vec![2, 2, 0]);
        // Vertex 1: in-edge 0 (from 0), out-edge 1 (to 2).
        let inc1 = g.incident(1);
        assert_eq!(inc1.len(), 2);
        assert!(inc1.contains(&0));
        assert!(inc1.contains(&1));
        // Vertex 2: in-edge 1.
        assert_eq!(g.incident(2), vec![1]);
    }

    #[test]
    fn incidence_is_sorted_by_other_endpoint_within_direction() {
        let g = Graph::new(5, false, &[(0, 3), (0, 1), (0, 4), (0, 2)]).unwrap();
        // Verified against C igraph: edges are stored with from >= to, so
        // for node 0 the out-index is empty and the in-index holds all four
        // edges sorted by the "other" endpoint (from): 1, 2, 3, 4.
        let inc = g.incident(0);
        assert_eq!(inc, vec![1, 3, 0, 2]);
        assert!(inc.windows(2).all(|w| g.other(w[0], 0) <= g.other(w[1], 0)));
    }

    #[test]
    fn empty_graph() {
        let g = Graph::new(0, false, &[]).unwrap();
        assert_eq!(g.vcount(), 0);
        assert_eq!(g.ecount(), 0);

        let g = Graph::new(4, false, &[]).unwrap();
        for v in 0..4 {
            assert_eq!(g.incident(v), Vec::<i64>::new());
        }
    }
}
