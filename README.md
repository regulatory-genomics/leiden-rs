# leiden-rs

A standalone Rust port of [igraph](https://igraph.org)'s Leiden
community-detection implementation (`src/community/leiden.c`, commit
`8225c3a4b`), validated bit-for-bit against the C implementation.

## What is ported

- **`leiden()`** — port of `igraph_community_leiden()`: the generic
  interface taking explicit edge weights and vertex out/in weights.
- **`leiden_simple()`** — port of `igraph_community_leiden_simple()`: the
  simplified interface selecting one of three objective functions:
  generalized modularity, the Constant Potts Model (CPM), or an
  Erdős–Rényi (ER) null model.
- **`modularity()`** — port of `igraph_modularity()`.
- **`strength()`** — port of `igraph_strength()` for all vertices.
- **`Graph`** — the internal graph representation with igraph's exact
  `os`/`is`/`oi`/`ii` indexed-edge-list layout, including igraph's edge
  ordering conventions and incidence-list construction (self-loops appear
  twice with `IGRAPH_ALL` + `IGRAPH_LOOPS_TWICE`, as used by Leiden).
- **`Pcg32` + `Rng`** — igraph's default random number generator (PCG32),
  ported literally from `src/random/rng_pcg32.c` and
  `vendor/pcg/pcg_variants.h`, together with the generic sampling paths
  from `src/core/vector.c` / `src/random/random.c` (Lemire bounded
  integers, Fisher–Yates shuffle, 52-bit-mantissa uniforms).

The port is faithful to the C implementation, including its random number
stream: given the same seed and inputs, memberships are identical and
quality values are bit-for-bit identical to C igraph.

### Floating-point note

The reference C build must be compiled with `-ffp-contract=off` for exact
parity: with optimization enabled GCC contracts `a*b + c*d` into fused
multiply-add (FMA) instructions, which changes rounding in FP-boundary
cases. Rust never contracts floating-point operations, so the
`-ffp-contract=off` build is the canonical reference. All differential
tests and the golden output of igraph's own unit test
(`tests/unit/community_leiden.out`) reproduce exactly under this setting.

## Usage

```rust
use leiden_rs::{leiden_simple, Graph, Objective, Pcg32};

// Undirected graph: edges are stored with `from >= to` (igraph's
// `igraph_add_edges` convention); self-loops are allowed.
let graph = Graph::new(
    10,
    false,
    &[(0, 1), (0, 2), (0, 3), (0, 4), (5, 6), (5, 7), (0, 5)],
)
.unwrap();

let mut membership: Vec<i64> = Vec::new(); // filled with the singleton partition
let mut rng = Pcg32::seeded(123);
let outcome = leiden_simple(
    &graph,
    None,                // edge weights (None = unweighted)
    Objective::Modularity,
    1.0,                 // resolution parameter
    0.01,                // beta (refinement randomness)
    false,               // start from a given membership?
    2,                   // number of iterations (-1 = until stable)
    &mut membership,
    &mut rng,
)
.unwrap();

println!("{} clusters, quality {}", outcome.nb_clusters, outcome.quality);
```

For directed graphs the generic `leiden()` interface takes explicit vertex
out/in weight vectors; `leiden_simple()` computes them via `strength()`
just as `igraph_community_leiden_simple()` does.

## Tests

All gates are checked by `cargo test`:

- **Golden parity** (`tests/golden.rs`): a port of igraph's own unit test
  `tests/unit/community_leiden.c`, comparing printed quality (5 decimals)
  and membership against igraph's expected output, including the
  `test_last_level_moves_are_kept` regression case and the input-validation
  checks (exact igraph error messages).
- **Differential** (`tests/differential.rs`): fixed and randomized cases
  (60 random graphs × generic/simple interfaces × all three objectives)
  compared bit-for-bit against the C implementation.
- **Structure differential** (`tests/differential_structure.rs`): 200 random
  graphs, comparing incidence-list construction against C.
- **Properties** (`tests/properties.rs`): determinism, membership validity,
  cluster connectivity, quality non-decreasing with more iterations, and
  quality cross-checked against the independent `modularity()` port.
- **RNG stream** (`tests/rng_stream_test.rs`): the PCG32 stream matches C
  values for `get_integer`, and uniform sampling.

### Differential harness

The differential tests compile C helper programs from `tests/c/` against a
local igraph checkout at test time. The checkout is expected in a sibling
directory (`../igraph`) by default and must be built with
`-ffp-contract=off` (see above); override the location with the
`IGRAPH_DIR` environment variable. The differential tests are skipped
automatically when the reference checkout or a C compiler is unavailable,
so the crate builds and unit-tests standalone without igraph.

## Benchmarks

`cargo bench` runs `benches/compare.rs`, which compares this port against
the C implementation (`tests/c/leiden_bench.c`) on identical inputs:
Karate club, Erdős–Rényi random, planted-partition and Barabási–Albert
graphs (78 to ~100k edges, weighted and unweighted), across the
Modularity, CPM and ER objectives.

Measured results (Linux x86-64, both sides re-seeded per run, best/median
over 10–50 runs; reference igraph built with `-ffp-contract=off` so all
cluster counts match bit-for-bit and both sides do identical work):

| Graph | Edges | C best (µs) | Rust best (µs) | C/Rust |
|---|---|---|---|---|
| karate (modularity, 2 it) | 78 | 69 | 37 | 1.78× |
| random-1k (modularity, 2 it) | 10 178 | 7 578 | 5 723 | 1.32× |
| planted-2k (modularity, 2 it) | 18 923 | 11 128 | 8 354 | 1.33× |
| ba-1k (modularity, 2 it) | 3 990 | 3 414 | 2 557 | 1.34× |
| ba-10k (modularity, 2 it) | 39 990 | 45 848 | 35 762 | 1.28× |
| random-100k (modularity, 2 it) | 99 964 | 101 314 | 88 822 | 1.14× |
| planted-10k (modularity, 2 it) | 98 698 | 62 737 | 52 252 | 1.20× |
| random-100k-w (modularity, 2 it) | 99 964 | 115 485 | 97 543 | 1.19× |

Overall (median-weighted total across all 54 graph/objective/iteration
combinations): the Rust port is ≈ **1.20× faster** than C igraph, winning
on every measured case.

- The port stores the "other endpoint" of each edge directly in its CSR
  incidence list, so the hot refinement loops use a branch-free lookup
  instead of igraph's bounds-checked `other()` calls, and its scratch
  buffers are reused across the refinement loops; the level-0 incidence
  list is cached across iterations.
- The remaining gap to *even faster* is the sequential PCG32 RNG stream
  (inherent to bit-exactness) and Rust's bounds checks outside the
  hot loops.

Note: when the reference igraph is built with FMA contraction (default
`-O3`), the final cluster count can differ between the two sides on a few
CPM cases. This is a floating-point rounding artifact of the FMA build,
not a port defect — with the canonical `-ffp-contract=off` reference
build all cluster counts match bit-for-bit (see the differential tests).

## License

GPL-2.0-or-later, matching the igraph sources this port is derived from.
