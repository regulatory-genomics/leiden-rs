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
- **`leiden_parallel()` / `leiden_simple_parallel()`** — parallel variants
  of the two entry points (rayon-based), see
  [Parallel implementation](#parallel-implementation) below.
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

The parallel variant has the same signature and usage; swap the call to
`leiden_simple_parallel` (or `leiden_parallel` for the generic interface):

```rust
let outcome = leiden_rs::leiden_simple_parallel(
    &graph, None, Objective::Modularity, 1.0, 0.01, false, 2,
    &mut membership, &mut rng,
)
.unwrap();
```

For directed graphs the generic `leiden()` interface takes explicit vertex
out/in weight vectors; `leiden_simple()` computes them via `strength()`
just as `igraph_community_leiden_simple()` does.

## Resolution profile

`resolution_profile()` sweeps the resolution parameter over a range and
returns the optimal partition at each resolution, as `ProfileEntry` values
(resolution, number of communities, quality, internal edge weight and the
membership vector):

```rust
use leiden_rs::{resolution_profile, Graph, Objective, Pcg32};

let mut rng = Pcg32::seeded(42);
let profile = resolution_profile(
    &graph,
    None,                    // edge weights (None = unweighted)
    Objective::Cpm,          // objective function
    &[0.1, 0.5, 1.0, 2.0],   // resolutions to scan
    0.01,                    // beta (refinement randomness)
    -1,                      // number of iterations (-1 = until stable)
    &mut rng,
)
.unwrap();

for entry in &profile {
    println!(
        "γ={:.3}: {} communities (quality={:.4})",
        entry.resolution, entry.num_communities, entry.quality
    );
}
```

The sweep goes from low to high resolution and warm-starts each run from
the previous partition, so subsequent runs converge in far fewer iterations
than independent cold starts.

The **bisection sweep** (`resolution_profile_bisect()`, mirroring
leidenalg's `Optimiser.resolution_profile()`) instead binary-searches for
the resolutions where the partition actually changes — the optimal
partition is piecewise-constant in γ, so an interval stops being subdivided
when its endpoints have the same partition (up to `min_diff_bisect_value`
internal edges) or the resolution gap falls below `min_diff_resolution`
(logarithmic gap by default, `linear_bisection` for linear). This finds
every *distinct* partition over a wide γ range at a fraction of the cost of
a dense grid, which can miss jump points entirely:

```rust
use leiden_rs::{resolution_profile_bisect, Objective, Pcg32};

let mut rng = Pcg32::seeded(42);
let profile = resolution_profile_bisect(
    &graph,
    None,
    Objective::Cpm,
    (0.01, 10.0), // resolution range (low, high)
    0.01,         // beta
    -1,           // number of iterations
    1e-3,         // min_diff_resolution (bisection precision)
    1.0,          // min_diff_bisect_value (a single edge does not trigger)
    false,        // linear bisection? (false = logarithmic)
    &mut rng,
)
.unwrap();
```

Note that quality values are not comparable across entries with different
resolutions (the objective itself changes with γ); use the number of
communities, the internal edge weight, or a fixed-γ recomputation (e.g.
`modularity()`) to compare partitions across the profile.

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
  quality cross-checked against the independent `modularity()` port — for
  both the sequential and the parallel variants (including parallel
  determinism across thread-pool configurations).
- **RNG stream** (`tests/rng_stream_test.rs`): the PCG32 stream matches C
  values for `get_integer`, and uniform sampling.
- **Resolution profile** (`tests/profile.rs` and unit tests in
  `src/profile.rs`): linear-scan determinism and ordering, warm-start
  chaining matching manual `leiden_simple` calls, bisection finding every
  partition transition with few probes, quality cross-checked against the
  independent `modularity()` port, and input validation.

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

## Parallel implementation

`leiden_parallel()` and `leiden_simple_parallel()` run the same Leiden
algorithm with a rayon-parallel variant of the **cluster refinement**
phase (the step that decides how each cluster is split before
aggregation):

- The clusters of the current level are split into chunks of at least 32
  clusters each (at most `rayon::current_num_threads()` chunks, and only
  when there are enough clusters to justify it); smaller cluster counts
  are refined sequentially.
- Each chunk is refined independently by a rayon task that owns its
  scratch workspace (created once, reused across levels and iterations)
  and its own `Pcg32` sub-RNG, seeded from one 64-bit value drawn per
  chunk from the master RNG stream **in chunk order**.
- The per-chunk local cluster IDs are renumbered consecutively in cluster
  order, reproducing the sequential numbering semantics.

Because the chunk boundaries, seed assignment and renumbering depend only
on the cluster order — never on thread scheduling — the result is
**deterministic run-to-run** for a given thread configuration.

Guarantees (checked by the parallel property tests in
`tests/properties.rs`):

- valid partition: consecutive cluster IDs, every cluster (weakly)
  connected;
- per-iteration quality never decreases with more iterations;
- the returned quality matches the independent `modularity()` port;
- deterministic for a fixed thread-pool configuration;
- quality lands in the same ballpark as the sequential result.

**Not** bit-identical to the sequential `leiden()` / C igraph: the
refinement's random decisions are drawn from per-chunk streams instead of
a single stream, so the refined partitions (and hence the final
membership and quality) generally differ. The local-moving phase and the
aggregation step are unchanged (still sequential), and the refinement
phase is the dominant parallelizable component — the sequential
dependency chain of the local-moving loop is not parallelized.

Measured impact (`RAYON_NUM_THREADS=16`, 128-core Linux x86-64; timings
are load-sensitive, so treat ratios as approximate): the parallel
variant beats C igraph by up to **≈2×** on the cases where refinement
dominates (e.g. ba-10k CPM until stable: 2.0×; ba-1k CPM until stable:
1.4×; planted-10k-w CPM until stable: 1.3×; ba-10k modularity 2 it:
1.3×), and is roughly at parity with the sequential port on the
remaining cases — on some graphs a differently-seeded refinement
converges in more iterations, which can make the parallel variant
slower than sequential (e.g. random-100k-w CPM until stable). Choose
based on workload; the sequential port remains the bit-exact reference.
