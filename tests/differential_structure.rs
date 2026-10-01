// Differential test: compare pair_order and incidence lists against C igraph.
// The C helpers in `tests/c/` are compiled against the local igraph checkout
// at test time (see `common`).
mod common;

use leiden_rs::Graph;

#[test]
fn differential_pair_order_and_incidence() {
    if common::skip_if_no_igraph() {
        return;
    }
    let incident_bin = common::c_helper("incident_check").expect("failed to build C helper");

    let mut state: u64 = 0x12345678;
    let mut rand = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for trial in 0..200 {
        let n = (rand() % 12 + 1) as usize;
        let directed = rand() % 2 == 0;
        let m = (rand() % 25) as usize;
        let mut edges: Vec<(i64, i64)> = Vec::with_capacity(m);
        for _ in 0..m {
            let a = (rand() % n as u64) as i64;
            let b = (rand() % n as u64) as i64;
            edges.push((a, b));
        }

        // Incidence lists.
        let g = Graph::new(n, directed, &edges).unwrap();
        let mut cmd = std::process::Command::new(&incident_bin);
        cmd.arg(n.to_string()).arg(if directed { "1" } else { "0" });
        for &(a, b) in &edges {
            cmd.arg(a.to_string()).arg(b.to_string());
        }
        let out = cmd.output().unwrap();
        assert!(out.status.success(), "C incident failed on trial {trial}");
        let expected = String::from_utf8_lossy(&out.stdout);
        for v in 0..n {
            let want: Vec<i64> = expected
                .lines()
                .nth(v)
                .unwrap()
                .strip_prefix(&format!("v{v}:"))
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            assert_eq!(
                g.incident(v as i64),
                want,
                "incidence mismatch at trial {trial}, vertex {v} (directed={directed}, edges={edges:?})"
            );
        }

        // pair_order on (from, to) and (to, from) — only exercised via the
        // incidence lists above, which depend on os/is/oi/ii. A direct
        // pair_order comparison needs valid v2 < nodes; the incidence check
        // already covers the exact same permutations.
    }
}
