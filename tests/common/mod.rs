//! Shared helpers for the differential tests: locates the igraph checkout
//! and compiles the C helper programs in `tests/c/` against it.
//!
//! The igraph checkout is expected at a sibling directory of this crate
//! (`../igraph`) by default; override with the `IGRAPH_DIR` environment
//! variable. The reference igraph must be built (with
//! `-ffp-contract=off`, matching Rust's guaranteed lack of floating-point
//! contraction) so that `build/libigraph.a` exists.

use std::path::PathBuf;
use std::process::Command;

/// Path of the igraph checkout to use as the differential reference.
pub fn igraph_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("IGRAPH_DIR") {
        let dir = PathBuf::from(dir);
        return dir.exists().then_some(dir);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest.parent()?.join("igraph");
    dir.exists().then_some(dir)
}

/// Compile (or reuse a cached build of) the C helper `name` from `tests/c/`
/// against the igraph checkout. Returns `None` if the igraph checkout or a
/// C compiler is unavailable, so the differential tests can be skipped on
/// machines without the reference implementation.
pub fn c_helper(name: &str) -> Option<PathBuf> {
    let igraph = igraph_dir()?;

    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/c")
        .join(format!("{name}.c"));
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/c-helpers");
    let bin = out_dir.join(name);
    let flag_file = out_dir.join(format!("{name}.built"));

    // Reuse the previous build unless the source or the igraph library is
    // newer.
    let lib = igraph.join("build/src/libigraph.a");
    let up_to_date = bin.exists()
        && flag_file.exists()
        && [src.as_path(), lib.as_path()].iter().all(|p| {
            match (p.metadata(), flag_file.metadata()) {
                (Ok(s), Ok(f)) => s.modified().ok() <= f.modified().ok(),
                _ => false,
            }
        });
    if up_to_date {
        return Some(bin);
    }

    std::fs::create_dir_all(&out_dir).unwrap();
    let status = Command::new("cc")
        .arg(&src)
        .arg("-I")
        .arg(igraph.join("include"))
        .arg("-I")
        .arg(igraph.join("build/include"))
        .arg("-I")
        .arg(igraph.join("src"))
        .arg("-I")
        .arg(igraph.join("build/src"))
        .arg("-L")
        .arg(igraph.join("build/src"))
        .arg("-ligraph")
        .arg("-lm")
        .arg("-O2")
        .arg("-ffp-contract=off")
        .arg("-o")
        .arg(&bin)
        .status()
        .expect("failed to run cc");
    assert!(status.success(), "failed to compile C helper {name}");

    std::fs::write(&flag_file, b"ok").unwrap();
    Some(bin)
}

/// Run a compiled C helper with the given stdin, returning stdout.
#[allow(dead_code)]
pub fn run_helper(bin: &PathBuf, stdin: &str) -> String {
    use std::io::Write;
    let mut child = Command::new(bin)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn C helper");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Print a skip notice and return `true` when the reference igraph checkout
/// is unavailable.
#[allow(dead_code)]
pub fn skip_if_no_igraph() -> bool {
    if igraph_dir().is_none() {
        eprintln!(
            "skipping differential test: igraph checkout not found \
             (set IGRAPH_DIR to override)"
        );
        return true;
    }
    if c_helper_probe_fails() {
        eprintln!("skipping differential test: no C compiler available");
        return true;
    }
    false
}

#[allow(dead_code)]
fn c_helper_probe_fails() -> bool {
    Command::new("cc").arg("--version").output().is_err()
}
