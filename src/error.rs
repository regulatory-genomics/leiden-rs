//! Error type mirroring igraph's error reporting for the Leiden code.
//!
//! Validation errors carry the same message text as their C counterparts
//! in `src/community/leiden.c` and the support routines, so failures are
//! recognizable when comparing against the C implementation.

use std::fmt;

/// An error returned by the Leiden implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeidenError {
    /// Error message, matching the corresponding igraph error message.
    pub message: String,
}

impl LeidenError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        LeidenError {
            message: message.into(),
        }
    }
}

impl fmt::Display for LeidenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for LeidenError {}

/// Format a float like C's `%g` (default precision 6), for error-message
/// parity with igraph's `IGRAPH_ERRORF` messages.
pub(crate) fn format_g(x: f64) -> String {
    if x.is_nan() {
        return if x.is_sign_negative() { "-nan" } else { "nan" }.to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    let neg = x < 0.0;
    let a = x.abs();
    // %g decides between fixed and scientific style using the exponent of
    // the value rounded to 6 significant digits, not of the raw value
    // (e.g. 999999.5 rounds to 1e+06, not 1000000).
    let rounded = format!("{a:.5e}"); // "d.ddddde<exp>", 6 significant digits
    let exp: i32 = rounded.rsplit('e').next().unwrap().parse().unwrap();
    let body = if (-4..6).contains(&exp) {
        // Fixed style, precision (6 - 1 - exp); strip trailing zeros.
        let prec = (5 - exp).max(0) as usize;
        let mut t = format!("{a:.prec$}");
        if t.contains('.') {
            while t.ends_with('0') {
                t.pop();
            }
            if t.ends_with('.') {
                t.pop();
            }
        }
        t
    } else {
        // Scientific style, precision 5; printf pads the exponent to at
        // least two digits with an explicit sign.
        let (m, e) = rounded.split_once('e').unwrap();
        let m = m.trim_end_matches('0').trim_end_matches('.');
        let ev: i32 = e.parse().unwrap();
        format!("{m}e{}{:02}", if ev < 0 { '-' } else { '+' }, ev.abs())
    };
    if neg {
        format!("-{body}")
    } else {
        body
    }
}

#[allow(dead_code)]
pub(crate) type Result<T> = std::result::Result<T, LeidenError>;
