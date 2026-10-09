//! When something happened, on the clock clients are told about

/// A `CLOCK_MONOTONIC` instant, in the shape `wp_presentation_feedback` wants
///
/// Input events carry one too, stamped by the backend when the event happens rather
/// than by the compositor when it reads it.
#[derive(Debug, Clone, Copy)]
pub struct MonotonicTimeStamp {
    /// Whole seconds part
    pub tv_sec: i64,
    /// Nanosecond part
    pub tv_nsec: i64,
}

impl MonotonicTimeStamp {
    /// Read the clock now
    #[must_use]
    pub fn now() -> Self {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `clock_gettime` only writes through the pointer it is given
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts) };
        Self {
            tv_sec: ts.tv_sec,
            tv_nsec: ts.tv_nsec,
        }
    }

    /// This instant in the milliseconds most wire events want
    ///
    /// The wire value is 32 bits with an unspecified base, so the truncation — a wrap
    /// every 49.7 days — is the protocol's own.
    #[must_use]
    pub fn to_wire_ms(self) -> u32 {
        // Both fields are non-negative for a CLOCK_MONOTONIC reading, and the arithmetic
        // wraps deliberately: the wire value wraps anyway.
        let ms = self
            .tv_sec
            .cast_unsigned()
            .wrapping_mul(1000)
            .wrapping_add(self.tv_nsec.cast_unsigned() / 1_000_000);
        #[allow(clippy::cast_possible_truncation)]
        {
            ms as u32
        }
    }
}
