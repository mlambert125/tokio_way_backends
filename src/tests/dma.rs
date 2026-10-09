//! Tests for the acquire fence's waiting rules. A pipe stands in for a
//! sync file: both are descriptors that become readable on a signal.

use crate::dma::*;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

fn fake_sync_file() -> (OwnedFd, OwnedFd) {
    let mut fds = [0; 2];
    // SAFETY: `pipe` writes two descriptors into the array it is given,
    // and each is owned by exactly one `OwnedFd` from here on.
    unsafe {
        assert_eq!(libc::pipe(fds.as_mut_ptr()), 0);
        (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1]))
    }
}

#[test]
fn a_fence_signals_when_its_descriptor_becomes_readable() {
    let (read, write) = fake_sync_file();
    let fence = AcquireFence::new(read);
    assert!(
        !fence.wait_blocking(0),
        "nothing has signalled yet, so the wait must time out"
    );
    // SAFETY: writing one byte into the pipe's own write end.
    let written = unsafe { libc::write(write.as_raw_fd(), [1u8].as_ptr().cast(), 1) };
    assert_eq!(written, 1);
    assert!(fence.wait_blocking(1000), "signalled now");
}

#[test]
fn the_wait_is_claimed_exactly_once() {
    let (read, _write) = fake_sync_file();
    let fence = AcquireFence::new(read);
    assert!(fence.needs_wait(), "the first frame queues the wait");
    assert!(!fence.needs_wait(), "every later frame is already ordered");
}

#[test]
fn an_empty_release_cell_has_nothing_to_wait_for() {
    let fence = ReleaseFence::new();
    assert!(fence.take().is_none());
    assert!(
        fence.wait_blocking(0),
        "no fenced reads pending means already released"
    );
}

#[test]
fn a_pending_release_waits_and_survives_a_timeout() {
    let (read, write) = fake_sync_file();
    let fence = ReleaseFence::new();
    fence.replace(read);
    assert!(
        !fence.wait_blocking(0),
        "the reads have not finished, so the wait must time out"
    );
    // The timed-out fence went back into the cell, so it can still be
    // waited for — and once it signals, the wait succeeds.
    // SAFETY: writing one byte into the pipe's own write end.
    let written = unsafe { libc::write(write.as_raw_fd(), [1u8].as_ptr().cast(), 1) };
    assert_eq!(written, 1);
    assert!(fence.wait_blocking(1000));
    assert!(
        fence.take().is_none(),
        "a successful wait consumes the fence"
    );
}

#[test]
fn a_newer_release_fence_supersedes_the_old() {
    let (first_read, _first_write) = fake_sync_file();
    let (second_read, second_write) = fake_sync_file();
    let expected = second_read.as_raw_fd();
    let fence = ReleaseFence::new();
    fence.replace(first_read);
    fence.replace(second_read);
    drop(second_write);
    // Only the newest is held: the first, never signalled, is gone, and
    // what comes out is the second.
    let held = fence.take().expect("the newest fence should be held");
    assert_eq!(held.as_raw_fd(), expected);
    assert!(fence.take().is_none());
}
