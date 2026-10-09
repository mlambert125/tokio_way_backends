use crate::shm::*;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set when [`fake_previous_handler`] repaired a fault: the proof that a
/// foreign `SIGBUS` was chained to the displaced handler rather than
/// swallowed by the net or sent to the default action.
static PREVIOUS_RAN: AtomicBool = AtomicBool::new(false);

/// Stands in for a handler the application had before the net — Rust's
/// stack-overflow handler, a crash reporter. Repairs the page the same way
/// the net does, and records that it was the one who ran.
extern "C" fn fake_previous_handler(
    _signal: libc::c_int,
    info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    // SAFETY: the kernel hands a valid `siginfo_t` for a SIGBUS, and the
    // page patched over is one this test's own mapping faulted on.
    unsafe {
        let addr = (*info).si_addr() as usize;
        let page = read_system_page_size();
        let aligned = addr & !(page - 1);
        libc::mmap(
            aligned as *mut libc::c_void,
            page,
            libc::PROT_READ,
            libc::MAP_FIXED | libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
    }
    PREVIOUS_RAN.store(true, Ordering::Relaxed);
}

/// A memfd of `pages` pages with sealing never enabled, so the test can
/// shrink it under its mapping the way a misbehaving client would.
fn shrinkable_memfd(name: &std::ffi::CStr, pages: usize) -> RawFd {
    let len = i64::try_from(pages * read_system_page_size()).unwrap();
    // SAFETY: `memfd_create` takes a NUL-terminated name and no memory;
    // `ftruncate` takes the fd it returned.
    unsafe {
        let fd = libc::memfd_create(name.as_ptr(), 0);
        assert!(fd >= 0, "memfd_create failed");
        assert_eq!(libc::ftruncate(fd, len), 0, "ftruncate failed");
        fd
    }
}

/// One test rather than two, because the order matters process-wide: the
/// fake "previous" handler must be in place before the net installs over
/// it, and the `Once` only runs once per process.
#[test]
fn net_patches_own_pools_and_chains_foreign_faults() {
    let page = read_system_page_size();

    // The application's handler, in place before the net's.
    // SAFETY: `action` is fully initialised, and the handler async-signal-safe.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = fake_previous_handler as *const () as usize;
        action.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&raw mut action.sa_mask);
        assert_eq!(
            libc::sigaction(libc::SIGBUS, &raw const action, std::ptr::null_mut()),
            0
        );
    }

    // A pool mapped and registered by this library, then shrunk out from
    // under it: the net must patch the page to zeroes, not chain and not die.
    let fd = shrinkable_memfd(c"tokio-way-backends-test-pool", 2);
    let mapping =
        PoolMapping::new(fd, u32::try_from(2 * page).unwrap()).expect("pool should map");
    // SAFETY: fd is live; slice stays within the mapping, and the volatile
    // read is what springs the trap.
    unsafe {
        assert_eq!(libc::ftruncate(fd, 0), 0);
        let before = patched_pages();
        let bytes = mapping.slice(0, page).expect("in bounds");
        assert_eq!(std::ptr::read_volatile(bytes.as_ptr()), 0);
        assert!(patched_pages() > before, "the net should have patched");
        assert!(
            !PREVIOUS_RAN.load(Ordering::Relaxed),
            "a covered fault must not reach the previous handler"
        );
        libc::close(fd);
    }
    drop(mapping);

    // A mapping the net does not cover: the fault must arrive at the
    // handler that was there first.
    let fd = shrinkable_memfd(c"tokio-way-backends-test-foreign", 1);
    // SAFETY: a fresh private mapping of fd, unmapped below; the volatile
    // read faults and the fake handler patches it.
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            page,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            0,
        );
        assert_ne!(ptr, libc::MAP_FAILED);
        assert_eq!(libc::ftruncate(fd, 0), 0);
        assert_eq!(std::ptr::read_volatile(ptr.cast::<u8>()), 0);
        assert!(
            PREVIOUS_RAN.load(Ordering::Relaxed),
            "a foreign fault must chain to the previous handler"
        );
        libc::munmap(ptr, page);
        libc::close(fd);
    }
}
