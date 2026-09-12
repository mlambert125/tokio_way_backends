use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    os::fd::RawFd,
    sync::{
        Arc, Once,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use tracing::{info, warn};

/// Where the bytes of an uploaded texture live.
///
/// Both kinds go up the same way — `tex_image_2d` from a slice — and differ
/// only in who owns the memory underneath.
#[derive(Debug)]
pub enum UploadPixels {
    /// Shared Memory Pixels (zerocopy through guard)
    Mapped {
        /// Guard keeping the mapping alive
        guard: Arc<BufferGuard>,
        /// Byte offset of the first pixel within the mapping.
        offset: usize,
        /// Distance between rows, in bytes. Not necessarily `width * 4`.
        stride: usize,
    },
    /// Compositor created pixels.  (Cursors, etc.)
    Owned(Box<[u8]>),
}

/// A per-buffer handle on the pool memory that buffer lives in.
///
/// This is what makes zero-copy safe. A texture handed to the backend borrows
/// the client's shm mapping instead of copying it, so `wl_buffer.release` — the
/// signal that the client may draw into the buffer again — must wait until
/// every reader has finished. Readers hold a clone of this, so the compositor
/// can tell they are finished by finding itself the only owner left.
///
/// Counting references rather than watching for a drop keeps the compositor's
/// own handle in place for as long as the buffer exists, so a client that
/// re-attaches a buffer it has not been told about yet still renders.
#[derive(Debug)]
pub struct BufferGuard {
    /// Reference counted pool mapping so that the backend and compositor
    /// can share this
    mapping: Arc<PoolMapping>,
}

impl BufferGuard {
    /// Take a handle on a pool's memory for one buffer living in it.
    #[must_use]
    pub fn new(mapping: Arc<PoolMapping>) -> Self {
        Self { mapping }
    }

    /// Accessor for retrieving a mapping
    #[must_use]
    pub fn mapping(&self) -> &PoolMapping {
        &self.mapping
    }
}

/// A live `mmap` of a client's shm pool, unmapped when the last user drops it.
pub struct PoolMapping {
    /// A C pointer to the shared memory pool
    ptr: *mut libc::c_void,
    /// The size of the pool
    size: usize,
    /// Slot in the `SIGBUS` net covering this mapping, if one was free.
    guard_slot: Option<usize>,
}
unsafe impl Send for PoolMapping {}
unsafe impl Sync for PoolMapping {}

impl PoolMapping {
    /// Map a pool's file, or return `None` if it cannot safely be mapped.
    pub fn new(fd: RawFd, size: u32) -> Option<Self> {
        if let Err(e) = prepare_pool_file(fd, size) {
            warn!("refusing to map shm pool: {e}");
            return None;
        }

        let size = size as usize;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return None;
        }
        Some(Self {
            ptr,
            size,
            guard_slot: register_shm(ptr, size),
        })
    }

    /// Size accessor
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Borrow `len` bytes starting at `offset`.
    ///
    /// # Safety
    ///
    /// The returned slice borrows memory a client process can write to
    /// whenever the protocol lets it. What makes reading sound is holding the
    /// protocol's side of the bargain: the caller must only read while the
    /// buffer is committed and unreleased — that is, while the `wl_buffer`
    /// this range belongs to has not been sent `release` — and must not keep
    /// the slice past the life of whatever guard is deferring that release.
    /// A client shrinking the pool under the mapping is the one violation the
    /// caller cannot prevent, and it is what the `SIGBUS` net below repairs.
    #[must_use]
    pub unsafe fn slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if offset.checked_add(len)? > self.size {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>().add(offset), len) })
    }
}

impl Drop for PoolMapping {
    /// Cleans up a pool by unregistering the guard and munmapping the pool
    fn drop(&mut self) {
        if let Some(slot) = self.guard_slot {
            unregister_shm(slot);
        }
        unsafe { libc::munmap(self.ptr, self.size) };
    }
}

impl std::fmt::Debug for PoolMapping {
    /// Debug print of the pool mapping
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolMapping")
            .field("ptr", &self.ptr)
            .field("size", &self.size)
            .field("guard_slot", &self.guard_slot)
            .finish()
    }
}

/// How many pool mappings the `SIGBUS` net can cover at once.
///
/// A client normally has one or two pools. Beyond this a mapping still works,
/// it is simply not covered, so the cap trades a fixed table for never
/// allocating or locking inside a signal handler.
const MAX_GUARDED: usize = 512;

/// Marks a slot as claimed but not yet filled in. Never matches an address,
/// because a slot only matches once its real base has been published.
const CLAIMING: usize = usize::MAX;

/// Fallback page size
const FALLBACK_PAGE_SIZE: usize = 4096;

/// A fixed registry of guarded memory slots
static REGISTRY: [Slot; MAX_GUARDED] = [const {
    Slot {
        base: AtomicUsize::new(0),
        len: AtomicUsize::new(0),
    }
}; MAX_GUARDED];

/// Pages patched by the handler. Read from the compositor, which does the
/// logging: a signal handler cannot.
static PATCHED: AtomicUsize = AtomicUsize::new(0);

/// Installs the handler exactly once, however many pools are mapped.
///
/// `sigaction` is process-wide, so the handler belongs to the process rather
/// than to any one mapping; the thousandth pool has nothing to install.
static INSTALL: Once = Once::new();

/// The handler `SIGBUS` had before ours went in, chained to for every fault
/// that is not one of our pool mappings.
///
/// Written once, under [`INSTALL`], and published through [`OLD_SAVED`]; the
/// handler reads it only after seeing that flag. `MaybeUninit` because
/// `libc::sigaction` has no const initialiser.
static OLD_ACTION: OldActionCell = OldActionCell(UnsafeCell::new(MaybeUninit::uninit()));

/// Whether [`OLD_ACTION`] holds a real answer yet. Stored `Release` after the
/// write, loaded `Acquire` before the read — the whole synchronisation story.
static OLD_SAVED: AtomicBool = AtomicBool::new(false);

/// [`UnsafeCell`] wrapper so [`OLD_ACTION`] can be a static.
struct OldActionCell(UnsafeCell<MaybeUninit<libc::sigaction>>);
// SAFETY: written exactly once, before OLD_SAVED is published; read only
// after. The flag's Release/Acquire pair orders the two.
unsafe impl Sync for OldActionCell {}

/// The system page size, read once when the handler is installed.
///
/// Read here rather than in the handler so the fault path does as little as
/// possible, and so the question of whether `sysconf` may be called from a
/// signal handler never has to be answered.
static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

/// One covered mapping.
struct Slot {
    /// Where the mapping starts — and, through two sentinel values, whether
    /// this slot holds one at all.
    ///
    /// `0` is free, [`CLAIMING`] is taken but not yet filled in, and anything
    /// else is a real base address. That is the whole of the lock-free
    /// protocol: `register` claims a slot by moving `base` off `0`, and only
    /// the final store of a real address publishes it to the handler.
    base: AtomicUsize,
    /// Length of the mapping in bytes.
    ///
    /// Meaningful only once `base` holds a real address. `register` writes it
    /// first and `base` second, so a slot caught mid-claim is never read as a
    /// range — the handler sees the sentinel and skips it.
    len: AtomicUsize,
}

/// Why a pool file cannot back the pool the client asked for.
#[derive(Debug)]
pub enum PoolFileError {
    /// The file could not be inspected at all.
    Unreadable,
    /// The client declared a pool larger than the file behind it.
    TooSmall {
        /// The client-declared size of the pool file
        declared: u64,
        /// The actual size of the pool file
        actual: u64,
    },
}

impl std::fmt::Display for PoolFileError {
    /// The message a client's failure is logged with. `Debug` is derived
    /// separately and reports the variant; this reports the numbers.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreadable => write!(f, "the pool file could not be inspected"),
            Self::TooSmall { declared, actual } => write!(
                f,
                "the client declared {declared} bytes but the file holds {actual}"
            ),
        }
    }
}

/// How many pages have been replaced with zeroes after a `SIGBUS`.
///
/// Non-zero means some client shrank a pool it had already committed from, and
/// whatever it was showing there is now black.
pub fn patched_pages() -> usize {
    PATCHED.load(Ordering::Relaxed)
}

/// Cover a mapping with the `SIGBUS` net. Returns the slot to release later.
pub fn register_shm(base: *mut libc::c_void, len: usize) -> Option<usize> {
    install_sigbus_guard();

    let base = base as usize;
    for (index, slot) in REGISTRY.iter().enumerate() {
        if slot
            .base
            .compare_exchange(0, CLAIMING, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            // Length first, then base: the handler reads base and only trusts
            // the length once it has seen a real one.
            slot.len.store(len, Ordering::Release);
            slot.base.store(base, Ordering::Release);
            return Some(index);
        }
    }
    warn!("shm guard table full; a pool mapping is unprotected against truncation");
    None
}

/// Release a slot, once its mapping is being torn down.
///
/// The caller unmaps *after* this returns, which leaves an instant where the
/// mapping is live but uncovered. That is safe for the one reason that matters:
/// a `PoolMapping` is dropped only when its last `Arc` goes, so by then nobody
/// holds it and nobody can fault on it.
pub fn unregister_shm(index: usize) {
    let Some(slot) = REGISTRY.get(index) else {
        return;
    };
    // Base first, so the handler stops matching before the length goes.
    slot.base.store(0, Ordering::Release);
    slot.len.store(0, Ordering::Release);
}

/// Checks whether or not an address is covered by the guarded regions
fn check_address_covered(addr: usize) -> bool {
    REGISTRY.iter().any(|slot| {
        let base = slot.base.load(Ordering::Acquire);
        if base == 0 || base == CLAIMING {
            return false;
        }
        let len = slot.len.load(Ordering::Acquire);
        addr >= base && addr < base.saturating_add(len)
    })
}

/// The system page size, as cached at install time.
fn system_page_size() -> usize {
    match PAGE_SIZE.load(Ordering::Acquire) {
        0 => FALLBACK_PAGE_SIZE,
        size => size,
    }
}

/// Ask the system for its page size. Called once, off the fault path.
fn read_system_page_size() -> usize {
    // SAFETY: `sysconf` takes no pointers and cannot fail meaningfully here.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(size).unwrap_or(FALLBACK_PAGE_SIZE).max(1)
}

/// Replace the unreadable page with zeroes and let the read retry.
///
/// The patch is permanent. A private anonymous page goes down over what was a
/// shared mapping, so that page is holed for the mapping's life — if the client
/// grows its file back, the region stays black.
///
/// An address that is not ours is handed to whatever handler was here before
/// this one — Rust's own runtime claims `SIGBUS` at startup for stack-overflow
/// detection, and a crash reporter may have too — so unrelated `SIGBUS` bugs
/// still crash, and crash through the machinery the process set up. A net
/// that swallowed foreign faults would hide real defects.
///
/// Everything here is async-signal-safe: atomic loads, `mmap`, a call into
/// the previous handler, `signal`, and `raise`. No allocation, no locks, no
/// logging.
extern "C" fn on_sigbus(signal: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    if !info.is_null() {
        // SAFETY: the kernel hands us a valid `siginfo_t` for a SIGBUS.
        let addr = unsafe { (*info).si_addr() } as usize;
        if check_address_covered(addr) {
            let page = system_page_size();
            let aligned = addr & !(page - 1);
            // SAFETY: `aligned` is a page-aligned address inside a mapping we
            // own, so replacing that one page cannot disturb anything else.
            let result = unsafe {
                libc::mmap(
                    aligned as *mut libc::c_void,
                    page,
                    libc::PROT_READ,
                    libc::MAP_FIXED | libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if result != libc::MAP_FAILED {
                PATCHED.fetch_add(1, Ordering::Relaxed);
                // Returning retries the faulting instruction, which now reads
                // zeroes instead of faulting again.
                return;
            }
        }
    }

    // Not one of ours, or beyond repair. Chain to whoever had the signal
    // before us; failing that, die exactly as we would have with no handler,
    // rather than looping on the same fault forever.
    // SAFETY: called from the signal handler, which is the contract below.
    unsafe { forward_to_previous(signal, info, ctx) };
}

/// Hand a fault that is not ours to the handler that was here first.
///
/// "Not ours" does not mean "nobody's": Rust's runtime registers a `SIGBUS`
/// handler at startup for stack-overflow detection, and crash reporters claim
/// the signal too. `SIG_DFL` — and `SIG_IGN`, which the kernel does not honour
/// for a hardware fault anyway — fall through to dying exactly as an unhandled
/// `SIGBUS` would. So does the sliver of time before [`OLD_SAVED`] is
/// published, when what was saved cannot yet be read.
///
/// # Safety
/// Must be called from a `SIGBUS` handler, with that handler's arguments.
unsafe fn forward_to_previous(
    signal: libc::c_int,
    info: *mut libc::siginfo_t,
    ctx: *mut libc::c_void,
) {
    if OLD_SAVED.load(Ordering::Acquire) {
        // SAFETY: OLD_SAVED being set means OLD_ACTION was fully written, and
        // nothing writes it again.
        let old = unsafe { (*OLD_ACTION.0.get()).assume_init() };
        let handler = old.sa_sigaction;
        if handler != libc::SIG_DFL && handler != libc::SIG_IGN {
            // SAFETY: the process registered this address as a signal handler,
            // and `SA_SIGINFO` says which of the two shapes it has.
            unsafe {
                if old.sa_flags & libc::SA_SIGINFO == 0 {
                    let previous: extern "C" fn(libc::c_int) = std::mem::transmute(handler);
                    previous(signal);
                } else {
                    let previous: extern "C" fn(
                        libc::c_int,
                        *mut libc::siginfo_t,
                        *mut libc::c_void,
                    ) = std::mem::transmute(handler);
                    previous(signal, info, ctx);
                }
            }
            return;
        }
    }
    // SAFETY: both calls are async-signal-safe.
    unsafe {
        libc::signal(libc::SIGBUS, libc::SIG_DFL);
        libc::raise(libc::SIGBUS);
    }
}

/// Put the `SIGBUS` net in place now instead of at the first pool map.
///
/// Mapping a pool installs the handler anyway; what this controls is *when*,
/// and so which previous handler gets saved and chained to for faults that
/// are not a pool mapping's. Call it early — after a crash reporter has
/// installed itself, say — if the handler in place when the first client maps
/// a pool might not be the one you mean. Idempotent, and cheap to call any
/// number of times.
pub fn install_sigbus_guard() {
    INSTALL.call_once(install_sigbus_handler);
}

/// Put the `SIGBUS` handler in place, and cache what it will need.
///
/// This takes `SIGBUS` for the whole process, saving the handler it displaces
/// so that faults outside the registry — including the stack overflows Rust's
/// own runtime watches this signal for — go on being handled by whoever was
/// handling them.
fn install_sigbus_handler() {
    // Before the handler goes in, not after: once `sigaction` returns the
    // handler can run, and it must never find this unset.
    PAGE_SIZE.store(read_system_page_size(), Ordering::Release);

    // SAFETY: `action` is fully initialised before use, and the handler is
    // async-signal-safe.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        // `sa_sigaction` is a raw address; the cast has to go through a
        // pointer rather than straight from the function item.
        action.sa_sigaction = on_sigbus as *const () as usize;
        // SA_ONSTACK: if the process set up an alternate signal stack — Rust's
        // runtime does, for its stack-overflow handler — run there, so a
        // SIGBUS on an overflowed thread stack still has somewhere to run and
        // the handler chained to runs on the stack it expects.
        action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigemptyset(&raw mut action.sa_mask);
        let mut old: libc::sigaction = std::mem::zeroed();
        if libc::sigaction(libc::SIGBUS, &raw const action, &raw mut old) != 0 {
            warn!("could not install SIGBUS handler; a truncated pool will be fatal");
            return;
        }
        // Publish what was displaced, for the handler to chain to. A foreign
        // SIGBUS in the instant before the store finds the flag unset and
        // takes the default action instead.
        (*OLD_ACTION.0.get()).write(old);
        OLD_SAVED.store(true, Ordering::Release);
    }
}

/// Check a pool file is big enough, and stop it shrinking if the kernel lets us.
///
/// Sealing is best-effort: it succeeds for a `memfd` created with
/// `MFD_ALLOW_SEALING`, which is what libwayland's shm helpers produce, and
/// fails harmlessly for anything else. A pool that ends up sealed can never
/// trigger the handler above, because it can never shrink.
///
/// # Errors
///
/// If the pool is unreadable or too small
pub fn prepare_pool_file(fd: RawFd, size: u32) -> Result<(), PoolFileError> {
    // SAFETY: `fstat` only writes through the pointer it is given.
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &raw mut stat) } != 0 {
        return Err(PoolFileError::Unreadable);
    }
    let actual = stat.st_size.max(0).cast_unsigned();
    if u64::from(size) > actual {
        return Err(PoolFileError::TooSmall {
            declared: u64::from(size),
            actual,
        });
    }

    // SAFETY: `fcntl` with these commands takes an int and touches no memory.
    unsafe {
        libc::fcntl(fd, libc::F_ADD_SEALS, libc::F_SEAL_SHRINK);
        let seals = libc::fcntl(fd, libc::F_GET_SEALS);
        if seals < 0 || seals & libc::F_SEAL_SHRINK == 0 {
            info!(
                "shm pool fd cannot be sealed against shrinking; relying on the SIGBUS net instead"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
