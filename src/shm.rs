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

/// Where the bytes of an uploaded texture live
#[derive(Debug)]
pub enum UploadPixels {
    /// Shared Memory Pixels (zerocopy through guard)
    Mapped {
        /// Guard keeping the mapping alive
        guard: Arc<BufferGuard>,
        /// Byte offset of the first pixel within the mapping
        offset: usize,
        /// Distance between rows, in bytes. Not necessarily `width * 4`
        stride: usize,
    },
    /// Compositor created pixels.  (Cursors, etc.)
    Owned(Box<[u8]>),
}

/// A per-buffer handle on the pool memory that buffer lives in
///
/// Readers hold a clone, so the compositor can tell they are finished by finding
/// itself the only owner left — which is what gates `wl_buffer.release`.
#[derive(Debug)]
pub struct BufferGuard {
    /// Reference-counted pool mapping, shared between backend and compositor
    mapping: Arc<PoolMapping>,
}

impl BufferGuard {
    /// Take a handle on a pool's memory for one buffer living in it
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

/// A live `mmap` of a client's shm pool, unmapped when the last user drops it
pub struct PoolMapping {
    /// A C pointer to the shared memory pool
    ptr: *mut libc::c_void,
    /// The size of the pool
    size: usize,
    /// Slot in the `SIGBUS` net covering this mapping, if one was free
    guard_slot: Option<usize>,
}
unsafe impl Send for PoolMapping {}
unsafe impl Sync for PoolMapping {}

impl PoolMapping {
    /// Map a pool's file, or return `None` if it cannot safely be mapped
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

    /// Borrow `len` bytes starting at `offset`
    ///
    /// # Safety
    ///
    /// The slice borrows memory a client can write to. The caller must only read
    /// while the `wl_buffer` this range belongs to is committed and unreleased,
    /// and must not keep the slice past the guard deferring that release.
    #[must_use]
    pub unsafe fn slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if offset.checked_add(len)? > self.size {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>().add(offset), len) })
    }
}

impl Drop for PoolMapping {
    /// Unregister the guard and munmap the pool
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

/// How many pool mappings the `SIGBUS` net can cover at once. Beyond this a mapping
/// works but is uncovered — a fixed table buys never allocating in a signal handler.
const MAX_GUARDED: usize = 512;

/// Marks a slot as claimed but not yet filled in. Never matches an address
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

/// Pages patched by the handler. Read from the compositor, which does the logging
static PATCHED: AtomicUsize = AtomicUsize::new(0);

/// Installs the handler exactly once, however many pools are mapped
static INSTALL: Once = Once::new();

/// The handler `SIGBUS` had before ours went in, chained to for every fault that is
/// not one of our pool mappings. Written once under [`INSTALL`], published through [`OLD_SAVED`].
static OLD_ACTION: OldActionCell = OldActionCell(UnsafeCell::new(MaybeUninit::uninit()));

/// Whether [`OLD_ACTION`] holds a real answer yet. Stored `Release` after the write, loaded `Acquire` before the read
static OLD_SAVED: AtomicBool = AtomicBool::new(false);

/// [`UnsafeCell`] wrapper so [`OLD_ACTION`] can be a static
struct OldActionCell(UnsafeCell<MaybeUninit<libc::sigaction>>);
// SAFETY: written exactly once, before OLD_SAVED is published; read only after
unsafe impl Sync for OldActionCell {}

/// The system page size, read once when the handler is installed, so the fault path does as little as possible
static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

/// One covered mapping
struct Slot {
    /// Where the mapping starts: `0` is free, [`CLAIMING`] is taken but not yet
    /// filled in, and anything else is a real base address
    base: AtomicUsize,
    /// Length of the mapping in bytes. Meaningful only once `base` holds a real address
    len: AtomicUsize,
}

/// Why a pool file cannot back the pool the client asked for
#[derive(Debug)]
pub enum PoolFileError {
    /// The file could not be inspected at all
    Unreadable,
    /// The client declared a pool larger than the file behind it
    TooSmall {
        /// The client-declared size of the pool file
        declared: u64,
        /// The actual size of the pool file
        actual: u64,
    },
}

impl std::fmt::Display for PoolFileError {
    /// The message a client's failure is logged with
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

/// How many pages have been replaced with zeroes after a `SIGBUS` — non-zero means
/// a client shrank a pool it had already committed from
pub fn patched_pages() -> usize {
    PATCHED.load(Ordering::Relaxed)
}

/// Cover a mapping with the `SIGBUS` net. Returns the slot to release later
pub fn register_shm(base: *mut libc::c_void, len: usize) -> Option<usize> {
    install_sigbus_guard();

    let base = base as usize;
    for (index, slot) in REGISTRY.iter().enumerate() {
        if slot
            .base
            .compare_exchange(0, CLAIMING, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            // Length first, then base: the handler reads base and only trusts the length once it has seen a real one
            slot.len.store(len, Ordering::Release);
            slot.base.store(base, Ordering::Release);
            return Some(index);
        }
    }
    warn!("shm guard table full; a pool mapping is unprotected against truncation");
    None
}

/// Release a slot, once its mapping is being torn down
///
/// The caller unmaps after this returns. Safe because a `PoolMapping` drops only
/// when its last `Arc` goes, so by then nobody can fault on it.
pub fn unregister_shm(index: usize) {
    let Some(slot) = REGISTRY.get(index) else {
        return;
    };
    // Base first, so the handler stops matching before the length goes
    slot.base.store(0, Ordering::Release);
    slot.len.store(0, Ordering::Release);
}

/// Whether an address is covered by the guarded regions
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

/// The system page size, as cached at install time
fn system_page_size() -> usize {
    match PAGE_SIZE.load(Ordering::Acquire) {
        0 => FALLBACK_PAGE_SIZE,
        size => size,
    }
}

/// Ask the system for its page size. Called once, off the fault path
pub(crate) fn read_system_page_size() -> usize {
    // SAFETY: `sysconf` takes no pointers and cannot fail meaningfully here
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(size).unwrap_or(FALLBACK_PAGE_SIZE).max(1)
}

/// Replace the unreadable page with zeroes and let the read retry
///
/// The patch is permanent: a private anonymous page goes down over what was a shared
/// mapping. An address that is not ours is chained to the previous handler, so
/// unrelated `SIGBUS` bugs still crash as the process intended. Everything here is
/// async-signal-safe: no allocation, no locks, no logging.
extern "C" fn on_sigbus(signal: libc::c_int, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    if !info.is_null() {
        // SAFETY: the kernel hands us a valid `siginfo_t` for a SIGBUS
        let addr = unsafe { (*info).si_addr() } as usize;
        if check_address_covered(addr) {
            let page = system_page_size();
            let aligned = addr & !(page - 1);
            // SAFETY: `aligned` is a page-aligned address inside a mapping we own
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
                // Returning retries the faulting instruction, which now reads zeroes
                return;
            }
        }
    }

    // Not one of ours, or beyond repair. Chain to whoever had the signal before us;
    // failing that, die exactly as we would have with no handler.
    // SAFETY: called from the signal handler, which is the contract below
    unsafe { forward_to_previous(signal, info, ctx) };
}

/// Hand a fault that is not ours to the handler that was here first
///
/// `SIG_DFL`, `SIG_IGN`, and the sliver before [`OLD_SAVED`] is published all fall
/// through to dying as an unhandled `SIGBUS` would.
///
/// # Safety
/// Must be called from a `SIGBUS` handler, with that handler's arguments.
unsafe fn forward_to_previous(
    signal: libc::c_int,
    info: *mut libc::siginfo_t,
    ctx: *mut libc::c_void,
) {
    if OLD_SAVED.load(Ordering::Acquire) {
        // SAFETY: OLD_SAVED being set means OLD_ACTION was fully written, and nothing writes it again
        let old = unsafe { (*OLD_ACTION.0.get()).assume_init() };
        let handler = old.sa_sigaction;
        if handler != libc::SIG_DFL && handler != libc::SIG_IGN {
            // SAFETY: the process registered this address as a signal handler, and `SA_SIGINFO` says which shape it has
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
    // SAFETY: both calls are async-signal-safe
    unsafe {
        libc::signal(libc::SIGBUS, libc::SIG_DFL);
        libc::raise(libc::SIGBUS);
    }
}

/// Put the `SIGBUS` net in place now instead of at the first pool map
///
/// What this controls is *when*, and so which previous handler gets saved and chained
/// to. Call it after a crash reporter installs itself. Idempotent.
pub fn install_sigbus_guard() {
    INSTALL.call_once(install_sigbus_handler);
}

/// Put the `SIGBUS` handler in place, and cache what it will need
fn install_sigbus_handler() {
    // Before the handler goes in, not after: once `sigaction` returns the handler can run
    PAGE_SIZE.store(read_system_page_size(), Ordering::Release);

    // SAFETY: `action` is fully initialised before use, and the handler is async-signal-safe
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        // `sa_sigaction` is a raw address; the cast has to go through a pointer
        action.sa_sigaction = on_sigbus as *const () as usize;
        // SA_ONSTACK: run on the alternate signal stack if the process set one up, so a
        // SIGBUS on an overflowed thread stack still has somewhere to run
        action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigemptyset(&raw mut action.sa_mask);
        let mut old: libc::sigaction = std::mem::zeroed();
        if libc::sigaction(libc::SIGBUS, &raw const action, &raw mut old) != 0 {
            warn!("could not install SIGBUS handler; a truncated pool will be fatal");
            return;
        }
        // Publish what was displaced, for the handler to chain to
        (*OLD_ACTION.0.get()).write(old);
        OLD_SAVED.store(true, Ordering::Release);
    }
}

/// Check a pool file is big enough, and stop it shrinking if the kernel lets us
///
/// Sealing is best-effort: it succeeds for a `memfd` created with `MFD_ALLOW_SEALING`
/// and fails harmlessly for anything else.
///
/// # Errors
///
/// If the pool is unreadable or too small
pub fn prepare_pool_file(fd: RawFd, size: u32) -> Result<(), PoolFileError> {
    // SAFETY: `fstat` only writes through the pointer it is given
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

    // SAFETY: `fcntl` with these commands takes an int and touches no memory
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
