//! GPU buffers shared as dma-buf file descriptors.  Import has to happen
//! on the backend thread

use std::{
    os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd},
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::scene_graph::PixelFormat;

/// `DRM_FORMAT_ARGB8888`: 32-bit BGRA, premultiplied alpha, little-endian.
///
/// A DRM fourcc, which is not the same numbering as `wl_shm`'s despite
/// describing the same bytes — `wl_shm` gives ARGB8888 and XRGB8888 the
/// special values 0 and 1, and every other format the fourcc.
pub const DRM_FORMAT_ARGB8888: u32 = fourcc(*b"AR24");
/// `DRM_FORMAT_XRGB8888`: as [`DRM_FORMAT_ARGB8888`], with the alpha byte
/// undefined and the image treated as opaque.
pub const DRM_FORMAT_XRGB8888: u32 = fourcc(*b"XR24");
/// The modifier meaning "no modifier": a plain linear layout, where a row is
/// `stride` bytes after the one above it and the extent can be worked out from
/// the outside.
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;
/// The modifier meaning "unspecified".
pub const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// The DRM device a renderer sits on, named by its device file.
///
/// The other half of the dma-buf handshake. Formats and modifiers say what a
/// backend can import; this says *where from*: buffers allocated on another
/// GPU import across devices only by luck. A compositor rendering some
/// content itself — the hybrid path, where it composes a group or an effect
/// into a GPU buffer and submits it as an ordinary scene element — opens GBM
/// on this path, so its allocations are same-device and the backend's import
/// is routine rather than a gamble.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderNode {
    /// Path to the device file, preferring the render node
    /// (`/dev/dri/renderD*`) over the primary node when the driver names
    /// both: a render node can be opened without DRM master, from any
    /// process.
    pub path: std::path::PathBuf,
}

/// A fence that signals when a buffer's producer has finished writing it.
///
/// Explicit sync for a [`DmabufImage`]. Implicit sync — the kernel ordering
/// GPU work on the buffer behind everyone's back — is what the absence of a
/// fence means, and it works on Mesa; drivers that do not do it (NVIDIA,
/// chiefly) hand over a *sync file* instead: a descriptor that becomes
/// readable when the producer's writes have landed. The producer is a client
/// speaking a syncobj protocol, or the compositor's own headless context on
/// the hybrid path (`EGL_ANDROID_native_fence_sync` exports one).
///
/// A backend must not sample the buffer before the fence signals. It waits
/// once per fence: waits are ordered into the context's command stream, so a
/// wait queued before the first draw covers every later one. The `waited`
/// flag is how that once is kept, shared across however many frames carry
/// the same fence.
#[derive(Debug)]
pub struct AcquireFence {
    /// The sync-file descriptor.
    fd: OwnedFd,
    /// Whether a wait has been queued already. Interior state rather than a
    /// consumed value because the scene holding the fence is shared and
    /// immutable.
    waited: AtomicBool,
}

impl AcquireFence {
    /// Wrap a sync-file descriptor as a fence.
    #[must_use]
    pub fn new(fd: OwnedFd) -> Self {
        Self {
            fd,
            waited: AtomicBool::new(false),
        }
    }

    /// The sync-file descriptor itself, for a backend to import or poll.
    #[must_use]
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// True exactly once, for whoever queues the wait. Later callers — the
    /// same element drawn on later frames — get `false`, because the wait
    /// already queued still orders their draws.
    pub fn needs_wait(&self) -> bool {
        !self.waited.swap(true, Ordering::AcqRel)
    }

    /// Block until the fence signals or the timeout (in milliseconds)
    /// passes, and say which it was. The CPU fallback for a driver with no
    /// GPU-side wait; an interrupted wait restarts with the full timeout.
    #[must_use]
    pub fn wait_blocking(&self, timeout_ms: i32) -> bool {
        poll_readable(self.fd.as_fd(), timeout_ms)
    }
}

/// Whether a descriptor becomes readable within the timeout — which for a
/// sync file means the fence signalled. An interrupted wait restarts with
/// the full timeout.
fn poll_readable(fd: BorrowedFd<'_>, timeout_ms: i32) -> bool {
    let mut pollfd = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        // SAFETY: `poll` writes only into the pollfd it is handed.
        let ret = unsafe { libc::poll(&raw mut pollfd, 1, timeout_ms) };
        if ret >= 0 {
            return ret > 0;
        }
        if std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
            return false;
        }
    }
}

/// A fence the *consumer* fills in: it signals when the backend's GPU reads
/// of a buffer have completed.
///
/// The release half of explicit sync, mirror to [`AcquireFence`]. The
/// producer attaches an empty cell to a submission; each frame the backend
/// draws sampling the buffer, it stores a fresh end-of-frame fence here,
/// replacing the last — fences on one command stream are ordered, so the
/// newest always covers every earlier read too.
///
/// The CPU-side half of release is unchanged: the producer still learns the
/// buffer is *structurally* free when its `Arc` comes back to sole
/// ownership. What this adds is the GPU-side half — at that moment the
/// backend's last draws may still be in flight, and this fence is what says
/// they have landed. Take it and wait, forward it to a client's syncobj
/// release point, or import it into another context's command stream. An
/// empty cell means no fenced reads happened: nothing to wait for beyond
/// implicit sync.
#[derive(Debug, Default)]
pub struct ReleaseFence {
    /// The newest fence covering the backend's reads, if any yet.
    fd: Mutex<Option<OwnedFd>>,
}

impl ReleaseFence {
    /// An empty cell, for the producer to attach to a submission.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Consumer side: store the newest fence, superseding any earlier one —
    /// on one command stream, later fences cover earlier reads.
    pub fn replace(&self, fd: OwnedFd) {
        *self.fd.lock().unwrap_or_else(PoisonError::into_inner) = Some(fd);
    }

    /// Producer side: take the newest fence, leaving the cell empty. `None`
    /// means no fenced reads are pending.
    #[must_use]
    pub fn take(&self) -> Option<OwnedFd> {
        self.fd
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// Block until the backend's reads have finished or the timeout (in
    /// milliseconds) passes, and say which it was. An empty cell is already
    /// finished. A wait that times out puts the fence back, so a later call
    /// can try again — unless a newer fence has landed meanwhile, which
    /// supersedes it anyway.
    #[must_use]
    pub fn wait_blocking(&self, timeout_ms: i32) -> bool {
        let Some(fd) = self.take() else {
            return true;
        };
        if poll_readable(fd.as_fd(), timeout_ms) {
            return true;
        }
        let mut slot = self.fd.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(fd);
        }
        false
    }
}

/// One plane of a dma-buf image.
#[derive(Debug)]
pub struct DmabufPlane {
    /// The descriptor this plane's memory lives in.
    pub fd: Arc<OwnedFd>,
    /// Byte offset of the plane within the descriptor.
    pub offset: u32,
    /// Distance between rows, in bytes.
    pub stride: u32,
}

/// A GPU buffer to be imported
#[derive(Debug)]
pub struct DmabufImage {
    /// Width in pixels.
    pub width: i32,
    /// Height in pixels.
    pub height: i32,
    /// DRM fourcc describing the pixel layout.
    pub fourcc: u32,
    /// DRM format modifier: the tiling and compression the pixels are stored
    /// with. [`DRM_FORMAT_MOD_INVALID`] means the buffer carries no claim.
    pub modifier: u64,
    /// The planes, in the order the format defines them.
    pub planes: Vec<DmabufPlane>,
}

impl DmabufImage {
    /// The fence covering this buffer's pending GPU writes, exported from the
    /// kernel's implicit-sync state, or `None` when there are none to wait
    /// for or the kernel predates the ioctl (5.16).
    ///
    /// This is the bridge for producers that never speak an explicit-sync
    /// protocol — a Vulkan client, say, whose WSI attaches its rendering
    /// fences to the dma-buf and assumes every reader honours them. GL
    /// sampling through an `EGLImage` does not reliably do so on its own,
    /// and the symptom is exactly what it sounds like: the compositor
    /// occasionally samples the buffer mid-clear and a window blinks blank
    /// for a frame.
    #[must_use]
    pub fn export_implicit_fence(&self) -> Option<std::os::fd::OwnedFd> {
        // Planes almost always share one descriptor; deduplicate so a
        // multi-plane format costs one ioctl, not three.
        let mut seen = Vec::new();
        let mut fence: Option<std::os::fd::OwnedFd> = None;
        for plane in &self.planes {
            let raw = std::os::fd::AsRawFd::as_raw_fd(&*plane.fd);
            if seen.contains(&raw) {
                continue;
            }
            seen.push(raw);
            // Asking as a reader: the fences handed back are the writers'.
            if let Some(exported) = export_sync_file(raw, DMA_BUF_SYNC_READ) {
                // The last one wins; in practice there is one descriptor.
                fence = Some(exported);
            }
        }
        fence
    }
}

/// `DMA_BUF_SYNC_READ`: the flag naming the reader's side of the implicit
/// sync contract in both ioctls below.
const DMA_BUF_SYNC_READ: u32 = 1;

/// `struct dma_buf_export_sync_file` / `dma_buf_import_sync_file`: one u32 of
/// flags and one descriptor, in both directions.
#[repr(C)]
struct DmaBufSyncFile {
    flags: u32,
    fd: i32,
}

/// `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`: `_IOWR('b', 2, struct dma_buf_export_sync_file)`.
const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: libc::c_ulong = 0xc008_6202;

/// Pull the current fences for `flags`-type access out of a dma-buf, as a
/// sync file. `None` for "nothing to wait on" as well as for any failure —
/// the caller cannot tell them apart and treats both as "sample and hope",
/// which is the pre-implicit-sync status quo.
fn export_sync_file(dmabuf_fd: i32, flags: u32) -> Option<std::os::fd::OwnedFd> {
    let mut arg = DmaBufSyncFile { flags, fd: -1 };
    // SAFETY: the ioctl reads and writes only `arg`, which lives across the
    // call; a non-negative returned descriptor is owned from that moment.
    unsafe {
        if libc::ioctl(dmabuf_fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &raw mut arg) != 0 {
            return None;
        }
        (arg.fd >= 0).then(|| std::os::fd::FromRawFd::from_raw_fd(arg.fd))
    }
}

/// A format a backend can import, and the modifiers it accepts for it.
#[derive(Debug, Clone)]
pub struct DmabufFormat {
    /// DRM fourcc.
    pub fourcc: u32,
    /// Modifiers accepted. Empty means the driver named none, and only [`DRM_FORMAT_MOD_INVALID`] will import.
    pub modifiers: Vec<u64>,
}

/// Build a fourcc from its four characters, as the DRM headers do.
#[must_use]
pub const fn fourcc(code: [u8; 4]) -> u32 {
    (code[0] as u32) | ((code[1] as u32) << 8) | ((code[2] as u32) << 16) | ((code[3] as u32) << 24)
}

/// Render a fourcc back into the four characters it was built from.
#[must_use]
pub fn fourcc_name(code: u32) -> String {
    code.to_le_bytes()
        .iter()
        .map(|&b| {
            if b.is_ascii_graphic() {
                char::from(b)
            } else {
                '?'
            }
        })
        .collect()
}

/// How a fourcc's pixels should be blended (alpha or no)
#[must_use]
pub fn pixel_format(fourcc: u32) -> PixelFormat {
    if fourcc == DRM_FORMAT_XRGB8888 {
        PixelFormat::Xrgb8888
    } else {
        PixelFormat::Argb8888
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the acquire fence's waiting rules. A pipe stands in for a
    //! sync file: both are descriptors that become readable on a signal.

    use super::*;
    use std::os::fd::FromRawFd;

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
}
