//! GPU buffers shared as dma-buf file descriptors

use std::{
    os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd},
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::scene_graph::PixelFormat;

/// 32-bit BGRA, premultiplied alpha, little-endian
pub const DRM_FORMAT_ARGB8888: u32 = fourcc(*b"AR24");
/// 32-bit BGRX, no alpha, little-endian
pub const DRM_FORMAT_XRGB8888: u32 = fourcc(*b"XR24");
/// The modifier meaning "no modifier"
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;
/// The modifier meaning "unspecified".
pub const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// The DRM device a renderer sits on, named by its device file
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderNode {
    /// Path to the device file
    pub path: std::path::PathBuf,
}

/// A fence that signals when a buffer's producer has finished writing it
#[derive(Debug)]
pub struct AcquireFence {
    /// The sync-file descriptor
    fd: OwnedFd,
    /// Whether a wait has been queued already
    waited: AtomicBool,
}

impl AcquireFence {
    /// Wrap a sync-file descriptor as a fence
    #[must_use]
    pub fn new(fd: OwnedFd) -> Self {
        Self {
            fd,
            waited: AtomicBool::new(false),
        }
    }

    /// The sync-file descriptor itself, for a backend to import or poll
    #[must_use]
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// True exactly once, for whoever queues the wait
    pub fn needs_wait(&self) -> bool {
        !self.waited.swap(true, Ordering::AcqRel)
    }

    /// Block until the fence signals or the timeout (in milliseconds) passes, and say which it was
    #[must_use]
    pub fn wait_blocking(&self, timeout_ms: i32) -> bool {
        poll_readable(self.fd.as_fd(), timeout_ms)
    }
}

/// Whether a descriptor becomes readable within the timeout
fn poll_readable(fd: BorrowedFd<'_>, timeout_ms: i32) -> bool {
    let mut pollfd = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        let ret = unsafe { libc::poll(&raw mut pollfd, 1, timeout_ms) };
        if ret >= 0 {
            return ret > 0;
        }
        if std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
            return false;
        }
    }
}

/// A fence the *consumer* fills in: it signals when the backend's GPU reads of a buffer have completed.
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

    /// Consumer side: store the newest fence, superseding any earlier one
    pub fn replace(&self, fd: OwnedFd) {
        *self.fd.lock().unwrap_or_else(PoisonError::into_inner) = Some(fd);
    }

    /// Producer side: take the newest fence, leaving the cell empty
    #[must_use]
    pub fn take(&self) -> Option<OwnedFd> {
        self.fd
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// Block until the backend's reads have finished or the timeout passes
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
    /// Width in pixels
    pub width: i32,
    /// Height in pixels
    pub height: i32,
    /// DRM fourcc describing the pixel layout
    pub fourcc: u32,
    /// DRM format modifier: the tiling and compression the pixels are stored with
    pub modifier: u64,
    /// The planes, in the order the format defines them
    pub planes: Vec<DmabufPlane>,
}

impl DmabufImage {
    /// The fence covering this buffer's pending GPU writes
    #[must_use]
    pub fn export_implicit_fence(&self) -> Option<std::os::fd::OwnedFd> {
        // Planes almost always share one descriptor; deduplicate so a multi-plane format costs one ioctl, not three
        let mut seen = Vec::new();
        let mut fence: Option<std::os::fd::OwnedFd> = None;
        for plane in &self.planes {
            let raw = std::os::fd::AsRawFd::as_raw_fd(&*plane.fd);
            if seen.contains(&raw) {
                continue;
            }
            seen.push(raw);
            // Asking as a reader: the fences handed back are the writers'
            if let Some(exported) = export_sync_file(raw, DMA_BUF_SYNC_READ) {
                // The last one wins; in practice there is one descriptor
                fence = Some(exported);
            }
        }
        fence
    }
}

/// `DMA_BUF_SYNC_READ`: the flag naming the reader's side of the implicit sync contract in both ioctls below
const DMA_BUF_SYNC_READ: u32 = 1;

/// Flags and one descriptor, in both directions
#[repr(C)]
struct DmaBufSyncFile {
    flags: u32,
    fd: i32,
}

/// `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`: `_IOWR('b', 2, struct dma_buf_export_sync_file)`.
const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: libc::c_ulong = 0xc008_6202;

/// Pull the current fences for `flags`-type access out of a dma-buf, as a sync file
fn export_sync_file(dmabuf_fd: i32, flags: u32) -> Option<std::os::fd::OwnedFd> {
    let mut arg = DmaBufSyncFile { flags, fd: -1 };
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
