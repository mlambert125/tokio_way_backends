//! dma-buf import, through EGL.
//!
//! Turns the descriptors in a [`DmabufImage`](crate::dma::DmabufImage) into an `EGLImage`, which the
//! renderer then binds as an ordinary GL texture. Nothing is copied: the
//! texture samples the memory the client already rendered into.
//!
//! A buffer may have several planes without needing anything special to sample:
//! a compressed RGB layout carries its metadata in a second plane and is still
//! one texture. What would need a different sampler is YUV, and those layouts
//! are excluded where the formats are chosen rather than here.
//!
//! The entry points are all extensions, loaded by name through the same
//! `eglGetProcAddress` the rest of GL comes in on, so this needs no new
//! dependency — only the constants from the extension specs, which are
//! declared below because no crate in the tree carries them. What is used:
//!
//! - `EGL_EXT_image_dma_buf_import` — the import itself. Without it there is
//!   no dma-buf support at all and
//!   [`DmabufImporter::new`](crate::dmabuf_import::DmabufImporter::new) refuses.
//! - `EGL_EXT_image_dma_buf_import_modifiers` — enumerating what the driver
//!   will actually take. Without it the format list falls back to the two RGB
//!   formats every driver imports, with no modifier.
//! - `EGL_MESA_image_dma_buf_export` — the reverse direction, used only by
//!   [`DmabufImporter::self_test`](crate::dmabuf_import::DmabufImporter::self_test)
//!   to make a real dma-buf to import.
//!
//! Everything here must run on the thread holding the GL context, like the
//! rest of the renderer.

use glow::HasContext;
use std::ffi::{CStr, c_char, c_void};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::Arc;
use tracing::debug;

use crate::dma::{
    DRM_FORMAT_ARGB8888, DRM_FORMAT_MOD_INVALID, DRM_FORMAT_XRGB8888, DmabufFormat, DmabufImage,
    DmabufPlane, RenderNode, fourcc_name,
};

// EGL types, as the headers define them.
type EGLDisplay = *const c_void;
type EGLContext = *const c_void;
type EGLClientBuffer = *const c_void;
type EGLImageKHR = *const c_void;
type EGLBoolean = u32;
type EGLenum = u32;
type EGLint = i32;

// Core EGL.
const EGL_NONE: EGLint = 0x3038;
const EGL_TRUE: EGLint = 1;
const EGL_EXTENSIONS: EGLint = 0x3055;
const EGL_HEIGHT: EGLint = 0x3056;
const EGL_WIDTH: EGLint = 0x3057;
const EGL_NO_CONTEXT: EGLContext = std::ptr::null();
const EGL_NO_IMAGE_KHR: EGLImageKHR = std::ptr::null();

// EGL_KHR_image_base.
const EGL_IMAGE_PRESERVED_KHR: EGLint = 0x30D2;
// EGL_KHR_gl_image, for exporting a texture in the self-test.
const EGL_GL_TEXTURE_2D_KHR: EGLenum = 0x30B1;

// EGL_EXT_image_dma_buf_import, and the modifier attributes from
// EGL_EXT_image_dma_buf_import_modifiers.
const EGL_LINUX_DMA_BUF_EXT: EGLenum = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;

// EGL_KHR_fence_sync / EGL_ANDROID_native_fence_sync: importing a sync file
// as a fence the GPU can wait on.
const EGL_SYNC_NATIVE_FENCE_ANDROID: EGLenum = 0x3144;
const EGL_SYNC_NATIVE_FENCE_FD_ANDROID: EGLint = 0x3145;
const EGL_NO_SYNC_KHR: EGLSyncKHR = std::ptr::null();

/// An EGL sync object handle, from `EGL_KHR_fence_sync`.
type EGLSyncKHR = *const c_void;

// EGL_EXT_device_query: which device a display sits on.
const EGL_DEVICE_EXT: EGLint = 0x322C;
// EGL_EXT_device_drm: the device's primary node file.
const EGL_DRM_DEVICE_FILE_EXT: EGLint = 0x3233;
// EGL_EXT_device_drm_render_node: its render node file, the one to prefer.
const EGL_DRM_RENDER_NODE_FILE_EXT: EGLint = 0x3377;

/// An EGL attribute value, `intptr_t` in the headers.
type EGLAttrib = isize;
/// An opaque device handle from `EGL_EXT_device_query`.
type EGLDeviceEXT = *const c_void;

/// Per-plane attribute names: fd, offset, pitch, modifier low, modifier high.
///
/// More than one plane does not mean YUV. A compressed RGB layout — Intel's
/// `Y_TILED_*_CCS`, which is what a Vulkan swapchain on this hardware asks for
/// — puts its compression metadata in a second plane and is still sampled as
/// one ordinary texture. Formats that genuinely need `samplerExternalOES` are
/// excluded earlier, by the `external_only` filter in [`advertisable_format`].
///
/// A modifier is 64 bits and an attribute value is 32, which is why it goes in
/// halves. The enum values are not contiguous between planes, hence a table
/// rather than arithmetic.
const PLANE_ATTRIBUTES: [[EGLint; 5]; 4] = [
    [0x3272, 0x3273, 0x3274, 0x3443, 0x3444],
    [0x3275, 0x3276, 0x3277, 0x3445, 0x3446],
    [0x3278, 0x3279, 0x327A, 0x3447, 0x3448],
    [0x3440, 0x3441, 0x3442, 0x3449, 0x344A],
];

/// Side of the image [`DmabufImporter::self_test`] round-trips.
///
/// Small enough to be free, big enough that a stride mistake shows up as a
/// shear rather than passing by luck.
const SELF_TEST_SIDE: i32 = 4;

/// Formats assumed importable when the driver will not enumerate them.
///
/// The base import extension has no way to ask, so this is the floor every
/// driver implementing it meets: the two 8-bit RGB layouts, with no modifier.
const ASSUMED_FORMATS: [u32; 2] = [DRM_FORMAT_ARGB8888, DRM_FORMAT_XRGB8888];

type PfnGetCurrentContext = unsafe extern "system" fn() -> EGLContext;
type PfnQueryString = unsafe extern "system" fn(EGLDisplay, EGLint) -> *const c_char;
type PfnQueryDisplayAttribExt =
    unsafe extern "system" fn(EGLDisplay, EGLint, *mut EGLAttrib) -> EGLBoolean;
type PfnQueryDeviceStringExt = unsafe extern "system" fn(EGLDeviceEXT, EGLint) -> *const c_char;
type PfnCreateSyncKhr =
    unsafe extern "system" fn(EGLDisplay, EGLenum, *const EGLint) -> EGLSyncKHR;
type PfnDestroySyncKhr = unsafe extern "system" fn(EGLDisplay, EGLSyncKHR) -> EGLBoolean;
type PfnWaitSyncKhr = unsafe extern "system" fn(EGLDisplay, EGLSyncKHR, EGLint) -> EGLint;
type PfnDupNativeFenceFd = unsafe extern "system" fn(EGLDisplay, EGLSyncKHR) -> EGLint;
type PfnCreateImageKhr = unsafe extern "system" fn(
    EGLDisplay,
    EGLContext,
    EGLenum,
    EGLClientBuffer,
    *const EGLint,
) -> EGLImageKHR;
type PfnDestroyImageKhr = unsafe extern "system" fn(EGLDisplay, EGLImageKHR) -> EGLBoolean;
type PfnImageTargetTexture2DOes = unsafe extern "system" fn(EGLenum, EGLImageKHR);
type PfnQueryDmabufFormats =
    unsafe extern "system" fn(EGLDisplay, EGLint, *mut EGLint, *mut EGLint) -> EGLBoolean;
type PfnQueryDmabufModifiers = unsafe extern "system" fn(
    EGLDisplay,
    EGLint,
    EGLint,
    *mut u64,
    *mut EGLBoolean,
    *mut EGLint,
) -> EGLBoolean;
type PfnExportDmabufQueryMesa = unsafe extern "system" fn(
    EGLDisplay,
    EGLImageKHR,
    *mut EGLint,
    *mut EGLint,
    *mut u64,
) -> EGLBoolean;
type PfnExportDmabufMesa = unsafe extern "system" fn(
    EGLDisplay,
    EGLImageKHR,
    *mut EGLint,
    *mut EGLint,
    *mut EGLint,
) -> EGLBoolean;

/// Everything a backend can say about its dma-buf import path, in one answer.
///
/// The three fields are three different questions: *where* imports land
/// (`device`), *what* imports (`formats`), and *whether the path actually
/// works* (`probe`). A compositor advertising `zwp_linux_dmabuf_v1` reads
/// the last two; one allocating buffers of its own — the hybrid rendering
/// path — reads the first.
#[derive(Debug, Clone)]
pub struct DmabufCapabilities {
    /// The DRM device imports happen on, if EGL would name it. `None` on a
    /// driver without the device-query extensions, which costs the hybrid
    /// path its guarantee but changes nothing else.
    pub device: Option<RenderNode>,
    /// Formats and modifiers that can be imported.
    pub formats: Vec<DmabufFormat>,
    /// What came of actually trying an import.
    pub probe: DmabufImportProbeResult,
}

/// What happened when a backend tried its import path end to end.
#[derive(Debug, Clone)]
pub enum DmabufImportProbeResult {
    /// A dma-buf was imported and read back with the pixels it went in with.
    Passed,
    /// Backend without GPU or driver without extension to import dma buffers
    Unsupported(String),
    /// Could not test, might work, might not
    Untested(String),
    /// The path exists and failed.
    Failed(String),
}

/// An `EGLImage`, destroyed when dropped.
///
/// Held alongside the GL texture it was bound to rather than destroyed after
/// binding. The two are siblings referring to the same memory, and keeping the
/// handle is what lets the texture be rebound — and makes the lifetime of the
/// import something the cache can reason about instead of a side effect.
pub struct EglImage {
    /// The image handle.
    raw: EGLImageKHR,
    /// The display it belongs to, needed to destroy it.
    display: EGLDisplay,
    /// `eglDestroyImageKHR`, kept here so dropping needs no other context.
    destroy: PfnDestroyImageKhr,
}

impl Drop for EglImage {
    fn drop(&mut self) {
        // SAFETY: the handle came from `eglCreateImageKHR` on this display and
        // is destroyed exactly once, here.
        unsafe { (self.destroy)(self.display, self.raw) };
    }
}

/// The EGL entry points needed to import a dma-buf, resolved once.
pub struct DmabufImporter {
    /// The display every call is made against.
    display: EGLDisplay,
    /// `eglCreateImageKHR`.
    create_image: PfnCreateImageKhr,
    /// `eglDestroyImageKHR`.
    destroy_image: PfnDestroyImageKhr,
    /// `glEGLImageTargetTexture2DOES`, which is GL rather than EGL but comes
    /// in through the same loader.
    image_target_texture: PfnImageTargetTexture2DOes,
    /// `eglQueryDmaBufFormatsEXT` and `eglQueryDmaBufModifiersEXT`, absent
    /// unless the modifiers extension is there.
    query: Option<(PfnQueryDmabufFormats, PfnQueryDmabufModifiers)>,
    /// `eglExportDMABUFImageQueryMESA` and `eglExportDMABUFImageMESA`, used
    /// only by [`Self::self_test`].
    export: Option<(PfnExportDmabufQueryMesa, PfnExportDmabufMesa)>,
    /// `eglGetCurrentContext`, likewise: exporting a texture needs the context
    /// that owns it.
    current_context: Option<PfnGetCurrentContext>,
    /// The DRM device this display sits on, if EGL would name it. Learned
    /// once at construction — it cannot change while the display lives.
    render_node: Option<RenderNode>,
    /// `eglCreateSyncKHR`, `eglDestroySyncKHR`, `eglWaitSyncKHR` and
    /// `eglDupNativeFenceFDANDROID`, present only when the driver can move
    /// fences across the sync-file boundary in both directions —
    /// `EGL_KHR_fence_sync`, `EGL_KHR_wait_sync` and
    /// `EGL_ANDROID_native_fence_sync` together.
    sync: Option<(
        PfnCreateSyncKhr,
        PfnDestroySyncKhr,
        PfnWaitSyncKhr,
        PfnDupNativeFenceFd,
    )>,
}

impl DmabufImporter {
    /// Resolve the entry points, or say why dma-buf import is unavailable.
    ///
    /// # Safety
    /// `display` must be the live `EGLDisplay` the GL context was made on, and
    /// `loader` must resolve symbols for it. The importer must then be used
    /// only while that context is current on this thread.
    ///
    /// # Errors
    /// Errors if anything went wrong creating the importer
    pub unsafe fn new(
        display: EGLDisplay,
        loader: &dyn Fn(&CStr) -> *const c_void,
    ) -> Result<Self, String> {
        if display.is_null() {
            return Err("no EGL display: this backend is not running on EGL".into());
        }

        // SAFETY: `eglQueryString` is core EGL and the display is live.
        let (query_string, extensions) = unsafe {
            let query: PfnQueryString = load(loader, c"eglQueryString")
                .ok_or_else(|| String::from("eglQueryString is missing"))?;
            let raw = query(display, EGL_EXTENSIONS);
            let extensions = if raw.is_null() {
                String::new()
            } else {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            };
            (query, extensions)
        };

        if !has_extension(&extensions, "EGL_EXT_image_dma_buf_import") {
            return Err("EGL_EXT_image_dma_buf_import is not supported by this driver".into());
        }

        // SAFETY: the extension is present, so these entry points exist.
        let (create_image, destroy_image, image_target_texture) = unsafe {
            (
                load::<PfnCreateImageKhr>(loader, c"eglCreateImageKHR")
                    .ok_or_else(|| String::from("eglCreateImageKHR is missing"))?,
                load::<PfnDestroyImageKhr>(loader, c"eglDestroyImageKHR")
                    .ok_or_else(|| String::from("eglDestroyImageKHR is missing"))?,
                load::<PfnImageTargetTexture2DOes>(loader, c"glEGLImageTargetTexture2DOES")
                    .ok_or_else(|| String::from("glEGLImageTargetTexture2DOES is missing"))?,
            )
        };

        // SAFETY: each is loaded only after its extension is confirmed, and a
        // null return leaves the pair `None` rather than a dangling pointer.
        let query = has_extension(&extensions, "EGL_EXT_image_dma_buf_import_modifiers")
            .then(|| unsafe {
                Some((
                    load::<PfnQueryDmabufFormats>(loader, c"eglQueryDmaBufFormatsEXT")?,
                    load::<PfnQueryDmabufModifiers>(loader, c"eglQueryDmaBufModifiersEXT")?,
                ))
            })
            .flatten();
        let export = has_extension(&extensions, "EGL_MESA_image_dma_buf_export")
            .then(|| unsafe {
                Some((
                    load::<PfnExportDmabufQueryMesa>(loader, c"eglExportDMABUFImageQueryMESA")?,
                    load::<PfnExportDmabufMesa>(loader, c"eglExportDMABUFImageMESA")?,
                ))
            })
            .flatten();
        // SAFETY: core EGL, and optional here — its absence only costs the
        // self-test.
        let current_context =
            unsafe { load::<PfnGetCurrentContext>(loader, c"eglGetCurrentContext") };

        // All three sync extensions or none: a fence that can be created but
        // not waited on, or waited on but not made from a sync file, is no
        // use to anyone.
        // SAFETY: each entry point is loaded behind the extension defining it.
        let sync = (has_extension(&extensions, "EGL_KHR_fence_sync")
            && has_extension(&extensions, "EGL_KHR_wait_sync")
            && has_extension(&extensions, "EGL_ANDROID_native_fence_sync"))
        .then(|| unsafe {
            Some((
                load::<PfnCreateSyncKhr>(loader, c"eglCreateSyncKHR")?,
                load::<PfnDestroySyncKhr>(loader, c"eglDestroySyncKHR")?,
                load::<PfnWaitSyncKhr>(loader, c"eglWaitSyncKHR")?,
                load::<PfnDupNativeFenceFd>(loader, c"eglDupNativeFenceFDANDROID")?,
            ))
        })
        .flatten();

        // SAFETY: every entry point is loaded behind its extension check,
        // and the display is live. Failure only costs the device name.
        let render_node = unsafe { query_render_node(display, loader, query_string) };

        Ok(Self {
            display,
            create_image,
            destroy_image,
            image_target_texture,
            query,
            export,
            current_context,
            render_node,
            sync,
        })
    }

    /// Queue a GPU-side wait on a sync file: nothing later in this context's
    /// command stream runs before the fence signals, and the CPU does not
    /// block. Returns `false` when the driver has no sync path or refused
    /// the descriptor — the caller falls back to a CPU wait.
    ///
    /// The descriptor is duplicated, because EGL adopts the one it is given
    /// and the fence's own must survive for later frames.
    ///
    /// # Safety
    /// The GL context must be current on this thread — the wait is queued
    /// into *its* command stream — and the display must be live.
    #[must_use]
    pub unsafe fn wait_native_fence(&self, fence: std::os::fd::BorrowedFd<'_>) -> bool {
        let Some((create_sync, destroy_sync, wait_sync, _)) = self.sync else {
            return false;
        };
        let Ok(duplicate) = fence.try_clone_to_owned() else {
            return false;
        };
        // SAFETY: the attribute list is NONE-terminated; the descriptor is
        // live, and ownership of the duplicate passes to EGL exactly when
        // creation succeeds.
        unsafe {
            let attributes = [
                EGL_SYNC_NATIVE_FENCE_FD_ANDROID,
                duplicate.as_raw_fd(),
                EGL_NONE,
            ];
            let sync = create_sync(
                self.display,
                EGL_SYNC_NATIVE_FENCE_ANDROID,
                attributes.as_ptr(),
            );
            if sync == EGL_NO_SYNC_KHR {
                // EGL did not adopt the duplicate; dropping it closes it.
                return false;
            }
            std::mem::forget(duplicate);
            let waited = wait_sync(self.display, sync, 0) == EGL_TRUE;
            // Destroying a sync with a wait pending is defined: the wait
            // already queued stands, only the handle goes.
            destroy_sync(self.display, sync);
            waited
        }
    }

    /// Insert a fence into the current context's command stream and export
    /// it as a sync file that signals when everything queued before it has
    /// completed — the raw material of a release fence. `None` when the
    /// driver has no sync path or refuses.
    ///
    /// The flush in the middle is load-bearing: the native fence object is
    /// only materialised by the next flush after the sync is created, so
    /// create, flush, *then* dup — dup before flush fails.
    ///
    /// # Safety
    /// The GL context must be current on this thread — the fence lands in
    /// *its* command stream — `gl` must be its loaded entry points, and the
    /// display must be live.
    #[must_use]
    pub unsafe fn export_native_fence(&self, gl: &glow::Context) -> Option<OwnedFd> {
        let (create_sync, destroy_sync, _, dup_fd) = self.sync?;
        // SAFETY: the attribute list is NONE-terminated, and a non-negative
        // descriptor from the dup is owned from the moment it is returned.
        unsafe {
            let attributes = [EGL_NONE];
            let sync = create_sync(
                self.display,
                EGL_SYNC_NATIVE_FENCE_ANDROID,
                attributes.as_ptr(),
            );
            if sync == EGL_NO_SYNC_KHR {
                return None;
            }
            gl.flush();
            let fd = dup_fd(self.display, sync);
            destroy_sync(self.display, sync);
            (fd >= 0).then(|| OwnedFd::from_raw_fd(fd))
        }
    }

    /// The DRM device this importer's display sits on, or `None` if EGL
    /// would not name it. The allocation target for a compositor making its
    /// own GPU buffers — see [`RenderNode`].
    #[must_use]
    pub fn render_node(&self) -> Option<&RenderNode> {
        self.render_node.as_ref()
    }

    /// The formats this driver will import, with their modifiers.
    ///
    /// Modifiers the driver marks `external_only` are dropped: they can only be
    /// sampled through `samplerExternalOES`, which this renderer has no program
    /// for. A format left with none of its modifiers after that filtering goes
    /// too — on a Mesa driver that is every YUV layout, and offering one would
    /// invite a buffer that imports cleanly and then cannot be drawn. They come
    /// back when there is an external-sampler program to draw them with.
    #[must_use]
    pub fn formats(&self) -> Vec<DmabufFormat> {
        let Some((query_formats, query_modifiers)) = self.query else {
            // No way to ask. Everything implementing the base extension takes
            // these two with an implicit layout.
            return ASSUMED_FORMATS
                .iter()
                .map(|&fourcc| DmabufFormat {
                    fourcc,
                    modifiers: Vec::new(),
                })
                .collect();
        };

        // SAFETY: the entry points came from the extension that defines them,
        // and each call is made with a count matching the buffer handed over —
        // first with none, to learn the count, then with exactly that many.
        let fourccs = unsafe {
            let mut count: EGLint = 0;
            if query_formats(self.display, 0, std::ptr::null_mut(), &raw mut count) != 1
                || count <= 0
            {
                return Vec::new();
            }
            let mut fourccs = vec![0 as EGLint; count.unsigned_abs() as usize];
            if query_formats(self.display, count, fourccs.as_mut_ptr(), &raw mut count) != 1 {
                return Vec::new();
            }
            fourccs.truncate(count.unsigned_abs() as usize);
            fourccs
        };

        fourccs
            .into_iter()
            .filter_map(|fourcc| {
                // SAFETY: as above.
                let enumerated =
                    unsafe { enumerate_modifiers(query_modifiers, self.display, fourcc) };
                advertisable_format(fourcc.cast_unsigned(), &enumerated)
            })
            .collect()
    }

    /// Import a dma-buf as an `EGLImage`.
    ///
    /// Failure is ordinary rather than exceptional: a client can describe a
    /// buffer this driver will not take, and the answer is to not draw it.
    ///
    /// # Errors
    ///
    /// Returns an error if the image can't be imported
    pub fn import(&self, image: &DmabufImage) -> Result<EglImage, String> {
        if image.planes.is_empty() || image.planes.len() > PLANE_ATTRIBUTES.len() {
            return Err(format!("{} plane(s) is not a buffer", image.planes.len()));
        }
        if image.width <= 0 || image.height <= 0 {
            return Err(format!("empty image: {}x{}", image.width, image.height));
        }

        let mut attributes = vec![
            EGL_WIDTH,
            image.width,
            EGL_HEIGHT,
            image.height,
            EGL_LINUX_DRM_FOURCC_EXT,
            image.fourcc.cast_signed(),
        ];
        // An explicit modifier is only legal to pass when the driver
        // understands modifiers at all; without that extension the layout is
        // whatever the two sides agreed out of band. It goes on every plane,
        // which is what the client described and what EGL expects.
        let explicit = (image.modifier != DRM_FORMAT_MOD_INVALID && self.query.is_some())
            .then_some(image.modifier);
        for (plane, names) in image.planes.iter().zip(PLANE_ATTRIBUTES) {
            let [fd, offset, pitch, modifier_lo, modifier_hi] = names;
            attributes.extend_from_slice(&[
                fd,
                plane.fd.as_raw_fd(),
                offset,
                plane.offset.cast_signed(),
                pitch,
                plane.stride.cast_signed(),
            ]);
            if let Some(modifier) = explicit {
                attributes.extend_from_slice(&[
                    modifier_lo,
                    u32::try_from(modifier & 0xffff_ffff)
                        .unwrap_or(0)
                        .cast_signed(),
                    modifier_hi,
                    u32::try_from(modifier >> 32).unwrap_or(0).cast_signed(),
                ]);
            }
        }
        attributes.push(EGL_NONE);

        // SAFETY: the attribute list is NONE-terminated and every descriptor in
        // it is alive for the call, held by the `DmabufImage` being imported.
        // EGL copies what it needs; the image does not borrow the list.
        let raw = unsafe {
            (self.create_image)(
                self.display,
                EGL_NO_CONTEXT,
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null(),
                attributes.as_ptr(),
            )
        };
        if raw == EGL_NO_IMAGE_KHR {
            return Err(format!(
                "driver refused a {}x{} {} buffer with modifier {:#x}",
                image.width,
                image.height,
                fourcc_name(image.fourcc),
                image.modifier,
            ));
        }
        Ok(EglImage {
            raw,
            display: self.display,
            destroy: self.destroy_image,
        })
    }

    /// Point the currently bound `TEXTURE_2D` at an imported image.
    ///
    /// # Safety
    /// A texture must be bound to `TEXTURE_2D` on the current context, and the
    /// image must outlive every draw that samples it.
    pub unsafe fn bind_to_texture(&self, image: &EglImage) {
        // SAFETY: delegated to this function's own contract.
        unsafe { (self.image_target_texture)(glow::TEXTURE_2D, image.raw) };
    }

    /// Check the import path end to end: export a texture the compositor made
    /// as a dma-buf, import that back, and read the pixels out again.
    ///
    /// Advertising dma-buf on the strength of an extension string is how a
    /// compositor ends up telling clients to allocate buffers it then cannot
    /// draw. This is the difference between "the entry points are there" and
    /// "it works", and it costs one 4x4 texture at startup.
    ///
    /// # Safety
    /// The GL context must be current on this thread, and `gl` must be its
    /// loaded entry points.
    pub unsafe fn self_test(&self, gl: &glow::Context) -> DmabufImportProbeResult {
        let (Some((export_query, export)), Some(current_context)) =
            (self.export, self.current_context)
        else {
            return DmabufImportProbeResult::Untested(
                "EGL_MESA_image_dma_buf_export is not available, so no dma-buf could be made to \
                 test the import with"
                    .into(),
            );
        };
        // SAFETY: delegated to this function's own contract; every GL object
        // created below is deleted before returning.
        unsafe {
            match self.round_trip(gl, export_query, export, current_context) {
                Ok(()) => DmabufImportProbeResult::Passed,
                Err(e) => DmabufImportProbeResult::Failed(e),
            }
        }
    }

    /// One export-and-import round trip, tearing down what it created.
    ///
    /// # Safety
    /// As [`Self::self_test`].
    unsafe fn round_trip(
        &self,
        gl: &glow::Context,
        export_query: PfnExportDmabufQueryMesa,
        export: PfnExportDmabufMesa,
        current_context: PfnGetCurrentContext,
    ) -> Result<(), String> {
        // A pattern with no repeats: every pixel differs in every channel, so
        // a transposed, sheared or offset readback cannot compare equal.
        let source: Vec<u8> = (0..SELF_TEST_SIDE * SELF_TEST_SIDE)
            .flat_map(|i| {
                let i = i.rem_euclid(256).unsigned_abs().to_le_bytes()[0];
                [
                    i.wrapping_mul(17),
                    i.wrapping_mul(29),
                    i.wrapping_mul(53),
                    0xff,
                ]
            })
            .collect();

        // SAFETY: the context is current, per this function's contract. The
        // texture is deleted on every path out.
        unsafe {
            let texture = gl
                .create_texture()
                .map_err(|e| format!("create_texture: {e}"))?;
            let result =
                self.round_trip_from(gl, texture, &source, export_query, export, current_context);
            gl.delete_texture(texture);
            result
        }
    }

    /// The round trip proper, given a texture to fill, export and re-import.
    ///
    /// # Safety
    /// As [`Self::self_test`], and `texture` must be a live texture name.
    unsafe fn round_trip_from(
        &self,
        gl: &glow::Context,
        texture: glow::Texture,
        source: &[u8],
        export_query: PfnExportDmabufQueryMesa,
        export: PfnExportDmabufMesa,
        current_context: PfnGetCurrentContext,
    ) -> Result<(), String> {
        // SAFETY: delegated to this function's own contract.
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST.cast_signed(),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST.cast_signed(),
            );
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8.cast_signed(),
                SELF_TEST_SIDE,
                SELF_TEST_SIDE,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(source)),
            );
            // The export hands over memory, not a promise of pending work.
            gl.finish();

            let described = self.export_texture(texture, export_query, export, current_context)?;
            let reimported = self
                .import(&described)
                .map_err(|e| format!("re-importing what was just exported failed: {e}"))?;
            self.read_back(gl, &reimported, source)
        }
    }

    /// Export a GL texture as a dma-buf, described the way a client would.
    ///
    /// # Safety
    /// As [`Self::self_test`], and `texture` must be a live texture name whose
    /// contents have been flushed.
    unsafe fn export_texture(
        &self,
        texture: glow::Texture,
        export_query: PfnExportDmabufQueryMesa,
        export: PfnExportDmabufMesa,
        current_context: PfnGetCurrentContext,
    ) -> Result<DmabufImage, String> {
        // SAFETY: delegated to this function's own contract. The intermediate
        // image is destroyed by its own `Drop` on every path out, and the
        // exported descriptor is owned from the moment it is handed over.
        unsafe {
            let preserve = [EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE];
            let raw = (self.create_image)(
                self.display,
                current_context(),
                EGL_GL_TEXTURE_2D_KHR,
                std::ptr::without_provenance(texture.0.get() as usize),
                preserve.as_ptr(),
            );
            if raw == EGL_NO_IMAGE_KHR {
                return Err("could not make an EGLImage from a GL texture".into());
            }
            let exported = EglImage {
                raw,
                display: self.display,
                destroy: self.destroy_image,
            };

            let (mut fourcc, mut planes, mut modifier) = (0 as EGLint, 0 as EGLint, 0u64);
            if export_query(
                self.display,
                exported.raw,
                &raw mut fourcc,
                &raw mut planes,
                &raw mut modifier,
            ) != 1
            {
                return Err("eglExportDMABUFImageQueryMESA failed".into());
            }
            if planes != 1 {
                return Err(format!("exported image has {planes} planes, expected 1"));
            }

            let (mut fd, mut stride, mut offset) = (-1 as EGLint, 0 as EGLint, 0 as EGLint);
            if export(
                self.display,
                exported.raw,
                &raw mut fd,
                &raw mut stride,
                &raw mut offset,
            ) != 1
                || fd < 0
            {
                return Err("eglExportDMABUFImageMESA failed".into());
            }

            debug!(
                "dma-buf self-test: exported {SELF_TEST_SIDE}x{SELF_TEST_SIDE} {} modifier {modifier:#x} stride {stride}",
                fourcc_name(fourcc.cast_unsigned()),
            );

            Ok(DmabufImage {
                width: SELF_TEST_SIDE,
                height: SELF_TEST_SIDE,
                fourcc: fourcc.cast_unsigned(),
                modifier,
                planes: vec![DmabufPlane {
                    // Owned from here on, so it is closed when the image is
                    // dropped however this ends.
                    fd: Arc::new(OwnedFd::from_raw_fd(fd)),
                    offset: offset.unsigned_abs(),
                    stride: stride.unsigned_abs(),
                }],
            })
        }
    }

    /// Bind an imported image to a texture, read it back, and compare.
    ///
    /// # Safety
    /// As [`Self::self_test`].
    unsafe fn read_back(
        &self,
        gl: &glow::Context,
        image: &EglImage,
        expected: &[u8],
    ) -> Result<(), String> {
        // SAFETY: the context is current, and both objects are deleted before
        // returning on every path.
        unsafe {
            let texture = gl
                .create_texture()
                .map_err(|e| format!("create_texture: {e}"))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST.cast_signed(),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST.cast_signed(),
            );
            self.bind_to_texture(image);

            // Reading a texture back means rendering from it: GLES has no
            // glGetTexImage, so it is attached to a framebuffer and read there.
            let framebuffer = match gl.create_framebuffer() {
                Ok(fb) => fb,
                Err(e) => {
                    gl.delete_texture(texture);
                    return Err(format!("create_framebuffer: {e}"));
                }
            };
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );

            let mut readback = vec![0u8; expected.len()];
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            let result = if status == glow::FRAMEBUFFER_COMPLETE {
                gl.read_pixels(
                    0,
                    0,
                    SELF_TEST_SIDE,
                    SELF_TEST_SIDE,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelPackData::Slice(Some(&mut readback)),
                );
                if readback == expected {
                    Ok(())
                } else {
                    Err(format!(
                        "imported pixels differ from what was exported: {:02x?} vs {:02x?}",
                        &readback[..8.min(readback.len())],
                        &expected[..8.min(expected.len())],
                    ))
                }
            } else {
                Err(format!(
                    "cannot read back the imported image: framebuffer status {status:#x}"
                ))
            };

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.delete_framebuffer(framebuffer);
            gl.delete_texture(texture);
            result
        }
    }
}

/// Ask the driver for one format's modifiers, each paired with whether it is
/// `external_only`.
///
/// An empty answer covers both ways of saying nothing — a driver that will not
/// enumerate, and one that enumerates none — because they mean the same thing:
/// the implicit layout is all there is. See [`advertisable_format`], which is
/// where that is acted on.
///
/// # Safety
/// `query` must be `eglQueryDmaBufModifiersEXT` for `display`.
unsafe fn enumerate_modifiers(
    query: PfnQueryDmabufModifiers,
    display: EGLDisplay,
    fourcc: EGLint,
) -> Vec<(u64, bool)> {
    // SAFETY: delegated to this function's own contract. Each call passes a
    // count matching the buffers handed over: first none, to learn the count,
    // then exactly that many.
    unsafe {
        let mut count: EGLint = 0;
        if query(
            display,
            fourcc,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &raw mut count,
        ) != 1
            || count <= 0
        {
            return Vec::new();
        }

        let len = count.unsigned_abs() as usize;
        let mut modifiers = vec![0u64; len];
        let mut external = vec![0 as EGLBoolean; len];
        if query(
            display,
            fourcc,
            count,
            modifiers.as_mut_ptr(),
            external.as_mut_ptr(),
            &raw mut count,
        ) != 1
        {
            return Vec::new();
        }
        modifiers.truncate(count.unsigned_abs() as usize);

        modifiers
            .into_iter()
            .zip(external)
            .map(|(modifier, external_only)| (modifier, external_only != 0))
            .collect()
    }
}

/// Whether a format can be offered to clients, and with which modifiers.
///
/// This is the rule that keeps the compositor from advertising what it cannot
/// draw, which matters more than it sounds: a client allocates against the list
/// it is given, and a buffer it allocated for a format we then refuse leaves it
/// with nothing to fall back to.
///
/// Two things get filtered. A modifier marked `external_only` needs
/// `samplerExternalOES`, which this renderer has no program for. And a format
/// whose every modifier was external is dropped outright — on Mesa that is
/// every YUV layout, which is why the advertised list is much shorter than the
/// driver's.
///
/// An empty `enumerated` is the opposite case and must not be confused with it:
/// the driver named no modifiers, so the implicit layout is all there is, and
/// that imports fine. "Nothing named" and "nothing usable" look alike and mean
/// opposite things.
pub(crate) fn advertisable_format(fourcc: u32, enumerated: &[(u64, bool)]) -> Option<DmabufFormat> {
    if enumerated.is_empty() {
        return Some(DmabufFormat {
            fourcc,
            modifiers: Vec::new(),
        });
    }

    let modifiers: Vec<u64> = enumerated
        .iter()
        .filter(|&&(_, external_only)| !external_only)
        .map(|&(modifier, _)| modifier)
        .collect();
    if modifiers.is_empty() {
        return None;
    }
    Some(DmabufFormat { fourcc, modifiers })
}

/// Ask EGL which DRM device the display sits on, as a device file path.
///
/// Three extensions deep, each checked before its entry points are trusted:
/// `EGL_EXT_device_query` (a *client* extension, so it is looked for in the
/// no-display extension string) turns the display into a device handle, and
/// `EGL_EXT_device_drm_render_node` or `EGL_EXT_device_drm` (device
/// extensions, on the device's own string) turn the handle into a path. The
/// render node is preferred — see [`RenderNode`]. Any absence is `None`:
/// the handshake degrades to "device unknown", never to an error.
///
/// # Safety
/// `display` must be live, `loader` must resolve EGL symbols for it, and
/// `query_string` must be `eglQueryString`.
unsafe fn query_render_node(
    display: EGLDisplay,
    loader: &dyn Fn(&CStr) -> *const c_void,
    query_string: PfnQueryString,
) -> Option<RenderNode> {
    // SAFETY: delegated to this function's own contract; each entry point is
    // loaded only after the extension that defines it is confirmed.
    unsafe {
        let raw = query_string(std::ptr::null(), EGL_EXTENSIONS);
        if raw.is_null() {
            return None;
        }
        let client_extensions = CStr::from_ptr(raw).to_string_lossy().into_owned();
        if !has_extension(&client_extensions, "EGL_EXT_device_query") {
            return None;
        }
        let query_display_attrib: PfnQueryDisplayAttribExt =
            load(loader, c"eglQueryDisplayAttribEXT")?;
        let query_device_string: PfnQueryDeviceStringExt =
            load(loader, c"eglQueryDeviceStringEXT")?;

        let mut device: EGLAttrib = 0;
        if query_display_attrib(display, EGL_DEVICE_EXT, &raw mut device) != 1 || device == 0 {
            return None;
        }
        let device: EGLDeviceEXT = std::ptr::without_provenance(device.cast_unsigned());

        let raw = query_device_string(device, EGL_EXTENSIONS);
        if raw.is_null() {
            return None;
        }
        let device_extensions = CStr::from_ptr(raw).to_string_lossy().into_owned();

        // The render node first: it can be opened without DRM master. The
        // primary node is still a truthful answer when it is all there is.
        for (extension, attribute) in [
            ("EGL_EXT_device_drm_render_node", EGL_DRM_RENDER_NODE_FILE_EXT),
            ("EGL_EXT_device_drm", EGL_DRM_DEVICE_FILE_EXT),
        ] {
            if !has_extension(&device_extensions, extension) {
                continue;
            }
            let raw = query_device_string(device, attribute);
            if raw.is_null() {
                continue;
            }
            let path = CStr::from_ptr(raw).to_string_lossy().into_owned();
            if !path.is_empty() {
                return Some(RenderNode { path: path.into() });
            }
        }
        None
    }
}

/// Resolve one entry point, or `None` if the driver does not have it.
///
/// # Safety
/// `T` must be the function pointer type the symbol actually has. Getting that
/// wrong is a call through a mistyped pointer, which is why every use sits next
/// to the extension spec the signature came from.
unsafe fn load<T: Copy>(loader: &dyn Fn(&CStr) -> *const c_void, symbol: &CStr) -> Option<T> {
    const {
        assert!(
            size_of::<T>() == size_of::<*const c_void>(),
            "entry points are function pointers"
        );
    }
    let address = loader(symbol);
    if address.is_null() {
        return None;
    }
    // SAFETY: the size is checked above and the signature is the caller's
    // promise; `address` is non-null and stays valid for the life of the
    // display it was loaded from.
    Some(unsafe { *std::ptr::from_ref(&address).cast::<T>() })
}

/// Whether an extension is in a space-separated EGL extension string.
///
/// Substring matching would say yes to `EGL_EXT_image_dma_buf_import` for a
/// driver offering only `EGL_EXT_image_dma_buf_import_modifiers`, which is a
/// different extension with different entry points.
pub(crate) fn has_extension(extensions: &str, wanted: &str) -> bool {
    extensions.split_whitespace().any(|e| e == wanted)
}

#[cfg(test)]
mod tests {
    //! Tests for the parts of the import path that need no GPU.
    //!
    //! The import itself is a driver call and is checked at runtime by
    //! [`DmabufImporter::self_test`] instead; what is unit-testable is the
    //! reasoning around it — which is also where a mistake is silent rather
    //! than loud.

    use super::{advertisable_format, has_extension};
    use crate::dma::{
        DRM_FORMAT_ARGB8888, DRM_FORMAT_MOD_INVALID, DRM_FORMAT_XRGB8888, fourcc, fourcc_name,
    };

    /// A linear layout, which any driver can sample.
    const LINEAR: u64 = 0;
    /// A tiled layout, standing in for anything a driver might name.
    const TILED: u64 = 0x0100_0000_0000_0001;

    #[test]
    fn an_extension_is_matched_whole() {
        let extensions =
            "EGL_KHR_image_base EGL_EXT_image_dma_buf_import_modifiers EGL_EXT_buffer_age";

        assert!(has_extension(extensions, "EGL_KHR_image_base"));
        assert!(has_extension(
            extensions,
            "EGL_EXT_image_dma_buf_import_modifiers"
        ));
        // The base import extension is a different one with different entry
        // points, and a substring match would claim the driver has it.
        assert!(!has_extension(extensions, "EGL_EXT_image_dma_buf_import"));
        assert!(!has_extension(extensions, "EGL_KHR_image"));
        assert!(!has_extension("", "EGL_EXT_image_dma_buf_import"));
    }

    #[test]
    fn fourccs_are_the_four_characters_they_are_named_for() {
        assert_eq!(fourcc_name(DRM_FORMAT_ARGB8888), "AR24");
        assert_eq!(fourcc_name(DRM_FORMAT_XRGB8888), "XR24");
        // Whatever a driver hands back, the log line stays readable.
        assert_eq!(fourcc_name(0), "????");
    }

    #[test]
    fn a_driver_that_names_no_modifiers_leaves_the_format_importable() {
        // Nothing named means the implicit layout is all there is, and that
        // imports. Dropping the format here would refuse buffers that work.
        let format = advertisable_format(DRM_FORMAT_ARGB8888, &[]).expect("should be advertised");

        assert_eq!(format.fourcc, DRM_FORMAT_ARGB8888);
        assert!(
            format.modifiers.is_empty(),
            "an empty list is what says {DRM_FORMAT_MOD_INVALID:#x} is the only option"
        );
    }

    #[test]
    fn modifiers_needing_an_external_sampler_are_not_offered() {
        let format = advertisable_format(DRM_FORMAT_XRGB8888, &[(LINEAR, false), (TILED, true)])
            .expect("should be advertised");

        // The tiled one could only be sampled through `samplerExternalOES`,
        // which this renderer has no program for, so a client must not be told
        // to allocate with it.
        assert_eq!(format.modifiers, [LINEAR]);
    }

    #[test]
    fn a_format_no_modifier_of_which_can_be_sampled_is_dropped() {
        // Every YUV format Mesa reports looks like this. Offering one would
        // have a client allocate NV12 against our list and then find nothing
        // drawn — and by then it has no fallback left.
        assert!(advertisable_format(fourcc(*b"NV12"), &[(LINEAR, true), (TILED, true)]).is_none());
    }

    #[test]
    fn naming_nothing_and_naming_nothing_usable_are_opposite_answers() {
        // The two look alike — both end with no usable modifier named — and
        // mean opposite things. Collapsing them is the mistake this whole rule
        // exists to avoid, in one direction or the other.
        let unnamed = advertisable_format(fourcc(*b"AR24"), &[]);
        let unusable = advertisable_format(fourcc(*b"NV12"), &[(TILED, true)]);

        assert!(unnamed.is_some_and(|f| f.modifiers.is_empty()));
        assert!(unusable.is_none());
    }
}
