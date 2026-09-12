//! One DRM device: its outputs, and the page-flip loop that paces them.
//!
//! This is the hardware twin of the winit backend's window. Where winit hands
//! its frames to a host compositor, here the frames go to a GBM surface, get
//! wrapped in a KMS framebuffer, and are put on screen by a page flip — and
//! the flip completing is the vblank that paces the next frame, exactly the
//! signal `RedrawRequested` stands in for on a host.
//!
//! Legacy modesetting, not atomic: `set_crtc` for the first frame of an
//! output and `page_flip` for every one after. Atomic's per-property commit
//! is what overlay planes and tear-free reconfiguration need, and is the
//! obvious next step; the legacy path is smaller, universally supported, and
//! enough to drive a plain scanout output correctly.
//!
//! One EGL display and context and one [`GlRenderer`] serve every output on
//! the device; each output keeps its own GBM surface, EGL window surface, and
//! the one-or-two framebuffers a double-buffered flip has in flight.

use std::ffi::{CStr, c_void};
use std::os::fd::{AsFd, AsRawFd, RawFd};
use std::ptr;

use drm::control::{Device as ControlDevice, Event, Mode, ModeTypeFlags, PageFlipFlags, connector,
    crtc, framebuffer};
use gbm::{AsRaw, BufferObject, BufferObjectFlags, Device as GbmDevice, Format, Surface as GbmSurface};
use khronos_egl as egl;
use libseat::Device as SeatDevice;
use tracing::{info, warn};

use crate::dmabuf_import::DmabufImporter;
use crate::gl_renderer::GlRenderer;
use crate::messages::PresentationFlags;
use crate::monotonic_timestamp::MonotonicTimeStamp;
use crate::outputs::{
    OUTPUT_MODE_CURRENT, OUTPUT_MODE_PREFERRED, Output, OutputGeometry, OutputId, OutputMode,
    OutputSubpixel, OutputTransform, Scale,
};
use crate::scene_graph::{Scene, SceneElement, SceneGraph};

/// `EGL_PLATFORM_GBM_KHR`, the platform enum for a display backed by a GBM
/// device. Not in khronos-egl's constants, so spelled out from the spec.
const PLATFORM_GBM_KHR: egl::Enum = 0x31D7;

/// A seat-opened DRM node, owning its fd through the libseat token.
///
/// The newtype is what lets gbm and drm-rs treat the seat's device as their
/// own: both build on [`AsFd`], and delegating it here means one object is at
/// once the GBM allocator and the KMS control device.
struct Card {
    device: SeatDevice,
}

impl AsFd for Card {
    fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        self.device.as_fd()
    }
}

// drm-rs's traits are all-default methods over `AsFd`; the empty impls opt
// the type in, and gbm then forwards them to `Device<Card>`.
impl drm::Device for Card {}
impl ControlDevice for Card {}

/// A framebuffer put on screen, and the buffer it wraps.
///
/// The two travel together because they die together: the KMS framebuffer
/// must be destroyed and the GBM buffer released only once scanout has moved
/// off them, which is a flip later.
struct Framebuffer {
    /// The locked GBM front buffer.
    bo: BufferObject<()>,
    /// The KMS framebuffer wrapping it.
    fb: framebuffer::Handle,
}

/// One output: a connector lit through a CRTC, with its own render surface
/// and the buffers a double-buffered flip keeps in flight.
struct OutputScanout {
    /// The id reported to the compositor.
    id: OutputId,
    /// The connector this drives.
    connector: connector::Handle,
    /// The CRTC feeding it.
    crtc: crtc::Handle,
    /// The mode it is set to.
    mode: Mode,
    /// The GBM surface EGL renders into.
    gbm_surface: GbmSurface<()>,
    /// The EGL window surface over that GBM surface.
    egl_surface: egl::Surface,
    /// What is on screen now, held until the next flip moves off it.
    front: Option<Framebuffer>,
    /// What a queued flip will show, promoted to `front` when it completes.
    pending: Option<Framebuffer>,
    /// Whether the first frame has modeset this output. Until it has, the
    /// output is driven by `set_crtc`; after, by `page_flip`.
    modeset_done: bool,
    /// Whether a flip is queued and its event not yet seen. While true the
    /// output must not render again — that is the one-frame-in-flight bound.
    awaiting_flip: bool,
    /// Frames presented, reported as the presentation sequence.
    sequence: u64,
    /// The layout position, in logical pixels.
    x: i32,
}

/// One DRM device and everything drawn through it.
pub struct Scanout {
    /// The GBM device, which is also the KMS control device.
    gbm: GbmDevice<Card>,
    /// The EGL entry points (statically linked).
    egl: egl::Instance<egl::Static>,
    /// The EGL display on the GBM device.
    display: egl::Display,
    /// The one context every output's surface is drawn with.
    context: egl::Context,
    /// The shared rasteriser.
    renderer: GlRenderer,
    /// The outputs, in the order they were enumerated.
    outputs: Vec<OutputScanout>,
    /// The scale every output is reported and rendered at. Compositor policy
    /// on real hardware; 1× until there is somewhere to configure it.
    scale: Scale,
}

/// The outcome of rendering an output, which decides what the loop reports.
pub enum Presented {
    /// The frame is already on screen — the initial modeset is synchronous —
    /// so the presentation can be reported at once.
    Immediately {
        /// Which output.
        output: OutputId,
        /// Its refresh period in nanoseconds.
        refresh_ns: u32,
        /// The presentation sequence.
        sequence: u64,
    },
    /// A flip is queued; the presentation is reported when its event arrives
    /// through [`Scanout::handle_events`].
    FlipQueued,
    /// Nothing was drawn: no such output, or one still awaiting its flip.
    Skipped,
}

/// A presentation a completed flip produced.
pub struct FlipDone {
    /// Which output flipped.
    pub output: OutputId,
    /// Its refresh period in nanoseconds.
    pub refresh_ns: u32,
    /// The presentation sequence.
    pub sequence: u64,
}

impl Scanout {
    /// Open the device, set up EGL and the renderer, and enumerate the
    /// connected outputs. The DRM node arrives already opened by the seat.
    ///
    /// # Errors
    /// If EGL will not initialise on the device, the renderer cannot be
    /// built, or no output can be brought up.
    pub fn open(device: SeatDevice) -> anyhow::Result<Self> {
        let card = Card { device };
        let gbm = GbmDevice::new(card)
            .map_err(|e| anyhow::anyhow!("could not create a GBM device: {e}"))?;

        let egl = egl::Instance::new(egl::Static);
        // SAFETY: the GBM device pointer is live for the display's life — the
        // GBM device outlives this Scanout, which owns both.
        let display = unsafe {
            egl.get_platform_display(
                PLATFORM_GBM_KHR,
                gbm.as_raw() as *mut c_void,
                &[egl::ATTRIB_NONE],
            )
        }
        .map_err(|e| anyhow::anyhow!("no EGL display on this GBM device: {e}"))?;
        egl.initialize(display)
            .map_err(|e| anyhow::anyhow!("could not initialise EGL: {e}"))?;
        egl.bind_api(egl::OPENGL_ES_API)
            .map_err(|e| anyhow::anyhow!("could not bind the GLES API: {e}"))?;

        let config = egl
            .choose_first_config(
                display,
                &[
                    egl::SURFACE_TYPE,
                    egl::WINDOW_BIT,
                    egl::RED_SIZE,
                    8,
                    egl::GREEN_SIZE,
                    8,
                    egl::BLUE_SIZE,
                    8,
                    egl::ALPHA_SIZE,
                    0,
                    egl::RENDERABLE_TYPE,
                    egl::OPENGL_ES3_BIT,
                    egl::NONE,
                ],
            )
            .map_err(|e| anyhow::anyhow!("no matching EGL config: {e}"))?
            .ok_or_else(|| anyhow::anyhow!("the driver offered no EGL config for scanout"))?;

        let context = egl
            .create_context(
                display,
                config,
                None,
                &[egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE],
            )
            .map_err(|e| anyhow::anyhow!("could not create a GLES context: {e}"))?;

        // The renderer and importer both load through EGL. One closure serves
        // both: a `&Fn` is an `FnMut`, so the same reference goes to each.
        let loader = |name: &CStr| -> *const c_void {
            name.to_str()
                .ok()
                .and_then(|n| egl.get_proc_address(n))
                .map_or(ptr::null(), |f| f as *const c_void)
        };

        // A context must be current before the renderer touches GL. There is
        // no surface yet, so bind the context with none — GLES allows a
        // surfaceless make-current for setup on any modern Mesa driver.
        egl.make_current(display, None, None, Some(context))
            .map_err(|e| anyhow::anyhow!("could not make the GLES context current: {e}"))?;

        // SAFETY: the display is the one the context was made on, current on
        // this thread for the renderer's life.
        let importer = match unsafe {
            DmabufImporter::new(display.as_ptr().cast::<c_void>(), &loader)
        } {
            Ok(importer) => Some(importer),
            Err(reason) => {
                info!("dma-buf import unavailable on this device: {reason}");
                None
            }
        };
        // SAFETY: the context is current on this thread and stays so for the
        // renderer's life.
        let renderer = unsafe { GlRenderer::new(&loader, importer)? };

        let mut scanout = Self {
            gbm,
            egl,
            display,
            context,
            renderer,
            outputs: Vec::new(),
            scale: Scale::ONE,
        };
        scanout.enumerate_outputs(config)?;
        if scanout.outputs.is_empty() {
            anyhow::bail!("no connected outputs on this device");
        }
        Ok(scanout)
    }

    /// Find every connected connector, pick a mode and a free CRTC for each,
    /// and build its render surfaces. Laid out left to right.
    fn enumerate_outputs(&mut self, config: egl::Config) -> anyhow::Result<()> {
        let resources = self
            .gbm
            .resource_handles()
            .map_err(|e| anyhow::anyhow!("could not read DRM resources: {e}"))?;

        let mut used_crtcs: Vec<crtc::Handle> = Vec::new();
        let mut next_x = 0;
        let mut next_id = 1u32;
        for &connector_handle in resources.connectors() {
            let Ok(info) = self.gbm.get_connector(connector_handle, false) else {
                continue;
            };
            if info.state() != connector::State::Connected {
                continue;
            }
            let Some(mode) = pick_mode(info.modes()) else {
                warn!("connected connector has no modes; skipping");
                continue;
            };
            let Some(crtc) = pick_crtc(&self.gbm, &resources, &info, &used_crtcs) else {
                warn!("no free CRTC for a connected connector; skipping");
                continue;
            };
            used_crtcs.push(crtc);

            let (width, height) = mode.size();
            let gbm_surface = self
                .gbm
                .create_surface::<()>(
                    u32::from(width),
                    u32::from(height),
                    Format::Xrgb8888,
                    BufferObjectFlags::SCANOUT | BufferObjectFlags::RENDERING,
                )
                .map_err(|e| anyhow::anyhow!("could not create a GBM surface: {e}"))?;
            // SAFETY: the GBM surface outlives the EGL surface — both live in
            // the OutputScanout below, surface declared to drop after.
            let egl_surface = unsafe {
                self.egl.create_window_surface(
                    self.display,
                    config,
                    gbm_surface.as_raw() as *mut c_void,
                    None,
                )
            }
            .map_err(|e| anyhow::anyhow!("could not create an EGL surface: {e}"))?;

            let id = OutputId(next_id);
            next_id += 1;
            info!(
                "output {} ({}-{}) at {}x{}",
                id.0,
                info.interface().as_str(),
                info.interface_id(),
                width,
                height,
            );
            self.outputs.push(OutputScanout {
                id,
                connector: connector_handle,
                crtc,
                mode,
                gbm_surface,
                egl_surface,
                front: None,
                pending: None,
                modeset_done: false,
                awaiting_flip: false,
                sequence: 0,
                x: next_x,
            });
            let (logical_w, _) = logical_size(mode, self.scale);
            next_x += logical_w;
        }
        Ok(())
    }

    /// How the compositor and its clients should see these outputs.
    #[must_use]
    pub fn output_descriptions(&self) -> Vec<Output> {
        self.outputs
            .iter()
            .map(|output| describe(output, self.scale))
            .collect()
    }

    /// The DRM fd to wait on for page-flip events.
    #[must_use]
    pub fn drm_fd(&self) -> RawFd {
        self.gbm.as_fd().as_raw_fd()
    }

    /// The nominal refresh of an output in nanoseconds, for the first frame
    /// request the loop sends before any flip has happened.
    #[must_use]
    pub fn refresh_ns(&self, output: OutputId) -> u32 {
        self.outputs
            .iter()
            .find(|o| o.id == output)
            .map_or(0, |o| refresh_ns_of(o.mode))
    }

    /// Draw a scene for one output and put it on screen — a synchronous
    /// modeset for the first frame, a queued page flip after.
    pub fn render_output(
        &mut self,
        output_id: OutputId,
        scene: &Scene,
        cursor: &[SceneElement],
    ) -> Presented {
        let Some(index) = self.outputs.iter().position(|o| o.id == output_id) else {
            return Presented::Skipped;
        };
        // Rendering while a flip is pending would lock a third buffer and race
        // the one on screen; the pacing forbids it, and this enforces it.
        if self.outputs[index].awaiting_flip {
            return Presented::Skipped;
        }

        let (mode, egl_surface) = {
            let o = &self.outputs[index];
            (o.mode, o.egl_surface)
        };
        let (width, height) = mode.size();

        // SAFETY: the surface and context belong to this display and thread.
        if let Err(e) = self.egl.make_current(
            self.display,
            Some(egl_surface),
            Some(egl_surface),
            Some(self.context),
        ) {
            warn!("make_current failed for output {}: {e}", output_id.0);
            return Presented::Skipped;
        }
        self.renderer
            .draw(scene, cursor, u32::from(width), u32::from(height));
        if let Err(e) = self.egl.swap_buffers(self.display, egl_surface) {
            warn!("swap_buffers failed for output {}: {e}", output_id.0);
            return Presented::Skipped;
        }

        let Some(framebuffer) = self.lock_framebuffer(index) else {
            return Presented::Skipped;
        };

        let output = &mut self.outputs[index];
        if output.modeset_done {
            match self.gbm.page_flip(
                output.crtc,
                framebuffer.fb,
                PageFlipFlags::EVENT,
                None,
            ) {
                Ok(()) => {
                    output.pending = Some(framebuffer);
                    output.awaiting_flip = true;
                    Presented::FlipQueued
                }
                Err(e) => {
                    warn!("page flip failed for output {}: {e}", output_id.0);
                    self.destroy_framebuffer(framebuffer);
                    Presented::Skipped
                }
            }
        } else {
            match self.gbm.set_crtc(
                output.crtc,
                Some(framebuffer.fb),
                (0, 0),
                &[output.connector],
                Some(output.mode),
            ) {
                Ok(()) => {
                    // Synchronous: it is on screen now. The old front, if any,
                    // is free.
                    if let Some(old) = output.front.take() {
                        self.destroy_framebuffer(old);
                    }
                    let output = &mut self.outputs[index];
                    output.front = Some(framebuffer);
                    output.modeset_done = true;
                    output.sequence += 1;
                    Presented::Immediately {
                        output: output_id,
                        refresh_ns: refresh_ns_of(mode),
                        sequence: output.sequence,
                    }
                }
                Err(e) => {
                    warn!("modeset failed for output {}: {e}", output_id.0);
                    self.destroy_framebuffer(framebuffer);
                    Presented::Skipped
                }
            }
        }
    }

    /// Lock the surface's front buffer and wrap it in a KMS framebuffer.
    fn lock_framebuffer(&mut self, index: usize) -> Option<Framebuffer> {
        // SAFETY: called right after swap_buffers, so a front buffer exists;
        // the returned bo is released when the Framebuffer is destroyed.
        let bo = match unsafe { self.outputs[index].gbm_surface.lock_front_buffer() } {
            Ok(bo) => bo,
            Err(e) => {
                warn!("could not lock the front buffer: {e}");
                return None;
            }
        };
        match self.gbm.add_framebuffer(&bo, 24, 32) {
            Ok(fb) => Some(Framebuffer { bo, fb }),
            Err(e) => {
                warn!("could not create a KMS framebuffer: {e}");
                None
            }
        }
    }

    /// Destroy a framebuffer and release its buffer.
    fn destroy_framebuffer(&self, framebuffer: Framebuffer) {
        if let Err(e) = self.gbm.destroy_framebuffer(framebuffer.fb) {
            warn!("destroying a framebuffer failed: {e}");
        }
        drop(framebuffer.bo);
    }

    /// Drain the page-flip events the DRM fd has ready, advancing each output
    /// whose flip completed and returning what to report as presented.
    ///
    /// # Errors
    /// If reading the DRM events fails.
    pub fn handle_events(&mut self) -> anyhow::Result<Vec<FlipDone>> {
        // Collect the CRTCs first: iterating the events borrows the device,
        // and advancing the outputs borrows it again to destroy framebuffers.
        let flipped: Vec<crtc::Handle> = self
            .gbm
            .receive_events()
            .map_err(|e| anyhow::anyhow!("reading DRM events failed: {e}"))?
            .filter_map(|event| match event {
                Event::PageFlip(flip) => Some(flip.crtc),
                _ => None,
            })
            .collect();

        let mut done = Vec::new();
        for crtc in flipped {
            let Some(index) = self.outputs.iter().position(|o| o.crtc == crtc) else {
                continue;
            };
            // The buffer that was on screen is now free; what was pending is
            // now on screen.
            let old_front = self.outputs[index].front.take();
            let pending = self.outputs[index].pending.take();
            if let Some(old) = old_front {
                self.destroy_framebuffer(old);
            }
            let output = &mut self.outputs[index];
            output.front = pending;
            output.awaiting_flip = false;
            output.sequence += 1;
            done.push(FlipDone {
                output: output.id,
                refresh_ns: refresh_ns_of(output.mode),
                sequence: output.sequence,
            });
        }
        Ok(done)
    }

    /// Drop the textures and programs the frame no longer references — the
    /// same cache trim the winit backend does each frame.
    pub fn prune_caches(&mut self, frame: &SceneGraph) {
        self.renderer.prune_caches(frame);
    }

    /// Effect-compile failures since the last call, to forward to the
    /// compositor.
    pub fn take_effect_failures(&mut self) -> Vec<(String, String)> {
        self.renderer.take_effect_failures()
    }

    /// The ids of the outputs this device drives.
    #[must_use]
    pub fn output_ids(&self) -> Vec<OutputId> {
        self.outputs.iter().map(|o| o.id).collect()
    }

    /// One output's physical size in pixels — the framebuffer size, and what
    /// touch coordinates resolve against.
    #[must_use]
    pub fn output_physical_size(&self, output: OutputId) -> Option<(i32, i32)> {
        self.outputs.iter().find(|o| o.id == output).map(|o| {
            let (w, h) = o.mode.size();
            (i32::from(w), i32::from(h))
        })
    }

    /// Whether an output is mid-flip and so must not be rendered again yet.
    #[must_use]
    pub fn awaiting_flip(&self, output: OutputId) -> bool {
        self.outputs
            .iter()
            .find(|o| o.id == output)
            .is_some_and(|o| o.awaiting_flip)
    }

    /// What this device can do with dma-bufs, for the compositor's probe.
    #[must_use]
    pub fn dmabuf_support(&self) -> crate::dmabuf_import::DmabufCapabilities {
        self.renderer.dmabuf_support()
    }

    /// Whether this device's driver will take a client's buffer.
    #[must_use]
    pub fn verify_import(&self, image: &crate::dma::DmabufImage) -> bool {
        self.renderer.verify_import(image)
    }

    /// Capture what an output is showing, by re-rendering its scene offscreen
    /// and reading it back. `None` if there is no such output or the readback
    /// fails.
    pub fn capture(
        &mut self,
        output: OutputId,
        scene: &Scene,
        cursor: &[SceneElement],
    ) -> Option<crate::messages::CapturedFrame> {
        let index = self.outputs.iter().position(|o| o.id == output)?;
        let (mode, egl_surface) = {
            let o = &self.outputs[index];
            (o.mode, o.egl_surface)
        };
        // A current context is all `render_to_cpu` needs — it draws into its
        // own framebuffer — but there must be one, so bind the output's.
        self.egl
            .make_current(
                self.display,
                Some(egl_surface),
                Some(egl_surface),
                Some(self.context),
            )
            .ok()?;
        let (width, height) = mode.size();
        let pixels = self
            .renderer
            .render_to_cpu(scene, cursor, u32::from(width), u32::from(height))?;
        Some(crate::messages::CapturedFrame {
            width: u32::from(width),
            height: u32::from(height),
            pixels,
        })
    }

    /// Forget the modeset and in-flight buffers, so the next render of each
    /// output modesets afresh. Used when the session comes back from a VT
    /// switch, where the kernel has torn down the CRTC configuration.
    pub fn mark_needs_modeset(&mut self) {
        // Drop the framebuffers first — nothing is scanning them out now.
        let stale: Vec<Framebuffer> = self
            .outputs
            .iter_mut()
            .flat_map(|o| {
                o.modeset_done = false;
                o.awaiting_flip = false;
                [o.front.take(), o.pending.take()]
            })
            .flatten()
            .collect();
        for framebuffer in stale {
            self.destroy_framebuffer(framebuffer);
        }
    }
}

/// A default presentation flags value for a DRM scanout: a real page flip
/// vsyncs, uses a hardware clock, signals completion, and — since the client
/// buffer went through the compositor's own GL composite — is not zero-copy.
#[must_use]
pub fn scanout_flags() -> PresentationFlags {
    PresentationFlags {
        vsync: true,
        hw_clock: true,
        hw_completion: true,
        zero_copy: false,
    }
}

/// The monotonic clock reading for a presentation. The kernel reports the
/// exact flip time in the event; reading the clock on receipt is close
/// enough for a first implementation and avoids threading the event's
/// timestamp through. Documented imprecision.
#[must_use]
pub fn presentation_time() -> MonotonicTimeStamp {
    MonotonicTimeStamp::now()
}

/// Prefer the mode the display marks preferred, else the first listed.
fn pick_mode(modes: &[Mode]) -> Option<Mode> {
    modes
        .iter()
        .find(|m| m.mode_type().contains(ModeTypeFlags::PREFERRED))
        .or_else(|| modes.first())
        .copied()
}

/// A CRTC that can drive this connector and is not already in use.
fn pick_crtc(
    device: &GbmDevice<Card>,
    resources: &drm::control::ResourceHandles,
    connector: &connector::Info,
    used: &[crtc::Handle],
) -> Option<crtc::Handle> {
    connector
        .encoders()
        .iter()
        .filter_map(|&handle| device.get_encoder(handle).ok())
        .flat_map(|encoder| resources.filter_crtcs(encoder.possible_crtcs()))
        .find(|crtc| !used.contains(crtc))
}

/// An output's refresh period in nanoseconds, from its mode's refresh rate.
fn refresh_ns_of(mode: Mode) -> u32 {
    let hz = mode.vrefresh();
    if hz == 0 {
        0
    } else {
        u32::try_from(1_000_000_000 / u64::from(hz)).unwrap_or(0)
    }
}

/// An output's logical size — physical mode size divided by the scale.
fn logical_size(mode: Mode, scale: Scale) -> (i32, i32) {
    let (w, h) = mode.size();
    (scale.logical(i32::from(w)), scale.logical(i32::from(h)))
}

/// Describe an output the way the compositor and its clients see it.
fn describe(output: &OutputScanout, scale: Scale) -> Output {
    let (width, height) = output.mode.size();
    Output {
        id: output.id,
        name: format!("DRM-{}", output.id.0),
        description: format!("DRM output {}", output.id.0),
        geometry: OutputGeometry {
            x: output.x,
            y: 0,
            physical_width: i32::from(width),
            physical_height: i32::from(height),
            subpixel: OutputSubpixel::Unknown,
            make: String::from("DRM"),
            model: String::from("DRM"),
            transform: OutputTransform::Normal,
        },
        modes: vec![OutputMode {
            flags: OUTPUT_MODE_CURRENT | OUTPUT_MODE_PREFERRED,
            width: i32::from(width),
            height: i32::from(height),
            refresh_mhz: i32::try_from(output.mode.vrefresh().saturating_mul(1000)).unwrap_or(0),
        }],
        scale,
    }
}
