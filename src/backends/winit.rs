//! Winit display backend.

use glutin::config::GlConfig;
use glutin::context::{
    ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext, Version,
};
use glutin::display::{AsRawDisplay, GetGlDisplay, GlDisplay, RawDisplay};
use glutin::surface::{GlSurface, Surface as GlSurfaceHandle, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasWindowHandle;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    platform::scancode::PhysicalKeyExtScancode,
    platform::wayland::EventLoopBuilderExtWayland,
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

use crate::backends::BackendChannels;
use crate::dma::{DmabufImage, fourcc_name};
use crate::dmabuf_import::{DmabufCapabilities, DmabufImportProbeResult, DmabufImporter};
use crate::gl_renderer::GlRenderer;
use crate::input::{ButtonState, KeyState, MouseButton, PointerConfinement, ScrollSource};
use crate::messages::{BackendMessage, BackendRequest, CapturedFrame, PresentationFlags};
use crate::monotonic_timestamp::MonotonicTimeStamp;
use crate::outputs::{
    OUTPUT_MODE_CURRENT, OUTPUT_MODE_PREFERRED, Output, OutputGeometry, OutputId, OutputMode,
    OutputSubpixel, OutputTransform, Scale,
};
use crate::scene_graph::{Scene, SceneElement, SceneGraph};

/// The one and only output id for this winit host
const WINIT_OUTPUT_ID: OutputId = OutputId(1);

/// The refresh this backend advertises for its output, in milli-hertz (60 Hz).
/// A hosted window has no vblank of its own, so this is the nominal rate used
/// both in the output mode and to estimate frame timing.
const WINIT_REFRESH_MHZ: i32 = 60_000;

/// That refresh as a period in nanoseconds, for the frame-timing estimates a
/// hosted backend can only guess at. `1e12 / mHz`, which for any real refresh
/// is well inside `u32`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const WINIT_REFRESH_NS: u32 = (1_000_000_000_000_i64 / WINIT_REFRESH_MHZ as i64) as u32;

/// A frame request for the host output.
///
/// The predicted present is one refresh out from now: `RedrawRequested` says a
/// frame drawn now will be shown, and the host shows it on its next refresh. It
/// is an estimate — a hosted window cannot see the host's vblank — but a
/// truthful one to compose an animation against.
fn frame_request() -> BackendMessage {
    let now = MonotonicTimeStamp::now();
    let predicted_present = MonotonicTimeStamp {
        tv_sec: now.tv_sec + i64::from(now.tv_nsec + i64::from(WINIT_REFRESH_NS) >= 1_000_000_000),
        tv_nsec: (now.tv_nsec + i64::from(WINIT_REFRESH_NS)) % 1_000_000_000,
    };
    BackendMessage::FrameRequested {
        output: WINIT_OUTPUT_ID,
        predicted_present,
        refresh_ns: WINIT_REFRESH_NS,
    }
}

/// Describe the host window as the one output this backend has, reading its
/// current size and scale.
///
/// Called wherever the description is (re)sent — startup, a resize, a scale
/// change — so every report is built the same way from the same source.
///
/// The scale is the host's, rounded to 120ths. `inner_size` is already
/// physical pixels, so the scale does not change how big the framebuffer is —
/// it changes how much of it one logical pixel covers, and so how large a
/// window the compositor lays out.
fn describe_output(window: &Window) -> Output {
    let size = window.inner_size();
    Output {
        id: WINIT_OUTPUT_ID,
        name: String::from("winit"),
        description: String::from("winit display backend"),
        geometry: OutputGeometry {
            x: 0,
            y: 0,
            physical_width: size.width.cast_signed(),
            physical_height: size.height.cast_signed(),
            subpixel: OutputSubpixel::None,
            make: String::from("winit"),
            model: String::from("winit"),
            transform: OutputTransform::Normal,
        },
        modes: vec![OutputMode {
            flags: OUTPUT_MODE_CURRENT | OUTPUT_MODE_PREFERRED,
            width: size.width.cast_signed(),
            height: size.height.cast_signed(),
            refresh_mhz: WINIT_REFRESH_MHZ,
        }],
        scale: Scale::from_f64(window.scale_factor()),
    }
}

/// Events coming from outside of the window that should be handled by the
/// winit event loop
enum UserEvent {
    /// A shutdown event happening from outside that should exit the event loop
    Shutdown,
    /// A new frame is in the slot. Carries nothing: the payload is read from
    /// the watch receiver at the point of drawing, so wake-ups that pile up
    /// behind a slow frame collapse into one draw of the newest frame instead
    /// of a backlog of stale ones.
    FrameReady,
    /// The compositor has asked the backend for something. Carried into the
    /// event loop rather than handled where it arrives, because answering it
    /// needs the GL context and that only exists on this thread.
    Request(BackendRequest),
}

/// The window and everything bound to its GL context.
///
/// Created together in `resumed` because none of it is useful without the
/// rest, and dropped together so the context outlives the renderer's textures.
struct GlState {
    /// What this driver can do with dma-bufs, worked out once when the context
    /// was made. Stored rather than re-derived because probing costs a texture
    /// round trip and the answer cannot change while the context lives.
    dmabuf_support: DmabufCapabilities,
    /// The winit window
    window: Arc<Window>,
    /// The window GL surface
    surface: GlSurfaceHandle<WindowSurface>,
    /// The Gl context
    context: PossiblyCurrentContext,
    /// The Gl renderer
    renderer: GlRenderer,
}

/// Winit application
struct App {
    /// Signals that the backend is up, which is what releases the rest of
    /// startup. Sent once the window and GL context exist and the backend has
    /// reported what it can do — not merely once the thread is running.
    ///
    /// Everything a client learns at connection time is decided by then: a
    /// socket advertised earlier would let a client enumerate the globals
    /// before dma-buf support is known, and it would pick shm for the rest of
    /// its life on the strength of that.
    ready: Option<tokio::sync::oneshot::Sender<()>>,
    /// GL State data
    gl: Option<GlState>,
    backend_sender: Sender<BackendMessage>,
    /// Cancellation token for this host to cancel the compositor at large
    /// when a window is closed, or an unrecoverable error occurs
    cancel_token: CancellationToken,
    /// The newest frame the compositor has published.
    frames: watch::Receiver<SceneGraph>,
    /// Last drawn frame, for repainting on resize
    last_frame: Option<SceneGraph>,
    /// The serial of the newest scene actually drawn on each output.
    ///
    /// A published frame carries the newest scene for every output, most of
    /// which have already been drawn — the compositor recomposes an output
    /// only when that output has asked. This is what separates the one scene
    /// that is new from the ones that are being carried along.
    drawn: HashMap<OutputId, u64>,
    /// The cursor serial last drawn, so a cursor-only frame update triggers a
    /// redraw of the cursor's output without being mistaken for new content.
    drawn_cursor: u64,
    /// A dma-buf probe that arrived before there was a GL context to answer it
    /// with. Answered from `resumed` instead of being refused: the compositor
    /// asks as it starts up, and a hosted backend has no context until its
    /// window exists.
    dmabuf_probe_pending: bool,
    /// How many frames this backend has presented on each output, reported as
    /// the presentation sequence. A hosted window has no true refresh counter.
    presented: HashMap<OutputId, u64>,
    /// Whether a touch has ever arrived. winit reports no touch devices, only
    /// touch events, so this is the only evidence a touchscreen exists.
    touch_seen: bool,
    /// The title the host window is created with — the compositor's name,
    /// which is the compositor's to say, not this library's.
    window_title: String,
}

impl App {
    /// Draw a scene and put it on screen.
    ///
    /// The drawable size is read back from the window rather than taken from
    /// the scene: a resize reaches winit before the compositor has produced a
    /// scene at the new size, and drawing the old scene into the new viewport
    /// is better than skipping the frame.
    fn present_scene(&mut self, scene: &Scene, cursor: &[SceneElement]) {
        if scene.output_id != WINIT_OUTPUT_ID {
            return;
        }
        self.drawn.insert(scene.output_id, scene.serial);
        let Some(gl) = self.gl.as_mut() else {
            return;
        };
        let size = gl.window.inner_size();
        let (Some(width), Some(height)) =
            (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        else {
            return;
        };

        gl.surface.resize(&gl.context, width, height);
        gl.renderer.draw(scene, cursor, size.width, size.height);
        if let Err(e) = gl.surface.swap_buffers(&gl.context) {
            warn!("failed to swap buffers: {e}");
        }
    }

    /// The cursor elements to draw over `output`, or nothing if the pointer is
    /// elsewhere. The cursor lives beside the scenes in the frame; this backend
    /// has no cursor plane, so it composites it on top.
    fn cursor_for(scene_graph: &SceneGraph, output: OutputId) -> &[SceneElement] {
        if scene_graph.cursor.output == Some(output) {
            &scene_graph.cursor.elements
        } else {
            &[]
        }
    }

    /// Draw whatever the compositor has published since the last draw.
    /// Returns whether anything was actually drawn.
    ///
    /// Several wake-ups can arrive for one frame, or one wake-up can cover
    /// several frames, so the receiver's own change flag decides whether there
    /// is anything to do rather than the number of events. Called only from
    /// `RedrawRequested`: the host saying a frame drawn now will be shown is
    /// the only permission to draw there is. Drawing the moment a frame was
    /// published — as this backend once did — swaps at whatever rate the
    /// compositor publishes, which under a drag is input rate: hundreds of
    /// swaps a second thrashing the host's buffer pool, every one of them
    /// redrawn a second time by the redraw that followed.
    fn present_pending_frames(&mut self) -> bool {
        if !self.frames.has_changed().unwrap_or(false) {
            return false;
        }
        let frame = self.frames.borrow_and_update().clone();
        let cursor_moved = self.drawn_cursor != frame.cursor.serial;
        self.drawn_cursor = frame.cursor.serial;
        let mut drew_any = false;
        let mut presented = Vec::new();
        for scene in &frame.scenes {
            // Outputs are paced apart, so a frame is mostly scenes already on
            // screen. A scene whose serial is unchanged normally needs no
            // redraw — except when the cursor over this output moved, since the
            // cursor is composited on top and a stale one would linger. That
            // cursor-only redraw does not count as a presentation: the content
            // did not change, so its clients' frame callbacks must not fire.
            let scene_new = self.drawn.get(&scene.output_id) != Some(&scene.serial);
            let cursor_here = frame.cursor.output == Some(scene.output_id);
            let needs_redraw = scene_new || (cursor_moved && cursor_here);
            if !needs_redraw {
                continue;
            }
            self.present_scene(scene, Self::cursor_for(&frame, scene.output_id));
            drew_any = true;
            if scene_new {
                presented.push(scene.output_id);
            }
        }
        if let Some(gl) = self.gl.as_mut() {
            gl.renderer.prune_caches(&frame);
            // A snippet that failed to compile drew plain; the compositor
            // hears about it once, here, on the frame that first tried it.
            for (effect, log) in gl.renderer.take_effect_failures() {
                let _ = self
                    .backend_sender
                    .try_send(BackendMessage::EffectCompileFailed { effect, log });
            }
        }
        self.last_frame = Some(frame);

        for output_id in presented {
            // A per-output refresh counter. A hosted window has no true msc, so
            // this counts frames presented rather than reading one off the
            // hardware — honest as "how many this backend has shown", and better
            // than a constant zero for a client watching it advance.
            let sequence = self.presented.entry(output_id).or_default();
            *sequence += 1;
            // Reported even if there was no context to draw with or the window
            // had no area. The scene is dealt with either way, and a backend
            // that went quiet here would strand every client waiting on a
            // frame callback. The flags stay default: a hosted frame goes
            // through a host compositor, so none of vsync/hw_clock/… can be
            // vouched for — see `PresentationFlags`.
            let _ = self
                .backend_sender
                .try_send(BackendMessage::FramePresented {
                    output: output_id,
                    time: MonotonicTimeStamp::now(),
                    refresh_ns: WINIT_REFRESH_NS,
                    sequence: *sequence,
                    flags: PresentationFlags::default(),
                });
        }
        drew_any
    }

    /// Ask the host for a `RedrawRequested`, the only moment anything draws.
    ///
    /// A hosted backend has no vblank of its own — it is a client of another
    /// compositor, and `RedrawRequested` is that compositor telling it when a
    /// frame it draws will be shown. That is the same signal a page flip
    /// completing will be on a DRM backend, arriving by a different route.
    fn ask_for_a_frame(&mut self) {
        if let Some(gl) = self.gl.as_ref() {
            gl.window.request_redraw();
        }
    }

    /// Redraw the last frame, for when the window changed but the scene did not.
    fn repaint(&mut self) {
        let Some(frame) = self.last_frame.clone() else {
            return;
        };
        for scene in &frame.scenes {
            self.present_scene(scene, Self::cursor_for(&frame, scene.output_id));
        }
    }

    /// Create the window, EGL context, and renderer.
    fn init_gl(&self, event_loop: &ActiveEventLoop) -> anyhow::Result<GlState> {
        let window_attributes = WindowAttributes::default().with_title(self.window_title.clone());
        let (window, config) = DisplayBuilder::new()
            .with_window_attributes(Some(window_attributes))
            .build(
                event_loop,
                glutin::config::ConfigTemplateBuilder::new(),
                pick_glutin_config,
            )
            .map_err(|e| anyhow::anyhow!("failed to create GL display: {e}"))?;
        let window = Arc::new(
            window.ok_or_else(|| anyhow::anyhow!("GL display builder returned no window"))?,
        );

        let display = config.display();
        // GLES rather than desktop GL: it is what the shaders target, and what
        // is universally available on the Mesa drivers a compositor runs on.
        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(Some(Version::new(3, 0))))
            .build(Some(window.window_handle()?.as_raw()));
        // SAFETY: the window outlives the context — both live in `GlState`,
        // and `window` is declared first so it is dropped last.
        let not_current = unsafe { display.create_context(&config, &context_attributes)? };

        let surface_attributes = window.build_surface_attributes(<_>::default())?;
        // SAFETY: the attributes carry this window's handle, and the window
        // outlives the surface for the same reason as the context.
        let surface = unsafe { display.create_window_surface(&config, &surface_attributes)? };
        let context = not_current.make_current(&surface)?;
        // Only one variant exists: glutin is built here with the EGL backend
        // alone, which is also the only one a dma-buf can be imported through.
        let RawDisplay::Egl(raw_display) = display.raw_display();

        // Pacing comes from the host, through `RedrawRequested`, and a frame is
        // only composed once this backend has asked for one. Blocking the swap
        // on vblank as well would pace nothing extra and would stall the
        // thread that handles input while it waited.
        if let Err(e) = surface.set_swap_interval(&context, SwapInterval::DontWait) {
            warn!("failed to disable vsync: {e}");
        }

        // SAFETY: the display is the one the context above was made on, and it
        // stays current on this thread for as long as the importer is used.
        let importer =
            unsafe { DmabufImporter::new(raw_display, &|symbol| display.get_proc_address(symbol)) };
        let importer = match importer {
            Ok(importer) => Some(importer),
            Err(reason) => {
                info!("dma-buf import unavailable: {reason}");
                None
            }
        };

        // SAFETY: the context was just made current on this thread and stays
        // current for as long as the renderer lives.
        let renderer =
            unsafe { GlRenderer::new(|symbol| display.get_proc_address(symbol), importer)? };

        // Ask the driver what it takes, and check that it means it. Done here,
        // once, because it needs the context and nothing about the answer can
        // change while that context lives.
        let dmabuf_support = renderer.dmabuf_support();
        report_dmabuf_support(&dmabuf_support);

        Ok(GlState {
            dmabuf_support,
            window,
            surface,
            context,
            renderer,
        })
    }

    /// Try importing one client buffer and report back.
    ///
    /// Answered even when there is no context to try with: a client's `create`
    /// is waiting on this, and silence would hang it.
    fn answer_dmabuf_import(&mut self, token: u64, image: &DmabufImage) {
        let imported = self
            .gl
            .as_ref()
            .is_some_and(|gl| gl.renderer.verify_import(image));
        let _ = self
            .backend_sender
            .blocking_send(BackendMessage::DmabufImportResult { token, imported });
    }

    /// Answer a capture request with the pixels the output is showing.
    ///
    /// Answered even when there is nothing to give: whoever asked — a
    /// screenshot tool, a portal — is waiting on the token.
    fn answer_capture(&mut self, token: u64, output: OutputId, overlay_cursor: bool) {
        let capture = self.capture_output(output, overlay_cursor);
        let _ = self
            .backend_sender
            .blocking_send(BackendMessage::CaptureResult { token, capture });
    }

    /// Re-render the output's newest scene offscreen and read it back, or
    /// `None` when there is no such output, no context, or nothing composed
    /// for it yet.
    fn capture_output(&mut self, output: OutputId, overlay_cursor: bool) -> Option<CapturedFrame> {
        if output != WINIT_OUTPUT_ID {
            return None;
        }
        // Cloned so the borrow of the frame and the mutable borrow of the
        // renderer cannot collide; the clone is arcs and cursor quads, not
        // pixels.
        let frame = self.last_frame.clone()?;
        let scene = frame.scenes.iter().find(|s| s.output_id == output)?;
        let cursor = if overlay_cursor {
            Self::cursor_for(&frame, output)
        } else {
            &[]
        };
        let gl = self.gl.as_mut()?;
        let size = gl.window.inner_size();
        let pixels = gl
            .renderer
            .render_to_cpu(scene, cursor, size.width, size.height)?;
        Some(CapturedFrame {
            width: size.width,
            height: size.height,
            pixels,
        })
    }

    /// Answer a dma-buf probe, or remember to once there is a context.
    fn answer_dmabuf_probe(&mut self) {
        let Some(gl) = self.gl.as_ref() else {
            self.dmabuf_probe_pending = true;
            return;
        };
        self.dmabuf_probe_pending = false;
        let support = &gl.dmabuf_support;
        let _ = self
            .backend_sender
            .blocking_send(BackendMessage::DmabufSupport {
                formats: support.formats.clone(),
                probe: support.probe.clone(),
                device: support.device.clone(),
            });
    }
}

impl ApplicationHandler<UserEvent> for App {
    /// Called once when the app starts and is ready.  This is a bit poorly
    /// named for platforms that don't do tombstoning (desktops), but that's
    /// what `winit::application::ApplicationHandler` calls it
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gl.is_some() {
            return;
        }

        let gl = match self.init_gl(event_loop) {
            Ok(gl) => gl,
            Err(e) => {
                // Without a context there is nothing to display, and no
                // software path to fall back to, so stop rather than run blind.
                error!("failed to initialise GL backend: {e:#}");
                // Release startup even so: it is waiting on this, and a
                // compositor that is shutting down should not also hang.
                if let Some(ready) = self.ready.take() {
                    let _ = ready.send(());
                }
                let _ = self.backend_sender.blocking_send(BackendMessage::Closed);
                self.cancel_token.cancel();
                event_loop.exit();
                return;
            }
        };

        // Report hardware capabilities
        let pending_probe = self.dmabuf_probe_pending;
        let _ = self
            .backend_sender
            .blocking_send(BackendMessage::SeatCapabilities {
                pointer: true,
                keyboard: true,
                // A host window is told about touch only when a touch happens,
                // so there is nothing to report up front. The capability is
                // announced from the first touch event instead — see the
                // `WindowEvent::Touch` arm.
                touch: false,
            });
        let _ = self
            .backend_sender
            .blocking_send(BackendMessage::OutputInfo {
                outputs: vec![describe_output(&gl.window)],
            });

        gl.window.set_cursor_visible(false);
        self.gl = Some(gl);

        // A probe that arrived before there was a context to answer it with.
        if pending_probe {
            self.answer_dmabuf_probe();
        }

        // Everything a connecting client will be told now exists.
        if let Some(ready) = self.ready.take() {
            let _ = ready.send(());
        }

        // Clear the window so it is not showing whatever was in the buffer
        // before the first scene arrives. Black rather than any policy colour:
        // the background is the compositor's to choose, and it has not yet
        // composed anything to choose it in.
        let empty = Scene {
            scale: Scale::ONE,
            output_id: WINIT_OUTPUT_ID,
            background: 0xff00_0000,
            serial: 0,
            elements: Vec::new(),
            damage_from: None,
            damage: Vec::new(),
        };
        self.present_scene(&empty, &[]);
        // Nothing has been composed for this output yet, and nothing will be
        // until it is asked for. This is the first turn of that loop.
        let _ = self.backend_sender.try_send(frame_request());
    }

    /// Window event handler (called by winit event loop)
    ///
    /// One arm per winit event, and each is short; the length is the number of
    /// events rather than the complexity of any of them.
    #[allow(clippy::too_many_lines)]
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                let _ = self.backend_sender.blocking_send(BackendMessage::Closed);
                self.cancel_token.cancel();
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(gl) = self.gl.as_ref() {
                    let _ = self
                        .backend_sender
                        .blocking_send(BackendMessage::OutputChanged {
                            output: describe_output(&gl.window),
                        });
                }
                self.repaint();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                // The window moved to a monitor with a different scale, or the
                // host's scale changed under it. Re-described in full — size
                // and scale are read together off the window — so the
                // compositor re-lays-out to the new logical size just as it
                // would for a resize.
                if let Some(gl) = self.gl.as_ref() {
                    let _ = self
                        .backend_sender
                        .blocking_send(BackendMessage::OutputChanged {
                            output: describe_output(&gl.window),
                        });
                }
            }
            WindowEvent::RedrawRequested => {
                // The host is ready to show another frame — the one moment
                // anything is drawn. The newest published frame wins; with
                // nothing newly published (the host asked on its own — an
                // expose, a resize) the last frame is repainted so the window
                // is never blank. Then the compositor may compose the next
                // one, which bounds it to one frame in flight per output.
                if !self.present_pending_frames() {
                    self.repaint();
                }
                let _ = self.backend_sender.try_send(frame_request());
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(scancode) = event.physical_key.to_scancode() {
                    // The evdev code plus eight, which is the xkb keycode the
                    // compositor feeds its own keymap. Modifiers are worked out
                    // there, not here — this backend has no keymap to serialise
                    // against, and computing one against the host's layout would
                    // only disagree with the one clients are given.
                    let keycode = scancode + 8;
                    let key_state = if event.state.is_pressed() {
                        KeyState::Pressed
                    } else {
                        KeyState::Released
                    };
                    let _ = self.backend_sender.blocking_send(BackendMessage::KeyInput {
                        time: MonotonicTimeStamp::now(),
                        keycode,
                        state: key_state,
                    });
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let _ = self
                    .backend_sender
                    .blocking_send(BackendMessage::MouseMovedTo {
                        time: MonotonicTimeStamp::now(),
                        x: position.x,
                        y: position.y,
                    });
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let btn = match button {
                    winit::event::MouseButton::Left => MouseButton::LEFT,
                    winit::event::MouseButton::Right => MouseButton::RIGHT,
                    winit::event::MouseButton::Middle => MouseButton::MIDDLE,
                    winit::event::MouseButton::Back => MouseButton::BACK,
                    winit::event::MouseButton::Forward => MouseButton::FORWARD,
                    // winit's `Other` codes are platform-defined numbers, not
                    // evdev codes, so forwarding one as if it were would name
                    // a button the user did not press.
                    winit::event::MouseButton::Other(_) => return,
                };
                let st = if state.is_pressed() {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                };
                let _ = self
                    .backend_sender
                    .blocking_send(BackendMessage::MouseButton {
                        time: MonotonicTimeStamp::now(),
                        button: btn,
                        state: st,
                    });
            }
            WindowEvent::Touch(touch) => {
                // winit has no "a touchscreen exists" signal, so the first
                // touch is the signal: the seat gains the capability then, and
                // clients that care re-read it. Announcing it up front would
                // claim a device that may not exist.
                if !self.touch_seen {
                    self.touch_seen = true;
                    let _ = self
                        .backend_sender
                        .blocking_send(BackendMessage::SeatCapabilities {
                            pointer: true,
                            keyboard: true,
                            touch: true,
                        });
                }

                // The id is a `u64` from winit and an `i32` on the wire. Real
                // devices number their fingers from zero, so the truncation is
                // theoretical, but wrapping it deliberately keeps two fingers
                // from ever colliding on one id.
                let id = i32::try_from(touch.id % u64::from(i32::MAX.cast_unsigned())).unwrap_or(0);
                let (x, y) = (touch.location.x, touch.location.y);
                let time = MonotonicTimeStamp::now();
                let message = match touch.phase {
                    winit::event::TouchPhase::Started => {
                        BackendMessage::TouchDown { time, id, x, y }
                    }
                    winit::event::TouchPhase::Moved => {
                        BackendMessage::TouchMotion { time, id, x, y }
                    }
                    winit::event::TouchPhase::Ended => BackendMessage::TouchUp { time, id },
                    winit::event::TouchPhase::Cancelled => BackendMessage::TouchCancel,
                };
                let _ = self.backend_sender.blocking_send(message);
            }
            WindowEvent::MouseWheel { delta, phase, .. } => {
                // A touchpad reports its scroll ending, and that end is
                // information a client cannot infer from the deltas.
                if phase == winit::event::TouchPhase::Ended {
                    let _ = self
                        .backend_sender
                        .blocking_send(BackendMessage::MouseScrollEnd {
                            time: MonotonicTimeStamp::now(),
                        });
                    return;
                }

                // The two delta kinds are the two sources: lines come from a
                // wheel, which clicks, and pixels from a touchpad, which does
                // not. winit does not say which device it was, but it does say
                // which unit — and the unit only exists because the devices
                // differ.
                let (dx, dy, source, v120_x, v120_y) = match delta {
                    #[allow(clippy::cast_possible_truncation)]
                    winit::event::MouseScrollDelta::LineDelta(x, y) => (
                        f64::from(x),
                        f64::from(y),
                        ScrollSource::Wheel,
                        (x * 120.0) as i32,
                        (y * 120.0) as i32,
                    ),
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        (pos.x, pos.y, ScrollSource::Finger, 0, 0)
                    }
                };
                let _ = self
                    .backend_sender
                    .blocking_send(BackendMessage::MouseScroll {
                        time: MonotonicTimeStamp::now(),
                        dx,
                        dy,
                        source,
                        v120_x,
                        v120_y,
                    });
            }
            WindowEvent::Focused(focused) => {
                let msg = if focused {
                    BackendMessage::FocusIn
                } else {
                    BackendMessage::FocusOut
                };
                let _ = self.backend_sender.blocking_send(msg);
            }
            _ => {}
        }
    }

    /// Device event handler (called by winit event loop): raw input that is
    /// not tied to the window.
    ///
    /// Relative pointer motion comes in here — unaccelerated deltas from the
    /// host's relative-pointer protocol — and flows alongside the absolute
    /// positions from `CursorMoved`. Both are sent all the time: a client
    /// consuming `zwp_relative_pointer_v1` wants the deltas whether or not
    /// the pointer is locked, and the compositor picks which to route.
    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            let _ = self
                .backend_sender
                .blocking_send(BackendMessage::MouseMovedBy {
                    time: MonotonicTimeStamp::now(),
                    dx,
                    dy,
                });
        }
    }

    /// User event handler (called by winit event loop)
    /// Winit separates this handler from the normal `windows_event` handler above
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Shutdown => {
                event_loop.exit();
            }
            // Only a nudge: the frame stays in the slot until the host says
            // a drawn frame will be shown, and several nudges coalesce into
            // the one redraw that follows.
            UserEvent::FrameReady => self.ask_for_a_frame(),
            UserEvent::Request(BackendRequest::ProbeDmabuf) => self.answer_dmabuf_probe(),
            UserEvent::Request(BackendRequest::ImportDmabuf { token, image }) => {
                self.answer_dmabuf_import(token, &image);
            }
            UserEvent::Request(BackendRequest::CaptureOutput {
                token,
                output,
                overlay_cursor,
            }) => {
                self.answer_capture(token, output, overlay_cursor);
            }
            UserEvent::Request(BackendRequest::SetPointerConfinement { mode }) => {
                // Ask the host; on hardware this would be ours to enforce,
                // but here the pointer belongs to the host compositor.
                if let Some(gl) = self.gl.as_ref() {
                    let grab = match mode {
                        PointerConfinement::None => CursorGrabMode::None,
                        PointerConfinement::Confined => CursorGrabMode::Confined,
                        PointerConfinement::Locked => CursorGrabMode::Locked,
                    };
                    if let Err(e) = gl.window.set_cursor_grab(grab) {
                        warn!("host refused pointer confinement {mode:?}: {e}");
                    }
                }
            }
            UserEvent::Request(BackendRequest::SetOutputSize {
                output,
                width,
                height,
            }) => {
                // Ask the host; whether it obliges is its call, and if it does
                // the `Resized` event reports the change as `OutputChanged` —
                // the same route an unasked-for resize takes.
                if output == WINIT_OUTPUT_ID
                    && let Some(gl) = self.gl.as_ref()
                {
                    let _ = gl.window.request_inner_size(
                        winit::dpi::PhysicalSize::new(width.max(1), height.max(1)).cast::<u32>(),
                    );
                }
            }
        }
    }
}

/// Log what the driver said about dma-buf, at the level the answer deserves.
///
/// A failed probe is a warning rather than a debug line: the extensions are
/// there, so a client will be told dma-buf works, and it will not.
fn report_dmabuf_support(support: &DmabufCapabilities) {
    let device = support.device.as_ref().map_or_else(
        || String::from("an unnamed device"),
        |node| node.path.display().to_string(),
    );
    match &support.probe {
        DmabufImportProbeResult::Passed => info!(
            "dma-buf import working on {device}: {} format(s), e.g. {}",
            support.formats.len(),
            support
                .formats
                .iter()
                .take(4)
                .map(|f| fourcc_name(f.fourcc))
                .collect::<Vec<_>>()
                .join(", "),
        ),
        DmabufImportProbeResult::Unsupported(reason) => {
            info!("dma-buf import unavailable: {reason}");
        }
        DmabufImportProbeResult::Untested(reason) => info!(
            "dma-buf import available on {device} for {} format(s) but unverified: {reason}",
            support.formats.len(),
        ),
        DmabufImportProbeResult::Failed(reason) => {
            warn!("dma-buf import is advertised by the driver but does not work: {reason}");
        }
    }
}

/// Prefer the config with the fewest samples: this compositor draws axis-aligned
/// quads, so multisampling would cost bandwidth and change nothing.
fn pick_glutin_config(
    configs: Box<dyn Iterator<Item = glutin::config::Config> + '_>,
) -> glutin::config::Config {
    configs
        .reduce(|best, config| {
            if config.num_samples() < best.num_samples() {
                config
            } else {
                best
            }
        })
        .expect("no GL config available")
}

/// Runs this wayland backend, waiting for frames from the compositor and
/// sending over input events from keyboard/mouse, etc.
///
/// Blocks on the window event loop until shutdown, so it must own its thread;
/// it also spawns onto the current tokio runtime, so that thread must have a
/// runtime handle entered. `window_title` names the host window — the
/// compositor's own name, since a hosted backend is a window on someone's
/// desktop and the title is how the user tells whose.
///
/// # Errors
/// Returns an error if there any problems initializing winit
pub fn run_winit_backend(window_title: &str, channels: BackendChannels) -> anyhow::Result<()> {
    let BackendChannels {
        messages: backend_sender,
        ready: ready_tx,
        frames: frame_rx,
        mut requests,
        cancel: cancel_token,
    } = channels;
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .with_any_thread(true)
        .build()?;

    let proxy: EventLoopProxy<UserEvent> = event_loop.create_proxy();
    let rt = tokio::runtime::Handle::current();

    let shutdown_proxy = proxy.clone();
    let cancel_token_for_shutdown = cancel_token.clone();
    rt.spawn(async move {
        cancel_token_for_shutdown.cancelled().await;
        let _ = shutdown_proxy.send_event(UserEvent::Shutdown);
    });

    // Only nudges the event loop; the frame itself stays in the slot until the
    // winit thread is ready to draw it.
    let frame_proxy = proxy.clone();
    let mut notify_rx = frame_rx.clone();
    rt.spawn(async move {
        while notify_rx.changed().await.is_ok() {
            if frame_proxy.send_event(UserEvent::FrameReady).is_err() {
                break;
            }
        }
    });

    // Requests are handled on the winit thread because answering them needs
    // the GL context, so they come in as user events rather than being read
    // where they arrive.
    let request_proxy = proxy.clone();
    rt.spawn(async move {
        while let Some(request) = requests.recv().await {
            if request_proxy
                .send_event(UserEvent::Request(request))
                .is_err()
            {
                break;
            }
        }
    });

    let mut app = App {
        ready: Some(ready_tx),
        gl: None,
        backend_sender,
        cancel_token: cancel_token.clone(),
        frames: frame_rx,
        last_frame: None,
        drawn: HashMap::new(),
        drawn_cursor: 0,
        dmabuf_probe_pending: false,
        presented: HashMap::new(),
        touch_seen: false,
        window_title: window_title.to_owned(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
