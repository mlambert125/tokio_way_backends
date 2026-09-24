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
const WINIT_REFRESH_MHZ: i32 = 60_000;

/// That refresh as a period in nanoseconds
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const WINIT_REFRESH_NS: u32 = (1_000_000_000_000_i64 / WINIT_REFRESH_MHZ as i64) as u32;

/// A frame request for the host output
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

/// Describe the host window as the one output this backend has, reading its current size and scale.
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

/// Events coming from outside of the window that should be handled by the winit event loop
enum UserEvent {
    /// A shutdown event happening from outside that should exit the event loop
    Shutdown,
    /// A new frame is in the slot
    FrameReady,
    /// A backend request from the compositor came in
    Request(BackendRequest),
}

/// The window and everything bound to its GL context
struct GlState {
    /// What this can do with dma-buf
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
    /// One-shot that signals that the backend is up
    ready: Option<tokio::sync::oneshot::Sender<()>>,
    /// GL State data
    gl: Option<GlState>,
    /// The channel to send messages to the compositor
    backend_sender: Sender<BackendMessage>,
    /// Cancellation token for this host to cancel the compositor at large when a window is closed
    cancel_token: CancellationToken,
    /// The newest frame the compositor has published
    frames: watch::Receiver<SceneGraph>,
    /// Last drawn frame, for repainting on resize
    last_frame: Option<SceneGraph>,
    /// The serial of the newest scene actually drawn on each output
    drawn: HashMap<OutputId, u64>,
    /// The cursor serial last drawn
    drawn_cursor: u64,
    /// Whether a dma-buf probe arrived before there was a context to answer it with
    dmabuf_probe_pending: bool,
    /// How many frames this backend has presented on each output
    presented: HashMap<OutputId, u64>,
    /// Whether a touch has ever arrived
    touch_seen: bool,
    /// The title the host window is created with
    window_title: String,
}

impl App {
    /// Draw a scene and put it on screen.
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

    /// The cursor elements to draw over `output`
    fn cursor_for(scene_graph: &SceneGraph, output: OutputId) -> &[SceneElement] {
        if scene_graph.cursor.output == Some(output) {
            &scene_graph.cursor.elements
        } else {
            &[]
        }
    }

    /// Draw whatever the compositor has published since the last draw
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
            for (effect, log) in gl.renderer.take_effect_failures() {
                let _ = self
                    .backend_sender
                    .try_send(BackendMessage::EffectCompileFailed { effect, log });
            }
        }
        self.last_frame = Some(frame);

        for output_id in presented {
            let sequence = self.presented.entry(output_id).or_default();
            *sequence += 1;
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

    /// Ask the host when to draw next, and pass that on as a frame request
    fn ask_for_a_frame(&mut self) {
        if let Some(gl) = self.gl.as_ref() {
            gl.window.request_redraw();
        }
    }

    /// Redraw the last frame, for when the window changed but the scene did not
    fn repaint(&mut self) {
        let Some(frame) = self.last_frame.clone() else {
            return;
        };
        for scene in &frame.scenes {
            self.present_scene(scene, Self::cursor_for(&frame, scene.output_id));
        }
    }

    /// Create the window, EGL context and renderer
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
        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(Some(Version::new(3, 0))))
            .build(Some(window.window_handle()?.as_raw()));
        let not_current = unsafe { display.create_context(&config, &context_attributes)? };

        let surface_attributes = window.build_surface_attributes(<_>::default())?;
        let surface = unsafe { display.create_window_surface(&config, &surface_attributes)? };
        let context = not_current.make_current(&surface)?;
        let RawDisplay::Egl(raw_display) = display.raw_display();

        if let Err(e) = surface.set_swap_interval(&context, SwapInterval::DontWait) {
            warn!("failed to disable vsync: {e}");
        }

        let importer =
            unsafe { DmabufImporter::new(raw_display, &|symbol| display.get_proc_address(symbol)) };
        let importer = match importer {
            Ok(importer) => Some(importer),
            Err(reason) => {
                info!("dma-buf import unavailable: {reason}");
                None
            }
        };

        let renderer =
            unsafe { GlRenderer::new(|symbol| display.get_proc_address(symbol), importer)? };

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

    /// Try importing one client buffer and report back
    fn answer_dmabuf_import(&mut self, token: u64, image: &DmabufImage) {
        let imported = self
            .gl
            .as_ref()
            .is_some_and(|gl| gl.renderer.verify_import(image));
        let _ = self
            .backend_sender
            .blocking_send(BackendMessage::DmabufImportResult { token, imported });
    }

    /// Answer a capture request with the pixels the output is showing
    fn answer_capture(&mut self, token: u64, output: OutputId, overlay_cursor: bool) {
        let capture = self.capture_output(output, overlay_cursor);
        let _ = self
            .backend_sender
            .blocking_send(BackendMessage::CaptureResult { token, capture });
    }

    /// Re-render the output's newest scene offscreen and read it back
    fn capture_output(&mut self, output: OutputId, overlay_cursor: bool) -> Option<CapturedFrame> {
        if output != WINIT_OUTPUT_ID {
            return None;
        }
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
    /// Called once when the app starts and is ready. Named this way for platforms that can tombstone
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gl.is_some() {
            return;
        }

        let gl = match self.init_gl(event_loop) {
            Ok(gl) => gl,
            Err(e) => {
                error!("failed to initialise GL backend: {e:#}");
                if let Some(ready) = self.ready.take() {
                    let _ = ready.send(());
                }
                let _ = self.backend_sender.blocking_send(BackendMessage::Closed);
                self.cancel_token.cancel();
                event_loop.exit();
                return;
            }
        };

        let pending_probe = self.dmabuf_probe_pending;
        let _ = self
            .backend_sender
            .blocking_send(BackendMessage::SeatCapabilities {
                pointer: true,
                keyboard: true,
                touch: false,
            });
        let _ = self
            .backend_sender
            .blocking_send(BackendMessage::OutputInfo {
                outputs: vec![describe_output(&gl.window)],
            });

        gl.window.set_cursor_visible(false);
        self.gl = Some(gl);

        if pending_probe {
            self.answer_dmabuf_probe();
        }

        if let Some(ready) = self.ready.take() {
            let _ = ready.send(());
        }

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
        let _ = self.backend_sender.try_send(frame_request());
    }

    /// Window event handler
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
                if let Some(gl) = self.gl.as_ref() {
                    let _ = self
                        .backend_sender
                        .blocking_send(BackendMessage::OutputChanged {
                            output: describe_output(&gl.window),
                        });
                }
            }
            WindowEvent::RedrawRequested => {
                if !self.present_pending_frames() {
                    self.repaint();
                }
                let _ = self.backend_sender.try_send(frame_request());
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(scancode) = event.physical_key.to_scancode() {
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
                if phase == winit::event::TouchPhase::Ended {
                    let _ = self
                        .backend_sender
                        .blocking_send(BackendMessage::MouseScrollEnd {
                            time: MonotonicTimeStamp::now(),
                        });
                    return;
                }

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

    /// Device event handler
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

    /// User event handler
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Shutdown => {
                event_loop.exit();
            }
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

/// Log what the driver said about dma-buf
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

/// Prefer the config with the fewest samples
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

    let frame_proxy = proxy.clone();
    let mut notify_rx = frame_rx.clone();
    rt.spawn(async move {
        while notify_rx.changed().await.is_ok() {
            if frame_proxy.send_event(UserEvent::FrameReady).is_err() {
                break;
            }
        }
    });

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
