//! DRM/KMS + libinput backend (bare hardware).
//!
//! The compositor as a login session on a Linux seat, driving the display
//! and input devices directly — no host compositor beneath it. This is the
//! backend a shipped compositor runs on; the winit backend is for developing
//! against it from inside another desktop.
//!
//! This backend links system libraries the others do not (libdrm, gbm,
//! libinput, libudev, libseat, EGL), so the crate as a whole builds only
//! where those are present — see `flake.nix` for the build inputs.
//!
//! # Untested on hardware
//!
//! Every line here is compiled against the real crates but has not been run
//! on a GPU in this repository's environment. Treat it as a careful first
//! implementation to iterate on, not a proven one.
//!
//! # Shape
//!
//! Three pieces, tied together by a tokio event loop:
//!
//! - [`session`] — a seat, through libseat: the master DRM fd and the input
//!   device fds without running as root, and the enable/disable that VT
//!   switching turns on and off.
//! - [`scanout`] — one DRM device: connectors and their modes, a GBM surface
//!   fed by EGL and the shared [`GlRenderer`](crate::gl_renderer::GlRenderer),
//!   and the page-flip loop that paces each output.
//! - [`libinput_source`] — libinput, translated into the same
//!   [`BackendMessage`] stream the other backends produce.
//!
//! Every fd the loop waits on — DRM, libinput, libseat — is registered with
//! tokio through [`AsyncFd`], so the backend is driven by the same runtime as
//! the channels rather than a thread of its own blocking on `poll`.

pub mod libinput_source;
pub mod scanout;
pub mod session;
pub mod vt_switch;

use std::cell::RefCell;
use std::collections::HashMap;
use std::os::fd::{AsRawFd, RawFd};
use std::path::PathBuf;
use std::rc::Rc;

use tokio::io::unix::AsyncFd;
use tracing::{info, warn};

use crate::backends::BackendChannels;
use crate::messages::{BackendMessage, BackendRequest};
use crate::monotonic_timestamp::MonotonicTimeStamp;
use crate::outputs::OutputId;
use crate::scene_graph::{SceneElement, SceneGraph};

use libinput_source::Input;
use scanout::{Presented, Scanout};
use session::Session;
use vt_switch::{KeyAction, VtKeys};

/// How to start the DRM backend.
#[derive(Debug, Clone, Default)]
pub struct DrmConfig {
    /// The DRM device to drive, e.g. `/dev/dri/card0`. `None` picks the
    /// first card node the seat will open.
    pub device: Option<PathBuf>,
    /// The seat name, as libseat and udev know it. `None` uses `seat0`, the
    /// seat a single-seat machine always has.
    pub seat: Option<String>,
    /// Whether a tap on a touchpad is a click. `None` leaves each device on
    /// libinput's own default, which is usually off.
    pub tap_to_click: Option<bool>,
}

/// A borrowed fd registered with tokio only to be watched, never closed.
///
/// The seat, libinput and DRM fds are owned by those libraries; tokio's
/// [`AsyncFd`] takes a `T: AsRawFd` and never closes it unless `T`'s drop
/// does, and this wrapper has no drop — so registering it watches without
/// taking ownership.
struct WatchedFd(RawFd);

impl AsRawFd for WatchedFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

/// Run the DRM/KMS + libinput backend until cancelled.
///
/// The whole backend is one thread's worth of `!Send` state — the GL context,
/// libseat, libinput — so this future is `!Send`. Run it on a current-thread
/// tokio runtime with IO enabled, on a thread of its own:
///
/// ```no_run
/// # use tokio_way_backends::backends::{BackendChannels, drm::{run_drm_backend, DrmConfig}};
/// # fn go(channels: BackendChannels) {
/// std::thread::spawn(move || {
///     let rt = tokio::runtime::Builder::new_current_thread()
///         .enable_all()
///         .build()
///         .unwrap();
///     rt.block_on(run_drm_backend(DrmConfig::default(), channels))
/// });
/// # }
/// ```
///
/// # Errors
/// If no seat, device, or output can be brought up. Once running, per-frame
/// failures are logged and the loop continues.
// One `select!` over every source the backend waits on; the length is the
// number of arms, not the complexity of any one.
#[allow(clippy::too_many_lines)]
pub async fn run_drm_backend(config: DrmConfig, channels: BackendChannels) -> anyhow::Result<()> {
    let BackendChannels {
        messages,
        ready,
        mut frames,
        mut requests,
        cancel,
    } = channels;

    // The seat first: without it there is no device to open. Wait for the
    // first enable before touching hardware.
    let mut session = Session::open()?;
    let seat_raw = session.poll_fd()?;
    let seat_afd = AsyncFd::new(WatchedFd(seat_raw))?;
    while !session.is_active() {
        tokio::select! {
            () = cancel.cancelled() => return Ok(()),
            guard = seat_afd.readable() => {
                guard?.clear_ready();
                session.dispatch()?;
            }
        }
    }

    // The device, and the outputs it lights.
    let device_path = choose_device(&config)?;
    info!("opening DRM device {}", device_path.display());
    let drm_device = session.open_device(&device_path)?;
    let mut scanout = Scanout::open(drm_device)?;

    // Everything a connecting client will be told, then the go-ahead.
    let _ = messages
        .send(BackendMessage::SeatCapabilities {
            pointer: true,
            keyboard: true,
            // A touchscreen is reported by its first event rather than
            // enumerated up front, as in the winit backend.
            touch: false,
        })
        .await;
    let _ = messages
        .send(BackendMessage::OutputInfo {
            outputs: scanout.output_descriptions(),
        })
        .await;
    let _ = ready.send(());

    // libinput shares the session, so the seat can hand it device fds.
    let seat_name = config.seat.clone().unwrap_or_else(|| String::from("seat0"));
    let session = Rc::new(RefCell::new(session));
    let mut input = Input::new(Rc::clone(&session), &seat_name, config.tap_to_click)?;

    let drm_afd = AsyncFd::new(WatchedFd(scanout.drm_fd()))?;
    let input_afd = AsyncFd::new(WatchedFd(input.poll_fd()))?;

    // The first turn of every output's pacing loop: nothing is composed until
    // it is asked for.
    for id in scanout.output_ids() {
        let _ = messages.send(frame_request(id, scanout.refresh_ns(id))).await;
    }

    // Per-output serial last drawn, and the cursor serial, exactly as the
    // winit backend tracks them.
    let mut drawn: HashMap<OutputId, u64> = HashMap::new();
    let mut drawn_cursor = 0u64;
    let mut last_frame: Option<SceneGraph> = None;

    // The held keys, watched for Ctrl+Alt+Fn — see [`vt_switch`].
    let mut vt_keys = VtKeys::default();

    loop {
        tokio::select! {
            () = cancel.cancelled() => break,

            // A page flip completed: report the presentation and ask for the
            // next frame — the hardware vblank driving the pace.
            guard = drm_afd.readable() => {
                let mut guard = guard?;
                match scanout.handle_events() {
                    Ok(done) => {
                        for flip in done {
                            let _ = messages.send(BackendMessage::FramePresented {
                                output: flip.output,
                                time: scanout::presentation_time(),
                                refresh_ns: flip.refresh_ns,
                                sequence: flip.sequence,
                                flags: scanout::scanout_flags(),
                            }).await;
                            let _ = messages.send(frame_request(flip.output, flip.refresh_ns)).await;
                        }
                    }
                    Err(e) => warn!("draining DRM events failed: {e}"),
                }
                guard.clear_ready();
            }

            // A seat event: enable/disable runs inside dispatch. Coming back
            // from a VT switch means the CRTC config is gone and must be
            // rebuilt, so re-modeset and re-request every output.
            guard = seat_afd.readable() => {
                let mut guard = guard?;
                let was_active = session.borrow().is_active();
                if let Err(e) = session.borrow_mut().dispatch() {
                    warn!("seat dispatch failed: {e}");
                }
                let now_active = session.borrow().is_active();
                if now_active && !was_active {
                    info!("session re-enabled; re-modesetting");
                    // The evdev fds died with the disable; this reopens the
                    // devices through the seat — see [`Input::resume`].
                    if let Err(e) = input.resume() {
                        warn!("{e}");
                    }
                    scanout.mark_needs_modeset();
                    drawn.clear();
                    // Put the last frame straight back on screen. Waiting for
                    // the compositor instead means waiting for it to have a
                    // reason to publish — and if nothing changed while the
                    // session was away, it has none, and the user is looking
                    // at a black screen until something does.
                    if let Some(frame) = &last_frame {
                        for scene in &frame.scenes {
                            let id = scene.output_id;
                            drawn.insert(id, scene.serial);
                            if let Presented::Immediately { output, refresh_ns, sequence } =
                                scanout.render_output(id, scene, cursor_for(frame, id))
                            {
                                let _ = messages.send(BackendMessage::FramePresented {
                                    output,
                                    time: scanout::presentation_time(),
                                    refresh_ns,
                                    sequence,
                                    flags: scanout::scanout_flags(),
                                }).await;
                            }
                        }
                    }
                    for id in scanout.output_ids() {
                        let _ = messages.send(frame_request(id, scanout.refresh_ns(id))).await;
                    }
                }
                if was_active && !now_active {
                    // Going away: every key still down will be released on
                    // some other VT where this process cannot see it, so the
                    // compositor is told now — otherwise it comes back with
                    // Ctrl and Alt stuck pressed. See [`VtKeys::release_all`].
                    for message in vt_keys.release_all() {
                        let _ = messages.send(message).await;
                    }
                    // Input's revoked fds closed before the acknowledgment
                    // below, so the manager hears "done" once it is true.
                    input.suspend();
                }
                // Unconditional, and last: a no-op unless a disable is
                // waiting, and anything the loop had to quiesce first has
                // been by now.
                session.borrow_mut().acknowledge_disable();
                guard.clear_ready();
            }

            // Input: translated and forwarded while active. While inactive the
            // fd is still drained, so the loop does not spin on it.
            guard = input_afd.readable() => {
                let mut guard = guard?;
                let active = session.borrow().is_active();
                let size = scanout
                    .output_ids()
                    .first()
                    .and_then(|id| scanout.output_physical_size(*id))
                    .unwrap_or((1, 1));
                match input.dispatch(size) {
                    Ok(events) if active => {
                        for message in events {
                            match vt_keys.on_key(&message) {
                                KeyAction::Forward => {
                                    let _ = messages.send(message).await;
                                }
                                KeyAction::Swallow => {}
                                KeyAction::SwitchVt(vt) => {
                                    info!("Ctrl+Alt chord: asking for VT {vt}");
                                    if let Err(e) = session.borrow_mut().switch_session(vt) {
                                        warn!("{e}");
                                    }
                                }
                            }
                        }
                    }
                    Ok(_) => {}
                    Err(e) => warn!("libinput dispatch failed: {e}"),
                }
                guard.clear_ready();
            }

            // A new frame from the compositor: draw the outputs whose scene
            // changed (or whose cursor moved), unless mid-flip. An immediate
            // present — the first frame's modeset — is reported at once; a
            // queued flip waits for its event above.
            changed = frames.changed() => {
                if changed.is_err() {
                    break;
                }
                let frame = frames.borrow_and_update().clone();
                if session.borrow().is_active() {
                    let cursor_moved = drawn_cursor != frame.cursor.serial;
                    drawn_cursor = frame.cursor.serial;
                    for scene in &frame.scenes {
                        let id = scene.output_id;
                        if scanout.awaiting_flip(id) {
                            continue;
                        }
                        let scene_new = drawn.get(&id) != Some(&scene.serial);
                        let cursor_here = frame.cursor.output == Some(id);
                        if !(scene_new || cursor_moved && cursor_here) {
                            continue;
                        }
                        drawn.insert(id, scene.serial);
                        let cursor = cursor_for(&frame, id);
                        if let Presented::Immediately { output, refresh_ns, sequence } =
                            scanout.render_output(id, scene, cursor)
                        {
                            let _ = messages.send(BackendMessage::FramePresented {
                                output,
                                time: scanout::presentation_time(),
                                refresh_ns,
                                sequence,
                                flags: scanout::scanout_flags(),
                            }).await;
                            let _ = messages.send(frame_request(output, refresh_ns)).await;
                        }
                    }
                    scanout.prune_caches(&frame);
                    for (effect, log) in scanout.take_effect_failures() {
                        let _ = messages.send(BackendMessage::EffectCompileFailed { effect, log }).await;
                    }
                }
                last_frame = Some(frame);
            }

            // A compositor request.
            request = requests.recv() => {
                match request {
                    Some(request) => {
                        handle_request(request, &mut scanout, last_frame.as_ref(), &messages).await;
                    }
                    None => break,
                }
            }
        }
    }
    Ok(())
}

/// The cursor elements over one output, or nothing if the pointer is
/// elsewhere — this backend composites the cursor rather than using a plane.
fn cursor_for(frame: &SceneGraph, output: OutputId) -> &[SceneElement] {
    if frame.cursor.output == Some(output) {
        &frame.cursor.elements
    } else {
        &[]
    }
}

/// A frame request predicting presentation one refresh out, as the winit
/// backend does — a frame drawn now is shown at the next vblank.
fn frame_request(output: OutputId, refresh_ns: u32) -> BackendMessage {
    let now = MonotonicTimeStamp::now();
    let nsec = now.tv_nsec + i64::from(refresh_ns);
    BackendMessage::FrameRequested {
        output,
        predicted_present: MonotonicTimeStamp {
            tv_sec: now.tv_sec + nsec / 1_000_000_000,
            tv_nsec: nsec % 1_000_000_000,
        },
        refresh_ns,
    }
}

/// Answer one compositor request.
async fn handle_request(
    request: BackendRequest,
    scanout: &mut Scanout,
    last_frame: Option<&SceneGraph>,
    messages: &tokio::sync::mpsc::Sender<BackendMessage>,
) {
    match request {
        BackendRequest::ProbeDmabuf => {
            let support = scanout.dmabuf_support();
            let _ = messages
                .send(BackendMessage::DmabufSupport {
                    formats: support.formats,
                    probe: support.probe,
                    device: support.device,
                })
                .await;
        }
        BackendRequest::ImportDmabuf { token, image } => {
            let imported = scanout.verify_import(&image);
            let _ = messages
                .send(BackendMessage::DmabufImportResult { token, imported })
                .await;
        }
        BackendRequest::CaptureOutput {
            token,
            output,
            overlay_cursor,
        } => {
            let capture = last_frame.and_then(|frame| {
                let scene = frame.scenes.iter().find(|s| s.output_id == output)?;
                let cursor = if overlay_cursor {
                    cursor_for(frame, output)
                } else {
                    &[]
                };
                scanout.capture(output, scene, cursor)
            });
            let _ = messages
                .send(BackendMessage::CaptureResult { token, capture })
                .await;
        }
        // Two requests this backend does not act on, both by their own
        // best-effort contract. Runtime mode switching is not implemented, so
        // the startup mode stands and no `OutputChanged` follows a resize.
        // And on hardware the cursor is the compositor's own to place, with no
        // host to ask to confine the pointer — the relative motion a
        // locked-pointer client consumes flows regardless of confinement.
        BackendRequest::SetOutputSize { .. } | BackendRequest::SetPointerConfinement { .. } => {}
    }
}

/// The DRM node to drive: the configured one, or the first `card*` in
/// `/dev/dri`. The seat validates it when it is opened.
fn choose_device(config: &DrmConfig) -> anyhow::Result<PathBuf> {
    if let Some(device) = &config.device {
        return Ok(device.clone());
    }
    let mut cards: Vec<PathBuf> = std::fs::read_dir("/dev/dri")
        .map_err(|e| anyhow::anyhow!("cannot read /dev/dri: {e}"))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("card"))
        })
        .collect();
    cards.sort();
    cards
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no DRM card device in /dev/dri"))
}

