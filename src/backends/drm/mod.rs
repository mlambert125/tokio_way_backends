//! DRM/KMS + libinput backend (bare hardware)

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
    /// The DRM device to drive, e.g. `/dev/dri/card0`. `None` picks the first card node
    pub device: Option<PathBuf>,
    /// The seat name, as libseat and udev know it. `None` uses `seat0`
    pub seat: Option<String>,
    /// Whether a tap on a touchpad is a click. `None` leaves each device on libinput's
    pub tap_to_click: Option<bool>,
}

/// A borrowed fd registered with tokio only to be watched, never closed
struct WatchedFd(RawFd);

impl AsRawFd for WatchedFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

/// Run the DRM/KMS + libinput backend until cancelled
///
/// # Errors
///
/// If no seat, device, or output can be opened
#[allow(clippy::too_many_lines)]
pub async fn run_drm_backend(config: DrmConfig, channels: BackendChannels) -> anyhow::Result<()> {
    let BackendChannels {
        messages,
        ready,
        mut frames,
        mut requests,
        cancel,
    } = channels;

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

    let device_path = choose_device(&config)?;
    info!("opening DRM device {}", device_path.display());
    let drm_device = session.open_device(&device_path)?;
    let mut scanout = Scanout::open(drm_device)?;

    let _ = messages
        .send(BackendMessage::SeatCapabilities {
            pointer: true,
            keyboard: true,
            touch: false,
        })
        .await;
    let _ = messages
        .send(BackendMessage::OutputInfo {
            outputs: scanout.output_descriptions(),
        })
        .await;
    let _ = ready.send(());

    let seat_name = config.seat.clone().unwrap_or_else(|| String::from("seat0"));
    let session = Rc::new(RefCell::new(session));
    let mut input = Input::new(Rc::clone(&session), &seat_name, config.tap_to_click)?;

    let drm_afd = AsyncFd::new(WatchedFd(scanout.drm_fd()))?;
    let input_afd = AsyncFd::new(WatchedFd(input.poll_fd()))?;

    for id in scanout.output_ids() {
        let _ = messages
            .send(frame_request(id, scanout.refresh_ns(id)))
            .await;
    }

    let mut drawn: HashMap<OutputId, u64> = HashMap::new();
    let mut drawn_cursor = 0u64;
    let mut last_frame: Option<SceneGraph> = None;

    let mut vt_keys = VtKeys::default();

    loop {
        tokio::select! {
            () = cancel.cancelled() => break,

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

            guard = seat_afd.readable() => {
                let mut guard = guard?;
                let was_active = session.borrow().is_active();
                if let Err(e) = session.borrow_mut().dispatch() {
                    warn!("seat dispatch failed: {e}");
                }
                let now_active = session.borrow().is_active();
                if now_active && !was_active {
                    info!("session re-enabled; re-modesetting");

                    if let Err(e) = input.resume() {
                        warn!("{e}");
                    }

                    scanout.mark_needs_modeset();
                    drawn.clear();

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
                    for message in vt_keys.release_all() {
                        let _ = messages.send(message).await;
                    }
                    input.suspend();
                }

                session.borrow_mut().acknowledge_disable();
                guard.clear_ready();
            }

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

/// The mouse cursor elements over one output
fn cursor_for(frame: &SceneGraph, output: OutputId) -> &[SceneElement] {
    if frame.cursor.output == Some(output) {
        &frame.cursor.elements
    } else {
        &[]
    }
}

/// A frame request predicting presentation one refresh out
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
        BackendRequest::SetOutputSize { .. } | BackendRequest::SetPointerConfinement { .. } => {}
    }
}

/// The DRM node to drive from config, or the first `card*` in `/dev/dri`
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
