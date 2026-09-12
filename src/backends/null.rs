//! Null backend (headless).
//!
//! Displays nothing and captures no input. Useful for testing protocol and
//! composition logic without a display server or GPU.
//!
//! It comes in two shapes, chosen by the [`VirtualOutput`] list:
//!
//! With no virtual outputs it never asks for a frame and never presents one:
//! rendering is paced per output by the backend that owns it, and this
//! backend owns none. Clients still get their `wl_surface.frame` callbacks —
//! the compositor paces the surfaces no output is showing itself, which
//! under this backend is all of them.
//!
//! With virtual outputs it reports them and paces them like real ones: each
//! asks for a frame at its own refresh rate, and every scene composed in
//! answer is "presented" the moment it arrives — reported as a presentation
//! with nothing vouched for, because no pixel reached any screen. That is
//! what lets a compositor's whole compose-and-present path run in CI.
//!
//! Either way the frame slot is drained: holding a frame borrowed would pin
//! the client buffers it references and keep them from being released.

use std::time::Duration;

use tokio::time::Instant;
use tracing::info;

use crate::backends::BackendChannels;
use crate::dmabuf_import::DmabufImportProbeResult;
use crate::messages::{BackendMessage, BackendRequest, PresentationFlags};
use crate::monotonic_timestamp::MonotonicTimeStamp;
use crate::outputs::{
    OUTPUT_MODE_CURRENT, OUTPUT_MODE_PREFERRED, Output, OutputGeometry, OutputId, OutputMode,
    OutputSubpixel, OutputTransform, Scale,
};

/// A display the headless backend pretends to have.
///
/// Ids are assigned by position: the first output is `OutputId(1)`, the
/// second `OutputId(2)`, and so on.
#[derive(Debug, Clone)]
pub struct VirtualOutput {
    /// Physical width in pixels.
    pub width: i32,
    /// Physical height in pixels.
    pub height: i32,
    /// Position of the top-left corner in global logical space. The caller's
    /// to choose, because layout is compositor policy — two outputs at the
    /// same origin overlap exactly as two real monitors configured that way
    /// would.
    pub x: i32,
    /// Likewise.
    pub y: i32,
    /// How many physical pixels one logical pixel covers.
    pub scale: Scale,
    /// Refresh rate in millihertz, which is the rate frames are requested
    /// at. A value of zero or less falls back to 60 Hz rather than never
    /// asking, or asking as fast as the loop can spin.
    pub refresh_mhz: i32,
    /// The name the output reports, as `wl_output.name` would carry it.
    pub name: String,
}

impl Default for VirtualOutput {
    /// A 1080p output at 60 Hz, scale 1, at the origin.
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            x: 0,
            y: 0,
            scale: Scale::ONE,
            refresh_mhz: 60_000,
            name: String::from("virtual"),
        }
    }
}

/// The pacing state of one virtual output: what a vblank interrupt would be
/// tracking, kept in a struct because there is no vblank here.
struct Pacing {
    /// The output being paced.
    id: OutputId,
    /// Its refresh interval in nanoseconds, reported on every message.
    refresh_ns: u32,
    /// The same interval as the timer wants it.
    period: Duration,
    /// When this output next asks for a frame.
    deadline: Instant,
    /// Whether a request is outstanding. While it is, ticks pass silently:
    /// asking again before the compositor has composed would stack requests
    /// for the same frame.
    awaiting_scene: bool,
    /// Serial of the scene last presented, so a frame carrying old scenes
    /// for this output is not presented twice.
    drawn_serial: Option<u64>,
    /// Rises with each presentation, reported as the refresh sequence.
    sequence: u64,
}

/// The refresh interval in nanoseconds for a millihertz rate, guarding the
/// degenerate rates a config could carry.
fn refresh_ns_of(refresh_mhz: i32) -> u32 {
    let mhz = if refresh_mhz > 0 { refresh_mhz } else { 60_000 };
    u32::try_from(1_000_000_000_000_i64 / i64::from(mhz)).unwrap_or(16_666_666)
}

/// Describe a virtual output the way the compositor and its clients see it.
fn describe(virtual_output: &VirtualOutput, id: OutputId) -> Output {
    Output {
        id,
        name: virtual_output.name.clone(),
        description: format!("virtual output {}", virtual_output.name),
        geometry: OutputGeometry {
            x: virtual_output.x,
            y: virtual_output.y,
            physical_width: virtual_output.width,
            physical_height: virtual_output.height,
            subpixel: OutputSubpixel::None,
            make: String::from("virtual"),
            model: String::from("virtual"),
            transform: OutputTransform::Normal,
        },
        modes: vec![OutputMode {
            flags: OUTPUT_MODE_CURRENT | OUTPUT_MODE_PREFERRED,
            width: virtual_output.width,
            height: virtual_output.height,
            refresh_mhz: virtual_output.refresh_mhz,
        }],
        scale: virtual_output.scale,
    }
}

/// A frame request for a virtual output, predicting presentation one refresh
/// out — the same estimate the winit backend makes, for the same reason: a
/// frame composed now is shown at the next tick.
fn frame_request(id: OutputId, refresh_ns: u32) -> BackendMessage {
    let now = MonotonicTimeStamp::now();
    let nsec = now.tv_nsec + i64::from(refresh_ns);
    BackendMessage::FrameRequested {
        output: id,
        predicted_present: MonotonicTimeStamp {
            tv_sec: now.tv_sec + nsec / 1_000_000_000,
            tv_nsec: nsec % 1_000_000_000,
        },
        refresh_ns,
    }
}

/// Answer one compositor request with what a backend without a GPU can say.
/// Returns `false` once the compositor has hung up and the loop should stop.
async fn answer_request(
    request: BackendRequest,
    backend_sender: &tokio::sync::mpsc::Sender<BackendMessage>,
) -> bool {
    match request {
        // No GPU, so nothing to import onto, no formats to offer, and no
        // device to allocate on. Answered rather than ignored: the
        // compositor is waiting to hear before it decides what to advertise.
        BackendRequest::ProbeDmabuf => backend_sender
            .send(BackendMessage::DmabufSupport {
                formats: Vec::new(),
                probe: DmabufImportProbeResult::Unsupported(
                    "the null backend has no GPU to import onto".into(),
                ),
                device: None,
            })
            .await
            .is_ok(),
        // No GPU to import onto. Answered rather than dropped: a client is
        // blocked on this one.
        BackendRequest::ImportDmabuf { token, .. } => backend_sender
            .send(BackendMessage::DmabufImportResult {
                token,
                imported: false,
            })
            .await
            .is_ok(),
        // Nothing is rendered here, so there is nothing to capture — but a
        // screenshot tool is waiting on the token, so the nothing is said
        // out loud.
        BackendRequest::CaptureOutput { token, .. } => backend_sender
            .send(BackendMessage::CaptureResult {
                token,
                capture: None,
            })
            .await
            .is_ok(),
        // No pointer exists to confine, and a virtual output's size is the
        // caller's configuration, not something to negotiate at runtime.
        // Both ignored, which each request's contract allows: confinement
        // is unacknowledged by design, and no `OutputChanged` follows an
        // ignored resize, so the compositor knows nothing changed.
        BackendRequest::SetPointerConfinement { .. } | BackendRequest::SetOutputSize { .. } => {
            true
        }
    }
}

/// Take the newest frame and "present" every scene in it not seen before.
///
/// At once, because there is no screen to wait for; nothing is vouched for
/// because nothing happened — no vsync, no hardware clock, no scanout. The
/// scenes to report are read inside the watch borrow and sent after it: a
/// watch `Ref` must not be held across an await, and holding it would also
/// pin the frame's buffers. Returns `false` once the compositor has hung up.
async fn present_new_scenes(
    frames: &mut tokio::sync::watch::Receiver<crate::scene_graph::SceneGraph>,
    pacing: &mut [Pacing],
    backend_sender: &tokio::sync::mpsc::Sender<BackendMessage>,
) -> bool {
    let presented: Vec<(OutputId, u64)> = {
        let frame = frames.borrow_and_update();
        frame
            .scenes
            .iter()
            .filter_map(|scene| {
                let pacer = pacing.iter().find(|p| p.id == scene.output_id)?;
                (pacer.drawn_serial != Some(scene.serial))
                    .then_some((scene.output_id, scene.serial))
            })
            .collect()
    };
    for (id, serial) in presented {
        let Some(pacer) = pacing.iter_mut().find(|p| p.id == id) else {
            continue;
        };
        pacer.drawn_serial = Some(serial);
        pacer.awaiting_scene = false;
        pacer.sequence += 1;
        if backend_sender
            .send(BackendMessage::FramePresented {
                output: id,
                time: MonotonicTimeStamp::now(),
                refresh_ns: pacer.refresh_ns,
                sequence: pacer.sequence,
                flags: PresentationFlags::default(),
            })
            .await
            .is_err()
        {
            return false;
        }
    }
    true
}

/// Run the null backend in a loop until stopped.
///
/// A future to spawn, unlike the winit backend, which needs a thread of its
/// own. Readiness fires as soon as the outputs are reported: with no display
/// and no GPU there is nothing else to wait for.
///
/// `outputs` is the list of displays to pretend to have; empty is the
/// original null backend, which owns no output and paces nothing.
pub async fn run_null_backend(outputs: Vec<VirtualOutput>, channels: BackendChannels) {
    let BackendChannels {
        messages: backend_sender,
        ready,
        mut frames,
        mut requests,
        cancel: cancel_token,
    } = channels;
    info!(
        "Null backend running ({} virtual output(s))",
        outputs.len()
    );

    let described: Vec<Output> = outputs
        .iter()
        .enumerate()
        .map(|(index, v)| describe(v, OutputId(u32::try_from(index).unwrap_or(0) + 1)))
        .collect();
    // Reported before `ready`, like any backend: the outputs are part of what
    // a connecting client will be told.
    if !described.is_empty()
        && backend_sender
            .send(BackendMessage::OutputInfo {
                outputs: described.clone(),
            })
            .await
            .is_err()
    {
        return;
    }
    let _ = ready.send(());

    let mut pacing: Vec<Pacing> = Vec::with_capacity(outputs.len());
    for (output, described) in outputs.iter().zip(&described) {
        let refresh_ns = refresh_ns_of(output.refresh_mhz);
        let period = Duration::from_nanos(u64::from(refresh_ns));
        // The first request goes out at once — the compositor cannot compose
        // for an output that has never asked.
        if backend_sender
            .send(frame_request(described.id, refresh_ns))
            .await
            .is_err()
        {
            return;
        }
        pacing.push(Pacing {
            id: described.id,
            refresh_ns,
            period,
            deadline: Instant::now() + period,
            awaiting_scene: true,
            drawn_serial: None,
            sequence: 0,
        });
    }

    loop {
        // The next tick is the earliest deadline; with no outputs there is no
        // tick, and the guard keeps that select arm out entirely.
        let next_deadline = pacing.iter().map(|p| p.deadline).min();
        tokio::select! {
            () = cancel_token.cancelled() => break,
            request = requests.recv() => {
                let Some(request) = request else { break };
                if !answer_request(request, &backend_sender).await {
                    break;
                }
            }
            changed = frames.changed() => {
                if changed.is_err()
                    || !present_new_scenes(&mut frames, &mut pacing, &backend_sender).await
                {
                    break;
                }
            }
            () = tokio::time::sleep_until(next_deadline.unwrap_or_else(Instant::now)),
                if next_deadline.is_some() =>
            {
                let now = Instant::now();
                for pacer in &mut pacing {
                    if pacer.deadline > now {
                        continue;
                    }
                    // From now rather than from the missed deadline, so a
                    // stall is a stall and not a burst of catch-up frames.
                    pacer.deadline = now + pacer.period;
                    // A request already out means the compositor has not
                    // composed yet; this tick passes and the next one asks —
                    // the same one-frame-in-flight bound a real vblank gives.
                    if !pacer.awaiting_scene {
                        pacer.awaiting_scene = true;
                        if backend_sender
                            .send(frame_request(pacer.id, pacer.refresh_ns))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
        }
    }
    info!("Null backend shutting down");
    drop(backend_sender);
}

#[cfg(test)]
mod tests {
    //! Tests for the headless backend.

    use super::*;
    use crate::scene_graph::{Scene, SceneGraph};
    use std::sync::Arc;
    use tokio::sync::mpsc::{Receiver, Sender, channel};
    use tokio::sync::{oneshot, watch};
    use tokio_util::sync::CancellationToken;

    /// The wiring a test drives the backend through, and the ends it keeps.
    fn wired() -> (
        BackendChannels,
        Receiver<BackendMessage>,
        watch::Sender<SceneGraph>,
        Sender<BackendRequest>,
        oneshot::Receiver<()>,
        CancellationToken,
    ) {
        let (backend_tx, backend_rx) = channel(8);
        let (frames_tx, frames_rx) = watch::channel(SceneGraph::default());
        let (requests_tx, requests_rx) = channel(8);
        let (ready_tx, ready_rx) = oneshot::channel();
        let cancel = CancellationToken::new();
        let channels = BackendChannels {
            messages: backend_tx,
            ready: ready_tx,
            frames: frames_rx,
            requests: requests_rx,
            cancel: cancel.clone(),
        };
        (channels, backend_rx, frames_tx, requests_tx, ready_rx, cancel)
    }

    #[tokio::test]
    async fn a_backend_with_no_outputs_presents_nothing() {
        let (channels, mut backend_rx, frames_tx, _requests_tx, ready_rx, cancel) = wired();
        let backend = tokio::spawn(run_null_backend(Vec::new(), channels));

        // Nothing to wait for, so readiness is immediate.
        ready_rx.await.expect("the null backend should report ready");

        drop(frames_tx.send_replace(SceneGraph::default()));

        // A presentation names the output it happened on, and this backend has
        // none — it never asks for a frame, so nothing is ever composed for
        // it. The clients' frame callbacks are the compositor's job here,
        // fired against the surfaces no output is showing. Claiming a
        // presentation would fire them against an output that does not exist.
        let quiet =
            tokio::time::timeout(std::time::Duration::from_millis(50), backend_rx.recv()).await;
        assert!(quiet.is_err(), "the backend should have nothing to report");

        cancel.cancel();
        backend.await.unwrap();
    }

    #[tokio::test]
    async fn there_is_no_dmabuf_import_without_a_gpu() {
        let (channels, mut backend_rx, _frames_tx, requests_tx, _ready_rx, cancel) = wired();
        let backend = tokio::spawn(run_null_backend(Vec::new(), channels));

        requests_tx.send(BackendRequest::ProbeDmabuf).await.unwrap();

        // Answered rather than ignored: the compositor decides what to
        // advertise on the strength of this, and would wait forever for a
        // backend that stayed quiet because it had nothing to say.
        let message = backend_rx.recv().await.expect("backend went quiet");
        let BackendMessage::DmabufSupport {
            formats,
            probe,
            device,
        } = message
        else {
            panic!("expected a dma-buf answer, got {message:?}");
        };
        assert!(formats.is_empty());
        assert!(matches!(probe, DmabufImportProbeResult::Unsupported(_)));
        assert!(device.is_none(), "no GPU means no device to allocate on");

        cancel.cancel();
        backend.await.unwrap();
    }

    #[tokio::test]
    async fn a_capture_request_is_answered_with_nothing() {
        let (channels, mut backend_rx, _frames_tx, requests_tx, _ready_rx, cancel) = wired();
        let backend = tokio::spawn(run_null_backend(Vec::new(), channels));

        requests_tx
            .send(BackendRequest::CaptureOutput {
                token: 7,
                output: crate::outputs::OutputId(1),
                overlay_cursor: true,
            })
            .await
            .unwrap();

        // Nothing is rendered here, but the asker is waiting on the token,
        // so the emptiness is reported rather than left hanging.
        let message = backend_rx.recv().await.expect("backend went quiet");
        let BackendMessage::CaptureResult { token, capture } = message else {
            panic!("expected a capture answer, got {message:?}");
        };
        assert_eq!(token, 7);
        assert!(capture.is_none());

        cancel.cancel();
        backend.await.unwrap();
    }

    #[tokio::test]
    async fn a_virtual_output_paces_the_whole_frame_loop() {
        let (channels, mut backend_rx, frames_tx, _requests_tx, ready_rx, cancel) = wired();
        // A fast refresh, so the test's second request arrives without a
        // human-noticeable wait.
        let output = VirtualOutput {
            refresh_mhz: 240_000,
            ..VirtualOutput::default()
        };
        let backend = tokio::spawn(run_null_backend(vec![output], channels));

        // The output is reported before ready fires: it is part of what a
        // connecting client will be told.
        let message = backend_rx.recv().await.expect("backend went quiet");
        let BackendMessage::OutputInfo { outputs } = message else {
            panic!("expected the outputs first, got {message:?}");
        };
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].id, OutputId(1));
        ready_rx.await.expect("the backend should report ready");

        // It asks for a frame — which is what makes composition testable
        // headlessly at all.
        let message = backend_rx.recv().await.expect("backend went quiet");
        let BackendMessage::FrameRequested { output: id, .. } = message else {
            panic!("expected a frame request, got {message:?}");
        };
        assert_eq!(id, OutputId(1));

        // Compose in answer, and the scene is presented...
        frames_tx.send_replace(SceneGraph {
            scenes: vec![Arc::new(Scene {
                output_id: OutputId(1),
                background: 0xff00_0000,
                serial: 1,
                elements: Vec::new(),
                scale: Scale::ONE,
                damage_from: None,
                damage: Vec::new(),
            })],
            cursor: crate::scene_graph::Cursor::default(),
        });
        let message = backend_rx.recv().await.expect("backend went quiet");
        let BackendMessage::FramePresented { output: id, sequence, .. } = message else {
            panic!("expected a presentation, got {message:?}");
        };
        assert_eq!(id, OutputId(1));
        assert_eq!(sequence, 1);

        // ...and the next tick asks again: the loop turns.
        let message = backend_rx.recv().await.expect("backend went quiet");
        assert!(
            matches!(message, BackendMessage::FrameRequested { .. }),
            "expected the next frame request, got {message:?}"
        );

        cancel.cancel();
        backend.await.unwrap();
    }
}
