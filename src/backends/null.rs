//! Null backend (headless).

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
#[derive(Debug, Clone)]
pub struct VirtualOutput {
    /// Physical width in pixels
    pub width: i32,
    /// Physical height in pixels
    pub height: i32,
    /// X position of the top-left corner in global logical space
    pub x: i32,
    /// Y position of the top-left corner in global logical space
    pub y: i32,
    /// Scale factor
    pub scale: Scale,
    /// Refresh rate
    pub refresh_mhz: i32,
    /// The name the output reports
    pub name: String,
}

impl Default for VirtualOutput {
    /// A 1080p output at 60 Hz, scale 1, at the origin
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

/// The pacing state of one virtual output
struct Pacing {
    /// The output being paced.
    id: OutputId,
    /// Its refresh interval in nanoseconds, reported on every message.
    refresh_ns: u32,
    /// The same interval as the timer wants it.
    period: Duration,
    /// When this output next asks for a frame.
    deadline: Instant,
    /// True if a frame has been requested and the compositor has not yet
    awaiting_scene: bool,
    /// Serial of the scene last presented
    drawn_serial: Option<u64>,
    /// Refresh sequence
    sequence: u64,
}

/// The refresh interval in nanoseconds
fn refresh_ns_of(refresh_mhz: i32) -> u32 {
    let mhz = if refresh_mhz > 0 { refresh_mhz } else { 60_000 };
    u32::try_from(1_000_000_000_000_i64 / i64::from(mhz)).unwrap_or(16_666_666)
}

/// Describe a virtual output the way the compositor and its clients see it
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

/// A frame request for a virtual output
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

/// Answer one compositor request
async fn answer_request(
    request: BackendRequest,
    backend_sender: &tokio::sync::mpsc::Sender<BackendMessage>,
) -> bool {
    match request {
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
        BackendRequest::ImportDmabuf { token, .. } => backend_sender
            .send(BackendMessage::DmabufImportResult {
                token,
                imported: false,
            })
            .await
            .is_ok(),
        BackendRequest::CaptureOutput { token, .. } => backend_sender
            .send(BackendMessage::CaptureResult {
                token,
                capture: None,
            })
            .await
            .is_ok(),
        BackendRequest::SetPointerConfinement { .. } | BackendRequest::SetOutputSize { .. } => true,
    }
}

/// Take the newest frame and "present" every scene in it not seen before
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

/// Run the null backend in a loop until stopped
pub async fn run_null_backend(outputs: Vec<VirtualOutput>, channels: BackendChannels) {
    let BackendChannels {
        messages: backend_sender,
        ready,
        mut frames,
        mut requests,
        cancel: cancel_token,
    } = channels;
    info!("Null backend running ({} virtual output(s))", outputs.len());

    let described: Vec<Output> = outputs
        .iter()
        .enumerate()
        .map(|(index, v)| describe(v, OutputId(u32::try_from(index).unwrap_or(0) + 1)))
        .collect();
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
                    pacer.deadline = now + pacer.period;
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
