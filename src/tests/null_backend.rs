use crate::backends::BackendChannels;
use crate::backends::null::*;
use crate::dmabuf_import::DmabufImportProbeResult;
use crate::messages::{BackendMessage, BackendRequest};
use crate::outputs::{OutputId, Scale};
use crate::scene_graph::{Scene, SceneGraph};
use std::sync::Arc;
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tokio::sync::{oneshot, watch};
use tokio_util::sync::CancellationToken;

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
    (
        channels,
        backend_rx,
        frames_tx,
        requests_tx,
        ready_rx,
        cancel,
    )
}

#[tokio::test]
async fn a_backend_with_no_outputs_presents_nothing() {
    let (channels, mut backend_rx, frames_tx, _requests_tx, ready_rx, cancel) = wired();
    let backend = tokio::spawn(run_null_backend(Vec::new(), channels));

    ready_rx
        .await
        .expect("the null backend should report ready");

    drop(frames_tx.send_replace(SceneGraph::default()));

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
    let output = VirtualOutput {
        refresh_mhz: 240_000,
        ..VirtualOutput::default()
    };
    let backend = tokio::spawn(run_null_backend(vec![output], channels));

    let message = backend_rx.recv().await.expect("backend went quiet");
    let BackendMessage::OutputInfo { outputs } = message else {
        panic!("expected the outputs first, got {message:?}");
    };
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].id, OutputId(1));
    ready_rx.await.expect("the backend should report ready");

    let message = backend_rx.recv().await.expect("backend went quiet");
    let BackendMessage::FrameRequested { output: id, .. } = message else {
        panic!("expected a frame request, got {message:?}");
    };
    assert_eq!(id, OutputId(1));

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
    let BackendMessage::FramePresented {
        output: id,
        sequence,
        ..
    } = message
    else {
        panic!("expected a presentation, got {message:?}");
    };
    assert_eq!(id, OutputId(1));
    assert_eq!(sequence, 1);

    let message = backend_rx.recv().await.expect("backend went quiet");
    assert!(
        matches!(message, BackendMessage::FrameRequested { .. }),
        "expected the next frame request, got {message:?}"
    );

    cancel.cancel();
    backend.await.unwrap();
}
