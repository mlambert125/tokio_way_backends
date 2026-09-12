//! The backends themselves, and the one bundle of channels every one of them
//! is driven through.

use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{oneshot, watch};
use tokio_util::sync::CancellationToken;

use crate::messages::{BackendMessage, BackendRequest};
use crate::scene_graph::SceneGraph;

pub mod drm;
pub mod null;
pub mod winit;

/// Everything a backend is wired up with, whichever backend it is.
///
/// One bundle rather than a parameter list so that every backend's entry
/// point has the same shape, and switching backends is a matter of which
/// function the bundle is handed to. What still differs is the calling
/// context, which is inherent: [`null::run_null_backend`] is a future to
/// spawn, while [`winit::run_winit_backend`] must own its thread — it blocks
/// on a window event loop — and be called where a tokio runtime handle is
/// current.
pub struct BackendChannels {
    /// Where the backend reports everything it has to say: input, outputs,
    /// presentations, answers to requests.
    pub messages: Sender<BackendMessage>,
    /// Fired exactly once, when everything a connecting client will be told
    /// has been decided — seat capabilities, outputs, dma-buf support. The
    /// compositor must not advertise its socket before this fires: a client
    /// that connects earlier enumerates the globals before dma-buf support is
    /// known, picks shm on the strength of that, and keeps it for the rest of
    /// its life.
    pub ready: oneshot::Sender<()>,
    /// The slot the compositor publishes each new frame into, holding the
    /// newest scene for every output at once.
    pub frames: watch::Receiver<SceneGraph>,
    /// Requests the compositor puts to the backend, answered over `messages`.
    pub requests: Receiver<BackendRequest>,
    /// Cancelled by either side to shut the whole compositor down; every
    /// backend stops when it fires, and a backend whose host closes fires it.
    pub cancel: CancellationToken,
}
