//! The backends themselves, and the one bundle of channels every one of them is driven through.

use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{oneshot, watch};
use tokio_util::sync::CancellationToken;

use crate::messages::{BackendMessage, BackendRequest};
use crate::scene_graph::SceneGraph;

pub mod drm;
pub mod null;
pub mod winit;

/// Everything a backend is wired up with, whichever backend it is
pub struct BackendChannels {
    /// Where the backend reports everything it has to say: input, outputs, presentations, answers to requests.
    pub messages: Sender<BackendMessage>,
    /// Fired exactly once, when everything is ready
    pub ready: oneshot::Sender<()>,
    /// The slot the compositor publishes each new frame into, holding the newest scene for every output at once.
    pub frames: watch::Receiver<SceneGraph>,
    /// Requests the compositor puts to the backend, answered over `messages`.
    pub requests: Receiver<BackendRequest>,
    /// Cancelled by either side to shut the whole compositor down
    pub cancel: CancellationToken,
}
