# tokio_way_backends

A tokio-based Wayland backends library: the display/input half of a Wayland
compositor, factored out behind a message-passing boundary.

A backend owns the display and the input devices. The compositor talks to it
over channels and nothing else — it publishes scenes to draw, and receives
input events, output descriptions, presentation feedback, and answers to its
questions. The crate provides the shared vocabulary (scene graph, messages,
output and buffer types), a GLES 3.0 renderer with dma-buf import, shm pool
mapping hardened against client truncation, and two backends:

- **winit** — runs the compositor as a window on a host compositor.
  Development mode: your compositor is a nested window on your desktop.
- **null** — headless. No display, no input; for exercising protocol logic
  in tests and CI. Configurable virtual outputs make the whole
  compose-and-present loop testable without a GPU.
- **drm** — bare hardware: DRM/KMS scanout, libinput, and a libseat session,
  for a compositor running as a login session with no host beneath it. It
  links system libraries the other backends do not (libdrm, gbm, libinput,
  libudev, libseat, EGL), so the crate builds only where those are present —
  see `flake.nix` for the build inputs. Legacy modesetting with page-flip
  pacing; atomic modeset, hardware cursor planes, and multi-GPU are not yet
  done.

## Wiring one up

Every backend is driven through the same bundle of channels:

```rust,no_run
use tokio_way_backends::backends::{BackendChannels, winit::run_winit_backend};
use tokio_way_backends::scene_graph::SceneGraph;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (messages_tx, mut messages_rx) = tokio::sync::mpsc::channel(64);
    let (frames_tx, frames_rx) = tokio::sync::watch::channel(SceneGraph::default());
    let (requests_tx, requests_rx) = tokio::sync::mpsc::channel(8);
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let cancel = tokio_util::sync::CancellationToken::new();

    let channels = BackendChannels {
        messages: messages_tx,
        ready: ready_tx,
        frames: frames_rx,
        requests: requests_rx,
        cancel: cancel.clone(),
    };

    // The winit backend blocks on its window event loop, so it owns a thread;
    // it spawns helper tasks onto the runtime, so enter a handle first.
    // (The null backend is just a future: `tokio::spawn(run_null_backend(channels))`.)
    let runtime = tokio::runtime::Handle::current();
    std::thread::spawn(move || {
        let _guard = runtime.enter();
        run_winit_backend("my-compositor", channels)
    });

    // Everything a connecting client will be told — outputs, seat,
    // dma-buf support — is decided when this fires. Only advertise your
    // wayland socket after it.
    ready_rx.await?;

    // The compositor loop: read input and frame requests off `messages_rx`,
    // compose a scene when an output asks for one and publish it with
    // `frames_tx.send_replace(...)`, put questions on `requests_tx`, and
    // `cancel.cancel()` to shut down.
    while let Some(message) = messages_rx.recv().await {
        todo!("compositor logic: {message:?}");
    }
    Ok(())
}
```

The contract, in brief:

- **Pull, not push.** Compose a scene for an output only when its
  `FrameRequested` arrives; the backend asks again after presenting. One
  frame in flight per output.
- **Ready gates the socket.** Don't let clients connect before `ready`
  fires (see above).
- **Zero-copy shm.** Textures borrow the client's mapped pool rather than
  copying it; hold `wl_buffer.release` back until the frame referencing the
  buffer has been dropped.

## SIGBUS

Mapping a client's shm pool installs a process-wide `SIGBUS` handler (once),
because a client can shrink a pool's file after it is mapped. The handler
patches faulting pages of registered pools to zeroes and forwards every other
`SIGBUS` to whatever handler it displaced, so crash reporters and Rust's own
stack-overflow handler keep working. See the crate docs for details and for
`shm::install_sigbus_guard` if you need to control installation timing.

## License

MIT OR Apache-2.0
