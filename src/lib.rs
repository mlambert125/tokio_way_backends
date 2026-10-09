#![warn(clippy::pedantic)]
#![warn(missing_docs)]

//! Tokio-based Wayland backends: the display and input half of a compositor
//!
//! A backend owns the display and the input devices and shares no state with the
//! compositor — everything passes through the channels bundled in
//! [`backends::BackendChannels`]. Three are provided: [`backends::winit`] (nested in a
//! host compositor), [`backends::null`] (headless, for tests and CI), and
//! [`backends::drm`] (DRM/KMS scanout, libinput, a libseat session).
//!
//! Three rules govern the boundary:
//!
//! - **Pull, not push.** Compose a scene for an output only in answer to that output's
//!   [`FrameRequested`](messages::BackendMessage::FrameRequested). One frame in flight per output.
//! - **Ready gates the socket.** The `ready` channel fires once seat capabilities,
//!   outputs and dma-buf support are decided. Do not advertise the compositor's socket
//!   before it; a client that connects earlier settles for shm and keeps it for life.
//! - **Zero-copy shm.** Textures borrow the client's mapped pool, so hold
//!   `wl_buffer.release` until the frame that referenced the buffer is dropped.
//!
//! Mapping a client's shm pool ([`shm::PoolMapping`]) installs a process-wide `SIGBUS`
//! handler the first time it happens, chaining to whatever it displaced. Call
//! [`shm::install_sigbus_guard`] yourself to control when that happens.

/// Outputs (monitors)
pub mod outputs;

/// The `SceneGraph` created by the compositor and passed to the backend
/// containing all information on how to draw the screens
pub mod scene_graph;

/// DMA buffers allocated on the GPU
pub mod dma;

/// DMA buffer importing
pub mod dmabuf_import;

/// SHM software buffers allocated in RAM
pub mod shm;

/// `CLOCK_MONOTONIC` instants
pub mod monotonic_timestamp;

/// Backend implementations
pub mod backends;

/// Messaging to send over thread channels to communicate between compositor and backend
pub mod messages;

/// Mouse/Keyboard Input
pub mod input;

/// OpenGL Renderer used by backends
pub mod gl_renderer;

#[cfg(test)]
mod tests;
