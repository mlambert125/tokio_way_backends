#![warn(clippy::pedantic)]
#![warn(missing_docs)]

//! Tokio-Based Wayland Backends Implementation
//!
//! This is an opinionated `tokio`+`tokio_utils` wayland backends implementation
//!
//! # Wiring
//!
//! A backend runs as its own task and shares no state with the compositor:
//! everything passes through the channels bundled in
//! [`backends::BackendChannels`]. The compositor publishes frames into a
//! watch slot, puts questions on a request channel, and reads everything the
//! backend has to say — input, outputs, presentations, answers — off one
//! message channel. [`backends::null::run_null_backend`] is a future to
//! spawn; [`backends::winit::run_winit_backend`] blocks on a window event
//! loop, so it needs a thread of its own with a tokio runtime handle entered.
//!
//! # Startup sequencing
//!
//! The `ready` channel fires once everything a connecting client will be
//! told has been decided: seat capabilities, outputs, and dma-buf support.
//! Do not advertise the compositor's socket before it fires. A client that
//! connects earlier enumerates the globals before dma-buf support is known,
//! settles for shm on the strength of that, and keeps it for the rest of its
//! life.
//!
//! # Frame pacing
//!
//! Composition is pull, not push: the compositor composes a scene for an
//! output only in answer to that output's
//! [`FrameRequested`](messages::BackendMessage::FrameRequested), and the
//! backend asks again only after presenting what it got. That bounds every
//! output to one frame in flight, and lets each output run at its own rate.
//! On the DRM backend that answer is the page flip completing; on winit it is
//! the host's `RedrawRequested`; the compositor writes to neither directly.
//!
//! # Backends
//!
//! Three, all built: [`backends::winit`] (nested in a host compositor),
//! [`backends::null`] (headless, for tests and CI), and [`backends::drm`]
//! (bare hardware — DRM/KMS scanout, libinput, a libseat session). The DRM
//! backend links system libraries the others do not (libdrm, gbm, libinput,
//! libudev, libseat, EGL), so the crate builds only where those are present;
//! see `flake.nix`.
//!
//! # `SIGBUS` and shm pools
//!
//! Mapping a client's shm pool ([`shm::PoolMapping`]) installs a process-wide
//! `SIGBUS` handler the first time it happens: a client can shrink the file
//! behind a pool the compositor has already mapped, and the handler is what
//! turns the resulting fault into a blacked-out page instead of a dead
//! compositor. The handler it displaces is saved, and every `SIGBUS` outside
//! a registered pool mapping is forwarded to it, so a crash reporter — or the
//! stack-overflow handler Rust's own runtime registers for this signal —
//! keeps working. If the handler in place at first-map time might not be the
//! one you mean to chain to, call [`shm::install_sigbus_guard`] yourself at
//! the moment that is.

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
