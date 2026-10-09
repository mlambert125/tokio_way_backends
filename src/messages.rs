//! What passes between the compositor and a backend through channels

use crate::{
    dma::{DmabufFormat, DmabufImage, RenderNode},
    dmabuf_import::DmabufImportProbeResult,
    input::{ButtonState, KeyState, MouseButton, PointerConfinement, ScrollSource},
    monotonic_timestamp::MonotonicTimeStamp,
    outputs::{Output, OutputId},
};
use std::sync::Arc;

/// One captured output frame in memory pixels
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    /// Width in physical pixels
    pub width: u32,
    /// Height in physical pixels
    pub height: u32,
    /// The pixels, top row first, `[R, G, B, A]` per pixel, premultiplied — `width * height * 4` bytes
    pub pixels: Vec<u8>,
}

/// Presentation feedback sync promises
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PresentationFlags {
    /// Presentation was synchronised to the display's vertical retrace
    pub vsync: bool,
    /// The timestamp came from a hardware clock
    pub hw_clock: bool,
    /// The hardware signalled completion, so the timestamp is when the frame became visible
    pub hw_completion: bool,
    /// The client's buffer was scanned out untouched, with no compositing copy
    pub zero_copy: bool,
}

/// A message from the backend to the compositor
#[derive(Debug)]
pub enum BackendMessage {
    /// A message reporting seat capabilities
    SeatCapabilities {
        /// A pointer is present and available
        pointer: bool,
        /// A keyboard is present and available
        keyboard: bool,
        /// A touchscreen is present and available
        touch: bool,
    },
    /// A message reporting output info
    OutputInfo {
        /// A vector of all available outputs
        outputs: Vec<Output>,
    },
    /// A message that the backend host has closed.  Only applicable to winit backend
    Closed,
    /// The backend is ready to show another frame on this output
    FrameRequested {
        /// Which output can take a frame
        output: OutputId,
        /// When a frame composed in answer is expected to reach the screen, on the
        /// same clock as [`BackendMessage::FramePresented`]. Best-effort.
        predicted_present: MonotonicTimeStamp,
        /// The display's nominal refresh interval in nanoseconds, or 0 if unknown
        refresh_ns: u32,
    },
    /// A scene has reached the screen on this output
    FramePresented {
        /// Which output presented
        output: OutputId,
        /// When it reached the screen, read at that moment by the backend
        time: MonotonicTimeStamp,
        /// The display's nominal refresh interval in nanoseconds, or 0 if unknown
        refresh_ns: u32,
        /// A counter that rises by one each time the output refreshes, or 0 if unknown
        sequence: u64,
        /// What the backend can vouch for about this presentation
        flags: PresentationFlags,
    },
    /// An output has appeared after startup: a monitor plugged in, or a window opened on a host
    OutputAdded {
        /// The new output, described in full
        output: Output,
    },
    /// An output has gone away: unplugged, or its host window closed
    OutputRemoved {
        /// Which output went
        output: OutputId,
    },
    /// Something about an existing output changed. The whole description is re-sent
    /// under the same id rather than naming which field moved
    OutputChanged {
        /// The output's new description, under its existing id
        output: Output,
    },
    /// A key has changed state (pressed/released). Only which physical key moved, and when —
    /// modifier masks are the compositor's to derive from the keymap it owns
    KeyInput {
        /// When the key changed, read where the event was produced
        time: MonotonicTimeStamp,
        /// The keycode: the evdev code plus eight, which is what xkb expects
        keycode: u32,
        /// The state of the key
        state: KeyState,
    },
    /// The mouse has been moved (absolute)
    MouseMovedTo {
        /// When it moved — see [`BackendMessage::KeyInput::time`]
        time: MonotonicTimeStamp,
        /// The new x coordinate
        x: f64,
        /// The new y coordinate
        y: f64,
    },
    /// The mouse has moved by this much. (relative)
    MouseMovedBy {
        /// When it moved — see [`BackendMessage::KeyInput::time`]
        time: MonotonicTimeStamp,
        /// The delta of x
        dx: f64,
        /// The delta of y
        dy: f64,
    },
    /// A mouse button has changed its state
    MouseButton {
        /// When it changed — see [`BackendMessage::KeyInput::time`]
        time: MonotonicTimeStamp,
        /// Which button changed
        button: MouseButton,
        /// The state of the button
        state: ButtonState,
    },
    /// A single finger has touched the screen
    TouchDown {
        /// When it landed — see [`BackendMessage::KeyInput::time`]
        time: MonotonicTimeStamp,
        /// Which finger — unique among those down, but says nothing about which one
        id: i32,
        /// X coordinate for where it landed, in global compositor coordinates
        x: f64,
        /// Y coordinate for where it landed, in global compositor coordinates
        y: f64,
    },
    /// A finger already down has moved
    TouchMotion {
        /// When it moved — see [`BackendMessage::KeyInput::time`]
        time: MonotonicTimeStamp,
        /// Which finger
        id: i32,
        /// Its new position, in global compositor coordinates
        x: f64,
        /// Likewise
        y: f64,
    },
    /// A finger has been lifted
    TouchUp {
        /// When it lifted — see [`BackendMessage::KeyInput::time`]
        time: MonotonicTimeStamp,
        /// Which finger
        id: i32,
    },
    /// The touch sequence has been taken over by something else. No timestamp: `wl_touch.cancel` carries none
    TouchCancel,
    /// The pointer's scroll axes have moved
    MouseScroll {
        /// When it scrolled — see [`BackendMessage::KeyInput::time`]
        time: MonotonicTimeStamp,
        /// The change in the x axis of the scroll
        dx: f64,
        /// The change in the y axis of the scroll
        dy: f64,
        /// What did the scrolling, which decides how a client should treat it
        source: ScrollSource,
        /// Wheel detents on x axis, in 120ths of a click
        v120_x: i32,
        /// Wheel detents on y axis, in 120ths of a click
        v120_y: i32,
    },
    /// A continuous scroll has finished — the fingers have left the touchpad
    MouseScrollEnd {
        /// When they left — see [`BackendMessage::KeyInput::time`]
        time: MonotonicTimeStamp,
    },
    /// What this backend can do with dma-bufs, in answer to [`BackendRequest::ProbeDmabuf`]
    DmabufSupport {
        /// Formats and modifiers that can be imported
        formats: Vec<DmabufFormat>,
        /// What came of actually trying it
        probe: DmabufImportProbeResult,
        /// The DRM device imports land on, if the backend could name it
        device: Option<RenderNode>,
    },
    /// What came of a [`BackendRequest::CaptureOutput`]
    CaptureResult {
        /// The token from the request
        token: u64,
        /// The pixels, or `None` when there was nothing to capture
        capture: Option<CapturedFrame>,
    },
    /// A shader effect failed to compile on this backend's driver, reported once per snippet
    EffectCompileFailed {
        /// The [`ShaderEffect::name`](crate::scene_graph::ShaderEffect::name) of the snippet that failed
        effect: String,
        /// The driver's compile log
        log: String,
    },
    /// What came of a [`BackendRequest::ImportDmabuf`]
    DmabufImportResult {
        /// The token from the request
        token: u64,
        /// Whether the driver took the buffer
        imported: bool,
    },
    /// The host window has gained focus (winit only)
    FocusIn,
    /// The host window has lost focus (winit only)
    FocusOut,
}

/// A request from the compositor to the backend
#[derive(Debug)]
pub enum BackendRequest {
    /// Report which dma-buf formats can be imported, having checked that importing
    /// actually works. Answered with [`BackendMessage::DmabufSupport`].
    ProbeDmabuf,
    /// Ask the backend to resize an output, in physical pixels. Best-effort, and answered
    /// indirectly as [`BackendMessage::OutputChanged`] if it takes effect at all.
    SetOutputSize {
        /// Which output
        output: OutputId,
        /// Physical width in pixels
        width: i32,
        /// Physical height in pixels
        height: i32,
    },
    /// Ask for the pixels an output is showing, answered with [`BackendMessage::CaptureResult`]
    /// carrying the same token. Always answered, even with `None`.
    CaptureOutput {
        /// Identifies this capture, never reused
        token: u64,
        /// Which output to capture
        output: OutputId,
        /// Whether to composite the cursor into the capture
        overlay_cursor: bool,
    },
    /// Ask the backend to confine or lock the pointer. Best-effort and unacknowledged;
    /// relative motion flows regardless of confinement.
    SetPointerConfinement {
        /// The confinement to apply, replacing whatever was in force
        mode: PointerConfinement,
    },
    /// Try importing one client buffer and report whether it took, answered with
    /// [`BackendMessage::DmabufImportResult`] carrying the same token.
    ImportDmabuf {
        /// Identifies this import, never reused
        token: u64,
        /// The buffer to try
        image: Arc<DmabufImage>,
    },
}
