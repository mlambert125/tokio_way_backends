//! libinput, translated into the backend message stream.
//!
//! libinput turns raw evdev into the events a compositor actually wants —
//! acceleration applied, tap-to-click resolved, scroll sources
//! distinguished — and this wraps it so the output is the same
//! [`BackendMessage`] stream the winit and null backends produce. A
//! compositor reading the channel cannot tell which backend filled it.
//!
//! The device fds come through the same [`Session`] the DRM node did: libinput
//! asks for a device by path, and the seat hands back one this process could
//! not open itself. The fd libinput is given is a dup of the seat's, so
//! libinput owning and closing it never disturbs the seat's own token, which
//! is what actually gets returned to the seat when the device goes away.

use std::cell::RefCell;
use std::collections::HashMap;
use std::os::fd::{AsFd, AsRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::rc::Rc;

use input::event::keyboard::{KeyState as LiKeyState, KeyboardEventTrait};
use input::event::pointer::{
    Axis, ButtonState as LiButtonState, PointerEventTrait, PointerScrollEvent,
};
use input::event::touch::{TouchEventPosition, TouchEventSlot, TouchEventTrait};
use input::event::{Event, KeyboardEvent, PointerEvent, TouchEvent};
use input::{Libinput, LibinputInterface};

use crate::input::{ButtonState, KeyState, MouseButton, ScrollSource};
use crate::messages::BackendMessage;
use crate::monotonic_timestamp::MonotonicTimeStamp;

use super::session::Session;

/// The bridge libinput calls to open and close device fds, backed by the
/// seat.
struct SeatInterface {
    /// The session every device is opened through, shared with the loop.
    session: Rc<RefCell<Session>>,
    /// The seat tokens for the fds libinput holds, keyed by the raw fd of
    /// the dup libinput was given. The token, not the fd, is what the seat
    /// wants back — this is where it waits until libinput lets go.
    devices: HashMap<RawFd, libseat::Device>,
}

impl LibinputInterface for SeatInterface {
    fn open_restricted(&mut self, path: &Path, _flags: i32) -> Result<OwnedFd, i32> {
        let mut session = self.session.borrow_mut();
        let device = session.open_device(path).map_err(|_| -libc::EACCES)?;
        // A dup, so libinput's close cannot close the seat's own fd. Both
        // share one open file description, so libinput's evdev ioctls work.
        let owned = device
            .as_fd()
            .try_clone_to_owned()
            .map_err(|_| -libc::EACCES)?;
        self.devices.insert(owned.as_raw_fd(), device);
        Ok(owned)
    }

    fn close_restricted(&mut self, fd: OwnedFd) {
        if let Some(device) = self.devices.remove(&fd.as_raw_fd()) {
            self.session.borrow_mut().close_device(device);
        }
        // `fd`, the dup, closes here.
    }
}

/// A libinput context, pumped when its fd is readable.
pub struct Input {
    /// The libinput handle, which is also its own event iterator.
    libinput: Libinput,
}

impl Input {
    /// Start libinput on a seat, discovering its devices through udev.
    ///
    /// # Errors
    /// If the seat cannot be assigned — no such seat, or the session has no
    /// authority over it.
    pub fn new(session: Rc<RefCell<Session>>, seat_name: &str) -> anyhow::Result<Self> {
        let interface = SeatInterface {
            session,
            devices: HashMap::new(),
        };
        let mut libinput = Libinput::new_with_udev(interface);
        libinput
            .udev_assign_seat(seat_name)
            .map_err(|()| anyhow::anyhow!("could not assign libinput to seat {seat_name}"))?;
        Ok(Self { libinput })
    }

    /// The fd to wait on. Readable when libinput has events for
    /// [`Self::dispatch`].
    ///
    /// Stable across [`Self::suspend`]/[`Self::resume`] — it is libinput's
    /// own epoll fd, not any device's — so registering it once is enough.
    #[must_use]
    pub fn poll_fd(&self) -> RawFd {
        self.libinput.as_raw_fd()
    }

    /// Close every device, keeping the context to be resumed.
    ///
    /// For a VT switch: the moment the session is disabled the kernel
    /// revokes every evdev fd, and a revoked fd is dead for good — it does
    /// not come back with the enable the way the DRM fd does. Suspending
    /// closes them while they are worthless, and [`Self::resume`] reopens
    /// the devices through the seat afresh.
    pub fn suspend(&mut self) {
        self.libinput.suspend();
    }

    /// Reopen the devices after [`Self::suspend`], on the session's
    /// re-enable.
    ///
    /// # Errors
    /// If libinput cannot restart its udev monitoring; the devices it
    /// could not reopen individually are simply absent, as they would be
    /// after an unplug.
    pub fn resume(&mut self) -> anyhow::Result<()> {
        self.libinput
            .resume()
            .map_err(|()| anyhow::anyhow!("libinput could not resume after the VT switch"))
    }

    /// Read whatever libinput has ready and translate it.
    ///
    /// `output_size` is the physical extent touch coordinates are resolved
    /// against — libinput reports touch as a fraction of a screen. For now
    /// the first output's size; multi-output touch mapping is not done.
    ///
    /// # Errors
    /// If libinput's own dispatch fails.
    pub fn dispatch(&mut self, output_size: (i32, i32)) -> anyhow::Result<Vec<BackendMessage>> {
        self.libinput
            .dispatch()
            .map_err(|e| anyhow::anyhow!("libinput dispatch failed: {e}"))?;
        let mut messages = Vec::new();
        for event in self.libinput.by_ref() {
            translate(&event, output_size, &mut messages);
        }
        Ok(messages)
    }
}

/// Turn one libinput event into zero or more backend messages.
fn translate(event: &Event, output_size: (i32, i32), out: &mut Vec<BackendMessage>) {
    match event {
        Event::Keyboard(KeyboardEvent::Key(key)) => {
            out.push(BackendMessage::KeyInput {
                time: usec_to_timestamp(key.time_usec()),
                // libinput reports the evdev code; xkb — and the compositor's
                // keymap — want it plus eight.
                keycode: key.key() + 8,
                state: match key.key_state() {
                    LiKeyState::Pressed => KeyState::Pressed,
                    LiKeyState::Released => KeyState::Released,
                },
            });
        }
        Event::Pointer(pointer) => translate_pointer(pointer, out),
        Event::Touch(touch) => translate_touch(touch, output_size, out),
        // Devices coming and going, gestures, tablets, switches: not yet
        // mapped. Seat capabilities are reported once at startup rather than
        // tracked per device — see the run loop.
        _ => {}
    }
}

/// Pointer motion, buttons, and the several kinds of scroll.
fn translate_pointer(pointer: &PointerEvent, out: &mut Vec<BackendMessage>) {
    match pointer {
        PointerEvent::Motion(motion) => {
            out.push(BackendMessage::MouseMovedBy {
                time: usec_to_timestamp(motion.time_usec()),
                dx: motion.dx(),
                dy: motion.dy(),
            });
        }
        PointerEvent::MotionAbsolute(motion) => {
            out.push(BackendMessage::MouseMovedTo {
                time: usec_to_timestamp(motion.time_usec()),
                x: motion.absolute_x(),
                y: motion.absolute_y(),
            });
        }
        PointerEvent::Button(button) => {
            out.push(BackendMessage::MouseButton {
                time: usec_to_timestamp(button.time_usec()),
                button: MouseButton::from_evdev(button.button()),
                state: match button.button_state() {
                    LiButtonState::Pressed => ButtonState::Pressed,
                    LiButtonState::Released => ButtonState::Released,
                },
            });
        }
        PointerEvent::ScrollWheel(scroll) => {
            let time = usec_to_timestamp(scroll.time_usec());
            #[allow(clippy::cast_possible_truncation)]
            out.push(BackendMessage::MouseScroll {
                time,
                dx: scroll.scroll_value(Axis::Horizontal),
                dy: scroll.scroll_value(Axis::Vertical),
                source: ScrollSource::Wheel,
                v120_x: scroll.scroll_value_v120(Axis::Horizontal) as i32,
                v120_y: scroll.scroll_value_v120(Axis::Vertical) as i32,
            });
        }
        PointerEvent::ScrollFinger(scroll) => {
            let time = usec_to_timestamp(scroll.time_usec());
            let dx = scroll.scroll_value(Axis::Horizontal);
            let dy = scroll.scroll_value(Axis::Vertical);
            // libinput signals the end of a two-finger scroll with a zero
            // event, which is the touchpad lifting — information a client
            // cannot infer from deltas.
            if dx == 0.0 && dy == 0.0 {
                out.push(BackendMessage::MouseScrollEnd { time });
            } else {
                out.push(BackendMessage::MouseScroll {
                    time,
                    dx,
                    dy,
                    source: ScrollSource::Finger,
                    v120_x: 0,
                    v120_y: 0,
                });
            }
        }
        // Button and continuous scroll sources: treated as finger-like
        // smooth scrolling, no detents.
        PointerEvent::ScrollContinuous(scroll) => {
            out.push(BackendMessage::MouseScroll {
                time: usec_to_timestamp(scroll.time_usec()),
                dx: scroll.scroll_value(Axis::Horizontal),
                dy: scroll.scroll_value(Axis::Vertical),
                source: ScrollSource::Finger,
                v120_x: 0,
                v120_y: 0,
            });
        }
        _ => {}
    }
}

/// Touch downs, moves, ups, and cancels.
fn translate_touch(touch: &TouchEvent, output_size: (i32, i32), out: &mut Vec<BackendMessage>) {
    #[allow(clippy::cast_sign_loss)]
    let (width, height) = (output_size.0.max(1) as u32, output_size.1.max(1) as u32);
    match touch {
        TouchEvent::Down(down) => out.push(BackendMessage::TouchDown {
            time: usec_to_timestamp(down.time_usec()),
            id: seat_slot(down.seat_slot()),
            x: down.x_transformed(width),
            y: down.y_transformed(height),
        }),
        TouchEvent::Motion(motion) => out.push(BackendMessage::TouchMotion {
            time: usec_to_timestamp(motion.time_usec()),
            id: seat_slot(motion.seat_slot()),
            x: motion.x_transformed(width),
            y: motion.y_transformed(height),
        }),
        TouchEvent::Up(up) => out.push(BackendMessage::TouchUp {
            time: usec_to_timestamp(up.time_usec()),
            id: seat_slot(up.seat_slot()),
        }),
        TouchEvent::Cancel(_) => out.push(BackendMessage::TouchCancel),
        // Frame marks the end of a batch of simultaneous touch points, which
        // the compositor does not need to act on the ones above; the wildcard
        // also covers any variant a newer libinput adds.
        _ => {}
    }
}

/// A libinput seat slot as the wire's finger id. Slots start at zero and are
/// small, so the cast never loses one.
#[allow(clippy::cast_possible_wrap)]
fn seat_slot(slot: u32) -> i32 {
    slot as i32
}

/// A `CLOCK_MONOTONIC` microsecond reading — libinput's clock — as the
/// timestamp the protocol carries. Same clock, so no conversion beyond units.
fn usec_to_timestamp(usec: u64) -> MonotonicTimeStamp {
    MonotonicTimeStamp {
        tv_sec: i64::try_from(usec / 1_000_000).unwrap_or(0),
        tv_nsec: i64::try_from((usec % 1_000_000) * 1000).unwrap_or(0),
    }
}
