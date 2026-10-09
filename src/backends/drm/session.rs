//! The seat, through libseat

use std::cell::Cell;
use std::os::fd::{AsRawFd, RawFd};
use std::path::Path;
use std::rc::Rc;

use libseat::{Device, Seat, SeatEvent};
use tracing::{info, warn};

/// A seat and its active state
pub struct Session {
    /// The libseat handle
    seat: Seat,
    /// Whether the session currently holds its devices
    active: Rc<Cell<bool>>,
    /// Whether a disable has arrived and not yet been acknowledged — see [`Self::acknowledge_disable`].
    disable_pending: Rc<Cell<bool>>,
}

impl Session {
    /// Open the seat and wait for it to become active.
    ///
    /// # Errors
    ///
    /// If no session manager will grant a seat
    pub fn open() -> anyhow::Result<Self> {
        let active = Rc::new(Cell::new(false));
        let disable_pending = Rc::new(Cell::new(false));
        let listener_active = Rc::clone(&active);
        let listener_pending = Rc::clone(&disable_pending);
        let seat = Seat::open(move |_seat, event| match event {
            SeatEvent::Enable => {
                info!("session enabled");
                listener_active.set(true);
            }
            SeatEvent::Disable => {
                info!("session disabled (VT switch)");
                listener_active.set(false);
                listener_pending.set(true);
            }
        })
        .map_err(|e| anyhow::anyhow!("could not open a seat: {e}"))?;
        Ok(Self {
            seat,
            active,
            disable_pending,
        })
    }

    /// The fd to wait on for seat events
    ///
    /// # Errors
    ///
    /// If libseat will not surface its fd
    pub fn poll_fd(&mut self) -> anyhow::Result<RawFd> {
        self.seat
            .get_fd()
            .map(|fd| fd.as_raw_fd())
            .map_err(|e| anyhow::anyhow!("seat has no pollable fd: {e}"))
    }

    /// Process whatever the seat has pending, running the enable/disable listener as a side effect
    ///
    /// # Errors
    ///
    /// If libseat's own dispatch fails, which is not recoverable
    pub fn dispatch(&mut self) -> anyhow::Result<()> {
        self.seat
            .dispatch(0)
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("seat dispatch failed: {e}"))
    }

    /// Whether the session holds its devices right now
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.get()
    }

    /// Open a device the seat controls — a DRM node, an input device — as an
    /// opaque token whose fd is reachable through [`AsFd`](std::os::fd::AsFd)
    ///
    /// # Errors
    ///
    /// If the seat refuses the device: not one it controls, or the session
    /// is not active.
    pub fn open_device(&mut self, path: &Path) -> anyhow::Result<Device> {
        self.seat
            .open_device(&path)
            .map_err(|e| anyhow::anyhow!("seat refused device {}: {e}", path.display()))
    }

    /// Answer the disable that arrived in the last dispatch, if one did
    pub fn acknowledge_disable(&mut self) {
        if self.disable_pending.replace(false)
            && let Err(e) = self.seat.disable()
        {
            warn!("acknowledging the seat disable failed: {e}");
        }
    }

    /// Ask the session manager to switch to another VT
    ///
    /// # Errors
    ///
    /// If the session manager refuses — no such VT, or the seat does not do
    /// VT switching at all.
    pub fn switch_session(&mut self, vt: i32) -> anyhow::Result<()> {
        self.seat
            .switch_session(vt)
            .map_err(|e| anyhow::anyhow!("could not switch to VT {vt}: {e}"))
    }

    /// Give a device back to the seat.
    pub fn close_device(&mut self, device: Device) {
        if let Err(e) = self.seat.close_device(device) {
            warn!("closing a seat device failed: {e}");
        }
    }
}
