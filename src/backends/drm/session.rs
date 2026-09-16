//! The seat, through libseat.
//!
//! A compositor on bare hardware needs a master DRM fd and open input
//! devices, and neither is given to an unprivileged process for free. libseat
//! is the broker: it talks to whatever session manager is present — seatd, or
//! logind — and hands back device fds the process could not open itself, and
//! it is the channel through which VT switching arrives, as an enable that
//! goes away and comes back.
//!
//! The device fds are the whole point of going through a seat. When the
//! session is disabled — the user switched to another VT — the DRM master is
//! dropped out from under us and every device fd stops working until the
//! enable returns; the loop watches [`Session::is_active`] and neither draws
//! nor reads while it is down.

use std::cell::Cell;
use std::os::fd::{AsRawFd, RawFd};
use std::path::Path;
use std::rc::Rc;

use libseat::{Device, Seat, SeatEvent};
use tracing::{info, warn};

/// A seat and its active state.
///
/// Not `Send`: libseat's `Seat` owns C state that must be touched from the
/// one thread that opened it, which is where the whole DRM backend runs.
pub struct Session {
    /// The libseat handle. Device opens, VT switches and dispatch all go
    /// through it.
    seat: Seat,
    /// Whether the session currently holds its devices. Flipped by the
    /// listener libseat calls from inside [`Self::dispatch`]; read by the
    /// loop to decide whether it may draw and read input. Shared through an
    /// `Rc<Cell>` because the listener and the loop are the same thread.
    active: Rc<Cell<bool>>,
}

impl Session {
    /// Open the seat and wait for it to become active.
    ///
    /// A freshly opened seat is not yet enabled; libseat signals the first
    /// enable through the listener, so the caller dispatches until
    /// [`Self::is_active`] turns true before touching any device.
    ///
    /// # Errors
    /// If no session manager will grant a seat — not on a seat at all, or
    /// neither seatd nor logind is reachable.
    pub fn open() -> anyhow::Result<Self> {
        let active = Rc::new(Cell::new(false));
        let listener_active = Rc::clone(&active);
        let seat = Seat::open(move |_seat, event| match event {
            SeatEvent::Enable => {
                info!("session enabled");
                listener_active.set(true);
            }
            SeatEvent::Disable => {
                info!("session disabled (VT switch)");
                listener_active.set(false);
            }
        })
        .map_err(|e| anyhow::anyhow!("could not open a seat: {e}"))?;
        Ok(Self { seat, active })
    }

    /// The fd to wait on for seat events. Readable when libseat has an
    /// enable, disable or device signal pending for [`Self::dispatch`].
    ///
    /// Raw rather than owned: the fd belongs to libseat and must not be
    /// closed here, only watched. Registered read-only with tokio.
    ///
    /// # Errors
    /// If libseat will not surface its fd.
    pub fn poll_fd(&mut self) -> anyhow::Result<RawFd> {
        self.seat
            .get_fd()
            .map(|fd| fd.as_raw_fd())
            .map_err(|e| anyhow::anyhow!("seat has no pollable fd: {e}"))
    }

    /// Process whatever the seat has pending, running the enable/disable
    /// listener as a side effect.
    ///
    /// # Errors
    /// If libseat's own dispatch fails, which is not recoverable.
    pub fn dispatch(&mut self) -> anyhow::Result<()> {
        self.seat
            .dispatch(0)
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("seat dispatch failed: {e}"))
    }

    /// Whether the session holds its devices right now.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.get()
    }

    /// Open a device the seat controls — a DRM node, an input device — as an
    /// opaque token whose fd is reachable through [`AsFd`](std::os::fd::AsFd)
    /// and which is handed back to [`Self::close_device`] when done.
    ///
    /// The fd is live only while the session is active; on a disable the
    /// kernel revokes it, and it starts working again on the next enable
    /// without being reopened — which is why the token, not the fd, is what
    /// is kept.
    ///
    /// # Errors
    /// If the seat refuses the device: not one it controls, or the session
    /// is not active.
    pub fn open_device(&mut self, path: &Path) -> anyhow::Result<Device> {
        self.seat
            .open_device(&path)
            .map_err(|e| anyhow::anyhow!("seat refused device {}: {e}", path.display()))
    }

    /// Ask the session manager to switch to another VT.
    ///
    /// Only asks: the switch itself arrives, if it is granted, as a disable
    /// through the listener, the same as a switch this process never asked
    /// for. Nothing is torn down here — the disable is where that happens.
    ///
    /// # Errors
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
