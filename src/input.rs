//! What the user did, as a backend reports it.

/// A mouse button.  This is not an enum because we have to cover extra buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MouseButton(u32);

impl MouseButton {
    /// `BTN_LEFT`.
    pub const LEFT: Self = Self(0x110);
    /// `BTN_RIGHT`.
    pub const RIGHT: Self = Self(0x111);
    /// `BTN_MIDDLE`.
    pub const MIDDLE: Self = Self(0x112);
    /// `BTN_SIDE` (browser "back" button)
    pub const BACK: Self = Self(0x113);
    /// `BTN_EXTRA` (browser "forward" button)
    pub const FORWARD: Self = Self(0x114);

    /// From a raw evdev code
    #[must_use]
    pub const fn from_evdev(code: u32) -> Self {
        Self(code)
    }

    /// Gets the inner value
    #[must_use]
    pub const fn evdev_code(self) -> u32 {
        self.0
    }
}

/// State of a mouse button
#[derive(Debug, Clone, Copy)]
pub enum ButtonState {
    /// Button is pressed down
    Pressed,
    /// Button is released
    Released,
}

/// State of a keyboard key
#[derive(Debug, Clone, Copy)]
pub enum KeyState {
    /// Key is pressed down
    Pressed,
    /// Key is released
    Released,
}

/// How the mouse pointer is held in place
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PointerConfinement {
    /// The pointer moves freely.
    #[default]
    None,
    /// The pointer stays within the output's bounds but moves within them
    Confined,
    /// The pointer is frozen where it is and only relative motion flows
    Locked,
}

/// What produced a scroll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollSource {
    /// A mouse wheel, clicking through detents
    Wheel,
    /// A touchpad or trackpoint, moving smoothly
    Finger,
}
