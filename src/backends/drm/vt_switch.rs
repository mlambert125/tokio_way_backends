//! Ctrl+Alt+F1..F12, caught before the compositor ever sees them.
use std::collections::HashSet;

use crate::input::KeyState;
use crate::messages::BackendMessage;
use crate::monotonic_timestamp::MonotonicTimeStamp;

/// Evdev codes for the keys the chord is made of.
pub(crate) const KEY_LEFTCTRL: u32 = 29;
pub(crate) const KEY_LEFTALT: u32 = 56;
pub(crate) const KEY_RIGHTCTRL: u32 = 97;
pub(crate) const KEY_RIGHTALT: u32 = 100;

/// What to do with one key event, decided by [`VtKeys::on_key`].
pub enum KeyAction {
    /// Not of concern, pass it on
    Forward,
    /// Part of a chord the backend already acted on
    Swallow,
    /// The chord: ask the seat for this VT, and swallow the key
    SwitchVt(i32),
}

/// The held keys, watched for VT-switch chords.
#[derive(Default)]
pub struct VtKeys {
    /// Every key currently down, as evdev codes
    pressed: HashSet<u32>,
    /// Function keys whose press was consumed for a switch
    swallowed: HashSet<u32>,
}

impl VtKeys {
    /// Track one key event and say what should happen to it
    pub fn on_key(&mut self, message: &BackendMessage) -> KeyAction {
        let &BackendMessage::KeyInput { keycode, state, .. } = message else {
            return KeyAction::Forward;
        };
        let evdev = keycode.saturating_sub(8);
        match state {
            KeyState::Pressed => {
                self.pressed.insert(evdev);
                if let Some(vt) = vt_of(evdev)
                    && self.ctrl_and_alt_held()
                {
                    self.swallowed.insert(evdev);
                    return KeyAction::SwitchVt(vt);
                }
                KeyAction::Forward
            }
            KeyState::Released => {
                self.pressed.remove(&evdev);
                if self.swallowed.remove(&evdev) {
                    KeyAction::Swallow
                } else {
                    KeyAction::Forward
                }
            }
        }
    }

    /// Whether some Ctrl and some Alt are both down, either side of each.
    fn ctrl_and_alt_held(&self) -> bool {
        (self.pressed.contains(&KEY_LEFTCTRL) || self.pressed.contains(&KEY_RIGHTCTRL))
            && (self.pressed.contains(&KEY_LEFTALT) || self.pressed.contains(&KEY_RIGHTALT))
    }

    /// A synthetic release for every key still down, for the moment the session is disabled.
    pub fn release_all(&mut self) -> Vec<BackendMessage> {
        self.swallowed.clear();
        let now = MonotonicTimeStamp::now();
        self.pressed
            .drain()
            .map(|evdev| BackendMessage::KeyInput {
                time: now,
                keycode: evdev + 8,
                state: KeyState::Released,
            })
            .collect()
    }
}

/// The VT a function key names, or `None` for any other key
fn vt_of(evdev: u32) -> Option<i32> {
    match evdev {
        59..=68 => Some(i32::try_from(evdev - 58).unwrap_or(1)),
        87 => Some(11),
        88 => Some(12),
        _ => None,
    }
}
