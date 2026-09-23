//! Ctrl+Alt+F1..F12, caught before the compositor ever sees them.
//!
//! On bare hardware nothing above this backend will switch VTs for the user:
//! there is no host compositor, and the kernel gave up its own handling of
//! these chords when the session took the tty into graphics mode. So the
//! chord is the backend's to honour, and it is handled here rather than in
//! the compositor's keybinding table on purpose — leaving the machine is
//! session mechanism, like the enable/disable that comes back, and it should
//! work even on a compositor whose config bound nothing at all.
//!
//! This is also where key state is squared with the compositor across a
//! switch. While the session is away every release happens on some other
//! VT and is never seen here, so without help the compositor would come
//! back believing Ctrl and Alt were still down and read every plain
//! keystroke as a chord. [`VtKeys::release_all`] is the help: synthetic
//! releases for everything held, sent at the moment of the disable.

use std::collections::HashSet;

use crate::input::KeyState;
use crate::messages::BackendMessage;
use crate::monotonic_timestamp::MonotonicTimeStamp;

/// Evdev codes for the keys the chord is made of.
const KEY_LEFTCTRL: u32 = 29;
const KEY_LEFTALT: u32 = 56;
const KEY_RIGHTCTRL: u32 = 97;
const KEY_RIGHTALT: u32 = 100;

/// What to do with one key event, decided by [`VtKeys::on_key`].
pub enum KeyAction {
    /// Not the backend's business: send it to the compositor as ever.
    Forward,
    /// Part of a chord the backend already acted on; the compositor must not
    /// see it, or a client would be handed half of a VT switch.
    Swallow,
    /// The chord: ask the seat for this VT, and swallow the key.
    SwitchVt(i32),
}

/// The held keys, watched for VT-switch chords.
#[derive(Default)]
pub struct VtKeys {
    /// Every key currently down, as evdev codes. Kept for two reasons: the
    /// chord test, and knowing what to release when the session goes away.
    pressed: HashSet<u32>,
    /// Function keys whose press was consumed for a switch, so that their
    /// release — if it arrives before the disable does — is swallowed too
    /// rather than reaching a client unpaired.
    swallowed: HashSet<u32>,
}

impl VtKeys {
    /// Track one key event and say what should happen to it.
    ///
    /// Anything that is not a key event is forwarded untouched.
    pub fn on_key(&mut self, message: &BackendMessage) -> KeyAction {
        let &BackendMessage::KeyInput { keycode, state, .. } = message else {
            return KeyAction::Forward;
        };
        // The stream carries xkb codes — evdev plus eight, see the
        // translation in `libinput_source` — and this undoes it.
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

    /// A synthetic release for every key still down, for the moment the
    /// session is disabled.
    ///
    /// The real releases are about to happen on another VT where this
    /// process cannot see them, so these stand in — without them the
    /// compositor keeps the chord's Ctrl and Alt (and whatever else was
    /// held) pressed forever. Clears the tracker: the session that comes
    /// back starts from no keys down, which is also what the hands on the
    /// keyboard will say.
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

/// The VT a function key names, or `None` for any other key.
///
/// F1..F10 are contiguous in evdev; F11 and F12 are not, a gap inherited
/// from the original PC keyboard's scancodes.
fn vt_of(evdev: u32) -> Option<i32> {
    match evdev {
        59..=68 => Some(i32::try_from(evdev - 58).unwrap_or(1)),
        87 => Some(11),
        88 => Some(12),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(keycode_evdev: u32, pressed: bool) -> BackendMessage {
        BackendMessage::KeyInput {
            time: MonotonicTimeStamp {
                tv_sec: 0,
                tv_nsec: 0,
            },
            keycode: keycode_evdev + 8,
            state: if pressed {
                KeyState::Pressed
            } else {
                KeyState::Released
            },
        }
    }

    #[test]
    fn ctrl_alt_f2_switches_to_vt_2_and_is_swallowed() {
        let mut keys = VtKeys::default();
        assert!(matches!(keys.on_key(&key(KEY_LEFTCTRL, true)), KeyAction::Forward));
        assert!(matches!(keys.on_key(&key(KEY_LEFTALT, true)), KeyAction::Forward));
        // F2 is evdev 60.
        assert!(matches!(keys.on_key(&key(60, true)), KeyAction::SwitchVt(2)));
        // Its release must not reach a client either.
        assert!(matches!(keys.on_key(&key(60, false)), KeyAction::Swallow));
    }

    #[test]
    fn a_function_key_without_the_chord_is_an_ordinary_key() {
        let mut keys = VtKeys::default();
        assert!(matches!(keys.on_key(&key(59, true)), KeyAction::Forward));
        assert!(matches!(keys.on_key(&key(59, false)), KeyAction::Forward));
    }

    #[test]
    fn the_noncontiguous_function_keys_map_to_their_vts() {
        let mut keys = VtKeys::default();
        keys.on_key(&key(KEY_RIGHTCTRL, true));
        keys.on_key(&key(KEY_RIGHTALT, true));
        assert!(matches!(keys.on_key(&key(87, true)), KeyAction::SwitchVt(11)));
        keys.on_key(&key(87, false));
        assert!(matches!(keys.on_key(&key(88, true)), KeyAction::SwitchVt(12)));
    }

    #[test]
    fn release_all_releases_exactly_what_was_held() {
        let mut keys = VtKeys::default();
        keys.on_key(&key(KEY_LEFTCTRL, true));
        keys.on_key(&key(KEY_LEFTALT, true));
        keys.on_key(&key(60, true));
        let released: HashSet<u32> = keys
            .release_all()
            .iter()
            .filter_map(|m| match m {
                BackendMessage::KeyInput {
                    keycode,
                    state: KeyState::Released,
                    ..
                } => Some(keycode - 8),
                _ => None,
            })
            .collect();
        assert_eq!(released, [KEY_LEFTCTRL, KEY_LEFTALT, 60].into());
        // And it starts over: nothing held, nothing swallowed.
        assert!(keys.release_all().is_empty());
        assert!(matches!(keys.on_key(&key(60, false)), KeyAction::Forward));
    }
}
