use crate::backends::drm::vt_switch::*;
use crate::input::KeyState;
use crate::messages::BackendMessage;
use crate::monotonic_timestamp::MonotonicTimeStamp;
use std::collections::HashSet;

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
