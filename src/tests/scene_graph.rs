//! Tests for buffer transforms: how a client's declared rotation/flip maps
//! destination coordinates back onto its buffer.

// Every coordinate here is 0.0 or 1.0, so the casts to `i32` are exact.
#![allow(clippy::cast_possible_truncation)]

use crate::scene_graph::BufferTransform;

/// Where a destination corner reads from in the buffer, under a transform.
fn sample(transform: BufferTransform, dx: f32, dy: f32) -> (f32, f32) {
    let ((ox, oy), basis) = transform.uv_map();
    (
        ox + basis[0][0] * dx + basis[1][0] * dy,
        oy + basis[0][1] * dx + basis[1][1] * dy,
    )
}

#[test]
fn an_untransformed_buffer_is_sampled_straight_through() {
    assert_eq!(sample(BufferTransform::Normal, 0.0, 0.0), (0.0, 0.0));
    assert_eq!(sample(BufferTransform::Normal, 1.0, 1.0), (1.0, 1.0));
}

#[test]
fn every_transform_maps_the_quad_onto_itself() {
    // Whatever the rotation or flip, the four destination corners must
    // land on the four buffer corners — exactly once each. A map that did
    // not would be sampling outside the buffer or reading part of it
    // twice.
    for transform in [
        BufferTransform::Normal,
        BufferTransform::Rotate90,
        BufferTransform::Rotate180,
        BufferTransform::Rotate270,
        BufferTransform::Flipped,
        BufferTransform::FlippedRotate90,
        BufferTransform::FlippedRotate180,
        BufferTransform::FlippedRotate270,
    ] {
        let mut corners: Vec<(i32, i32)> = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            .into_iter()
            .map(|(dx, dy)| {
                let (sx, sy) = sample(transform, dx, dy);
                // Exact in binary: every value here is 0 or 1.
                (sx.round() as i32, sy.round() as i32)
            })
            .collect();
        corners.sort_unstable();
        assert_eq!(
            corners,
            vec![(0, 0), (0, 1), (1, 0), (1, 1)],
            "{transform:?} does not cover the buffer exactly once"
        );
    }
}

#[test]
fn a_quarter_turn_is_the_inverse_of_the_client_rotation() {
    // The client rotated its buffer 90 degrees counter-clockwise, so the
    // top-left of the surface reads from the bottom-left of the buffer.
    assert_eq!(sample(BufferTransform::Rotate90, 0.0, 0.0), (1.0, 0.0));
    assert_eq!(sample(BufferTransform::Rotate90, 1.0, 0.0), (1.0, 1.0));
}

#[test]
fn only_the_quarter_turns_exchange_the_axes() {
    assert!(!BufferTransform::Normal.swaps_axes());
    assert!(!BufferTransform::Rotate180.swaps_axes());
    assert!(!BufferTransform::Flipped.swaps_axes());
    assert!(!BufferTransform::FlippedRotate180.swaps_axes());
    assert!(BufferTransform::Rotate90.swaps_axes());
    assert!(BufferTransform::Rotate270.swaps_axes());
    assert!(BufferTransform::FlippedRotate90.swaps_axes());
    assert!(BufferTransform::FlippedRotate270.swaps_axes());
}

use crate::scene_graph::ElementTransform;

/// Two points are the same place, allowing for the sin/cos round trip.
fn close(a: (f32, f32), b: (f32, f32)) -> bool {
    (a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4
}

#[test]
fn the_identity_transform_moves_nothing() {
    let transform = ElementTransform::default();
    assert!(transform.is_identity());
    assert_eq!(transform.apply(3.0, 4.0), (3.0, 4.0));
}

#[test]
fn element_rotation_is_clockwise_in_screen_coordinates() {
    // y grows downward, so a positive quarter turn takes "right" to
    // "down" — clockwise as the user sees it.
    let quarter = ElementTransform::rotate_about(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    assert!(!quarter.is_identity());
    assert!(close(quarter.apply(1.0, 0.0), (0.0, 1.0)));
    assert!(close(quarter.apply(0.0, 1.0), (-1.0, 0.0)));
}

#[test]
fn a_perspective_flip_foreshortens_for_real() {
    // A quarter-eighth turn about the vertical axis through (1, 1),
    // viewed from 10 logical pixels away.
    let flip =
        ElementTransform::perspective_rotate_y(std::f32::consts::FRAC_PI_4, 1.0, 1.0, 10.0);
    let cos = std::f32::consts::FRAC_PI_4.cos();
    assert!(!flip.is_affine(), "perspective lives in the bottom row");
    assert!(close(flip.apply(1.0, 1.0), (1.0, 1.0)), "the pivot holds");

    // The near (right) edge comes toward the viewer: it reaches further
    // than the flat squash, and stands taller. The far edge does the
    // opposite. That asymmetry IS the perspective — a flat squash has
    // none.
    let (near_x, _) = flip.apply(2.0, 1.0);
    let (far_x, _) = flip.apply(0.0, 1.0);
    assert!(near_x > 1.0 + cos, "near edge overshoots the affine squash");
    assert!((1.0 - far_x) < cos, "far edge undershoots it");
    let (_, near_y) = flip.apply(2.0, 2.0);
    let (_, far_y) = flip.apply(0.0, 2.0);
    assert!(near_y > 2.0, "the near corner grows away from the centre");
    assert!(far_y < 2.0, "the far corner shrinks toward it");

    // The x-axis twin tilts the bottom edge toward the viewer instead.
    let tilt =
        ElementTransform::perspective_rotate_x(std::f32::consts::FRAC_PI_4, 1.0, 1.0, 10.0);
    assert!(close(tilt.apply(1.0, 1.0), (1.0, 1.0)));
    let (bottom_x, _) = tilt.apply(2.0, 2.0);
    assert!(bottom_x > 2.0, "the bottom edge widens as it nears");
}

#[test]
fn infinite_depth_is_the_flat_squash() {
    // The limit of moving the eye away is the affine projection: the
    // element narrows by cos and nothing else moves. Degenerate depths
    // take the same road rather than dividing by them.
    let cos = std::f32::consts::FRAC_PI_4.cos();
    for depth in [f32::INFINITY, 0.0, -5.0, f32::NAN] {
        let flat = ElementTransform::perspective_rotate_y(
            std::f32::consts::FRAC_PI_4,
            1.0,
            1.0,
            depth,
        );
        assert!(flat.is_affine(), "no perspective at depth {depth}");
        assert!(close(flat.apply(2.0, 1.0), (1.0 + cos, 1.0)));
        assert!(close(flat.apply(2.0, 5.0), (1.0 + cos, 5.0)));
    }
}

#[test]
fn plain_rotations_stay_affine() {
    assert!(ElementTransform::IDENTITY.is_affine());
    assert!(ElementTransform::rotate_about(1.0, 3.0, 4.0).is_affine());
}

#[test]
fn rotating_about_the_centre_keeps_the_centre() {
    // A 2x4 element turned upside-down about its centre: the centre
    // stays, and the top-left corner lands on the bottom-right.
    let half = ElementTransform::rotate_about(std::f32::consts::PI, 1.0, 2.0);
    assert!(close(half.apply(1.0, 2.0), (1.0, 2.0)));
    assert!(close(half.apply(0.0, 0.0), (2.0, 4.0)));

    let skewed = ElementTransform::rotate_about(1.234, 5.0, 7.0);
    assert!(close(skewed.apply(5.0, 7.0), (5.0, 7.0)));
}

#[test]
fn a_transform_the_protocol_does_not_define_is_refused() {
    assert_eq!(BufferTransform::from_wire(0), Some(BufferTransform::Normal));
    assert_eq!(
        BufferTransform::from_wire(7),
        Some(BufferTransform::FlippedRotate270)
    );
    assert_eq!(BufferTransform::from_wire(8), None);
}
