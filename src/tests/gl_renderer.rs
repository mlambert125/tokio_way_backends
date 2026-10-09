//! Tests for the fragment template an effect snippet is spliced into.
//! The GL calls themselves need a live context and are exercised by
//! running a backend; what is unit-testable is the source generation,
//! where a mistake would fail every effect on every driver.

use crate::gl_renderer::fragment_source;

#[test]
fn the_version_directive_stays_on_the_first_line() {
    // GLSL requires #version before anything else, including whitespace;
    // a template edit that pushes it down breaks every program.
    assert!(fragment_source(None).starts_with("#version 300 es\n"));
    assert!(fragment_source(Some("// snippet")).starts_with("#version 300 es\n"));
}

#[test]
fn a_snippet_is_spliced_in_and_called() {
    let snippet = "vec4 effect(vec4 texel, vec2 uv) { return texel; }";
    let source = fragment_source(Some(snippet));
    assert!(source.contains(snippet), "the declaration must be present");
    assert!(
        source.contains("texel = effect(texel, v_unit);"),
        "and the call must run it on the repaired texel"
    );
}

#[test]
fn the_plain_template_calls_no_effect() {
    let source = fragment_source(None);
    assert!(
        !source.contains("effect("),
        "a program with no snippet must not reference the hook"
    );
}

#[test]
fn a_group_canvas_is_never_empty_and_never_oversized() {
    use crate::gl_renderer::group_extent;
    // Logical size times scale, rounded up.
    assert_eq!(group_extent(100.0, 1.0, 16384), 100);
    assert_eq!(group_extent(100.5, 2.0, 16384), 201);
    // A degenerate size still allocates one pixel rather than nothing —
    // a zero-extent texture would fail the framebuffer, not the maths.
    assert_eq!(group_extent(0.0, 1.0, 16384), 1);
    assert_eq!(group_extent(-5.0, 1.0, 16384), 1);
    // And a huge one stops at what the driver can take.
    assert_eq!(group_extent(1_000_000.0, 2.0, 16384), 16384);
}
