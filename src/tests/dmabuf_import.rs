//! Tests for the parts of the import path that need no GPU.
//!
//! The import itself is a driver call and is checked at runtime by
//! [`DmabufImporter::self_test`] instead; what is unit-testable is the
//! reasoning around it — which is also where a mistake is silent rather
//! than loud.

use crate::dmabuf_import::{advertisable_format, has_extension};
use crate::dma::{
    DRM_FORMAT_ARGB8888, DRM_FORMAT_MOD_INVALID, DRM_FORMAT_XRGB8888, fourcc, fourcc_name,
};

/// A linear layout, which any driver can sample.
const LINEAR: u64 = 0;
/// A tiled layout, standing in for anything a driver might name.
const TILED: u64 = 0x0100_0000_0000_0001;

#[test]
fn an_extension_is_matched_whole() {
    let extensions =
        "EGL_KHR_image_base EGL_EXT_image_dma_buf_import_modifiers EGL_EXT_buffer_age";

    assert!(has_extension(extensions, "EGL_KHR_image_base"));
    assert!(has_extension(
        extensions,
        "EGL_EXT_image_dma_buf_import_modifiers"
    ));
    // The base import extension is a different one with different entry
    // points, and a substring match would claim the driver has it.
    assert!(!has_extension(extensions, "EGL_EXT_image_dma_buf_import"));
    assert!(!has_extension(extensions, "EGL_KHR_image"));
    assert!(!has_extension("", "EGL_EXT_image_dma_buf_import"));
}

#[test]
fn fourccs_are_the_four_characters_they_are_named_for() {
    assert_eq!(fourcc_name(DRM_FORMAT_ARGB8888), "AR24");
    assert_eq!(fourcc_name(DRM_FORMAT_XRGB8888), "XR24");
    // Whatever a driver hands back, the log line stays readable.
    assert_eq!(fourcc_name(0), "????");
}

#[test]
fn a_driver_that_names_no_modifiers_leaves_the_format_importable() {
    // Nothing named means the implicit layout is all there is, and that
    // imports. Dropping the format here would refuse buffers that work.
    let format = advertisable_format(DRM_FORMAT_ARGB8888, &[]).expect("should be advertised");

    assert_eq!(format.fourcc, DRM_FORMAT_ARGB8888);
    assert!(
        format.modifiers.is_empty(),
        "an empty list is what says {DRM_FORMAT_MOD_INVALID:#x} is the only option"
    );
}

#[test]
fn modifiers_needing_an_external_sampler_are_not_offered() {
    let format = advertisable_format(DRM_FORMAT_XRGB8888, &[(LINEAR, false), (TILED, true)])
        .expect("should be advertised");

    // The tiled one could only be sampled through `samplerExternalOES`,
    // which this renderer has no program for, so a client must not be told
    // to allocate with it.
    assert_eq!(format.modifiers, [LINEAR]);
}

#[test]
fn a_format_no_modifier_of_which_can_be_sampled_is_dropped() {
    // Every YUV format Mesa reports looks like this. Offering one would
    // have a client allocate NV12 against our list and then find nothing
    // drawn — and by then it has no fallback left.
    assert!(advertisable_format(fourcc(*b"NV12"), &[(LINEAR, true), (TILED, true)]).is_none());
}

#[test]
fn naming_nothing_and_naming_nothing_usable_are_opposite_answers() {
    // The two look alike — both end with no usable modifier named — and
    // mean opposite things. Collapsing them is the mistake this whole rule
    // exists to avoid, in one direction or the other.
    let unnamed = advertisable_format(fourcc(*b"AR24"), &[]);
    let unusable = advertisable_format(fourcc(*b"NV12"), &[(TILED, true)]);

    assert!(unnamed.is_some_and(|f| f.modifiers.is_empty()));
    assert!(unusable.is_none());
}
