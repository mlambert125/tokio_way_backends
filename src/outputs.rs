//! How a backend describes the displays it has, and the compositor describes
//! them onward to clients as `wl_output`.

use strum::FromRepr;

/// The output mode is the currently selected mode
pub const OUTPUT_MODE_CURRENT: u32 = 0x1;
/// The output mode is the preferred output mode for this output
pub const OUTPUT_MODE_PREFERRED: u32 = 0x2;

/// Transform on this output (flipped/rotated)
#[derive(Debug, Clone, Copy, FromRepr)]
#[repr(u32)]
pub enum OutputTransform {
    /// No transform applied
    Normal = 0,
    /// Rotated 90 degrees
    Rotate90 = 1,
    /// Rotated 180 degrees
    Rotate180 = 2,
    /// Rotated 270 degrees
    Rotate270 = 3,
    /// Flipped
    Flipped = 4,
    /// Flipped and rotated 90 degrees
    Flipped90 = 5,
    /// Flipped and rotated 180 degrees
    Flipped180 = 6,
    /// Flipped and rotated 270 degrees
    Flipped270 = 7,
}

/// The arrangement of subpixels on the display
#[derive(Debug, Clone, Copy, FromRepr)]
#[repr(u32)]
pub enum OutputSubpixel {
    /// Unknown
    Unknown = 0,
    /// Explicitly not applicable (e.g. for winit)
    None = 1,
    /// Subpixels are horizontal in RGB order
    HorizontalRgb = 2,
    /// Subpixels are horizontal in BGR order
    HorizontalBgr = 3,
    /// Subpixels are vertical in RGB order
    VerticalRgb = 4,
    /// Subpixels are vertical in BGR order
    VerticalBgr = 5,
}

/// Output geometry
#[derive(Debug, Clone)]
pub struct OutputGeometry {
    /// The top-left pixel of this output's x location in global space
    pub x: i32,
    /// The top-left pixel of this output's y location in global space
    pub y: i32,
    /// The physical width in pixels of this output
    pub physical_width: i32,
    /// The physical height in pixels of this output
    pub physical_height: i32,
    /// The subpixel spec for this output
    pub subpixel: OutputSubpixel,
    /// The make of this output/monitor
    pub make: String,
    /// The model of this output/monitor
    pub model: String,
    /// The transform applied to this output
    pub transform: OutputTransform,
}

/// The mode of an output/monitor
#[derive(Debug, Clone)]
pub struct OutputMode {
    /// Flags indicating additional details of this mode: (`OUTPUT_MODE_CURRENT`, `OUTPUT_MODE_PREFERRED`)
    pub flags: u32,
    /// Width for this mode
    pub width: i32,
    /// Height for this mode
    pub height: i32,
    /// Refresh rate in mhz
    pub refresh_mhz: i32,
}

/// A unique output id
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputId(pub u32);

/// A display scale, in 120ths of one — the unit `wp_fractional_scale_v1` uses.
///
/// An integer count of 120ths rather than a float, so equality is exact and
/// there is no rounding to reason about at each comparison. 120 is one physical
/// pixel per logical pixel, 180 is 1.5×, 240 is 2×. Never below 1×: a scale
/// under one would magnify everything and has no meaning for a display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scale(u32);

impl Default for Scale {
    fn default() -> Self {
        Self::ONE
    }
}

impl Scale {
    /// One physical pixel per logical pixel.
    pub const ONE: Scale = Scale(120);

    /// From a whole-number factor (1×, 2×, …), clamped to at least 1×.
    #[must_use]
    pub fn from_integer(factor: i32) -> Self {
        Self(factor.max(1).unsigned_abs().saturating_mul(120))
    }

    /// From a count of 120ths, clamped to at least 1×. The form
    /// `wp_fractional_scale_v1` sends, for when a client's preferred scale is
    /// read back rather than the host's reported.
    #[must_use]
    pub fn from_120ths(ths: u32) -> Self {
        Self(ths.max(120))
    }

    /// From a floating factor a host reports (1.5, 2.0, …), rounded to the
    /// nearest 120th and clamped to at least 1×. A non-finite input falls back
    /// to 1×.
    #[must_use]
    pub fn from_f64(factor: f64) -> Self {
        if !factor.is_finite() {
            return Self::ONE;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let ths = (factor * 120.0).round().max(120.0) as u32;
        Self(ths)
    }

    /// The factor as a count of 120ths, for `wp_fractional_scale_v1`.
    #[must_use]
    pub fn as_120ths(self) -> u32 {
        self.0
    }

    /// The factor itself, for the one place that scales pixels: the renderer.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / 120.0
    }

    /// The integer `wl_output.scale` must carry: the ceiling of the real
    /// factor, so a client that cannot do fractional scale still allocates a
    /// buffer large enough rather than one a little too small.
    #[must_use]
    pub fn wl_output_scale(self) -> i32 {
        self.0.div_ceil(120).cast_signed()
    }

    /// Divide a physical length by this scale, rounded to the nearest logical
    /// pixel. This is the one rounding the whole fractional story turns on, so
    /// it lives in one place.
    #[must_use]
    pub fn logical(self, physical: i32) -> i32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let logical = (f64::from(physical) / self.as_f64()).round() as i32;
        logical
    }
}

/// An output/monitor available to this backend
#[derive(Debug, Clone)]
pub struct Output {
    /// The output id
    pub id: OutputId,
    /// The geometry for this output
    pub geometry: OutputGeometry,
    /// The modes for this output
    pub modes: Vec<OutputMode>,
    /// The scale of this output, possibly fractional.
    pub scale: Scale,
    /// The name for this output
    pub name: String,
    /// The description for this output
    pub description: String,
}

impl Output {
    /// The size the compositor lays windows out in.
    ///
    /// Two coordinate spaces meet at an output and they are not the same one.
    /// The *physical* size is how many pixels the display really has, and is
    /// what the framebuffer is and what `wl_output.mode` reports. The
    /// *logical* size is that divided by the scale, and is what everything
    /// above the renderer works in: window positions, maximised sizes, the
    /// cursor's range, hit testing.
    ///
    /// Keeping layout logical is what makes a scaled display work at all. A
    /// window asked to fill a 3840-wide display at scale 2 must be told it is
    /// 1920 wide, because that is the size it will draw at — and it then
    /// submits a 3840-wide buffer at `buffer_scale` 2 to fill the pixels.
    /// Laying out in physical pixels instead gives a window twice the size of
    /// the screen it is on.
    #[must_use]
    pub fn logical_size(&self) -> (i32, i32) {
        (
            self.scale.logical(self.geometry.physical_width),
            self.scale.logical(self.geometry.physical_height),
        )
    }

    /// How many physical pixels one logical pixel covers, possibly fractional.
    #[must_use]
    pub fn effective_scale(&self) -> Scale {
        self.scale
    }
}

/// The positions a cursor may occupy on an output, as an inclusive rectangle.
///
/// Logical, because the cursor is positioned in the space windows live in.
#[must_use]
pub fn cursor_bounds(output: &Output) -> Option<(f64, f64, f64, f64)> {
    let g = &output.geometry;
    let (width, height) = output.logical_size();
    if width <= 0 || height <= 0 {
        return None;
    }
    Some((
        f64::from(g.x),
        f64::from(g.y),
        f64::from(g.x + width - 1),
        f64::from(g.y + height - 1),
    ))
}

/// Whether an output's area contains a point, in global logical coordinates.
#[must_use]
pub fn output_contains(output: &Output, x: i32, y: i32) -> bool {
    let g = &output.geometry;
    let (width, height) = output.logical_size();
    x >= g.x && x < g.x + width && y >= g.y && y < g.y + height
}

#[cfg(test)]
mod tests {
    //! Tests for the two coordinate spaces an output has.
    //!
    //! An output's *physical* size is how many pixels the display really has,
    //! and is what the framebuffer is. Its *logical* size is that divided by
    //! the scale, and is what everything above the renderer works in.
    //! Confusing the two is not a small error: laying out in physical pixels
    //! on a scale-2 display gives every maximised window twice the size of the
    //! screen it is on.

    use super::*;

    fn output_of(width: i32, height: i32, scale: i32) -> Output {
        Output {
            id: OutputId(1),
            geometry: OutputGeometry {
                x: 0,
                y: 0,
                physical_width: width,
                physical_height: height,
                subpixel: OutputSubpixel::None,
                make: String::new(),
                model: String::new(),
                transform: OutputTransform::Normal,
            },
            modes: vec![OutputMode {
                flags: OUTPUT_MODE_CURRENT,
                width,
                height,
                refresh_mhz: 60000,
            }],
            scale: Scale::from_integer(scale),
            name: String::new(),
            description: String::new(),
        }
    }

    #[test]
    fn an_unscaled_output_has_one_size() {
        let output = output_of(1920, 1080, 1);
        assert_eq!(output.logical_size(), (1920, 1080));
        assert_eq!(output.effective_scale(), Scale::ONE);
    }

    #[test]
    fn a_scaled_output_lays_out_smaller_than_it_draws() {
        // The framebuffer is still 3840 wide. What changes is that a window
        // asked to fill this display is told it is 1920 wide, because that is
        // the size it draws at — it then submits a 3840-wide buffer at
        // buffer_scale 2.
        let output = output_of(3840, 2160, 2);
        assert_eq!(output.logical_size(), (1920, 1080));
        assert_eq!(
            output.geometry.physical_width, 3840,
            "the physical size is untouched: it is what wl_output.mode reports",
        );
    }

    #[test]
    fn a_fractional_scale_divides_the_logical_size() {
        // 1.5× of a 2560-wide display is a 1707-logical-pixel desktop
        // (2560 / 1.5, rounded), drawn onto the full 2560 pixels.
        let mut output = output_of(2560, 1440, 1);
        output.scale = Scale::from_f64(1.5);
        assert_eq!(output.logical_size(), (1707, 960));
        // wl_output.scale is integer, so a client too old for fractional scale
        // is told 2 — the ceiling — and allocates a buffer big enough.
        assert_eq!(output.scale.wl_output_scale(), 2);
        // The exact factor survives as 120ths, which is what the
        // fractional-scale protocol carries.
        assert_eq!(output.scale.as_120ths(), 180);
    }

    #[test]
    fn a_scale_below_one_is_treated_as_one() {
        // A backend that reports zero, or a negative, must not divide the
        // desktop by it — the layout would be empty or inverted, and every
        // window would land off screen.
        for bad in [0, -1, i32::MIN] {
            let output = output_of(800, 600, bad);
            assert_eq!(output.logical_size(), (800, 600), "scale {bad}");
            assert_eq!(output.effective_scale(), Scale::ONE, "scale {bad}");
        }
    }

    #[test]
    fn the_cursor_is_confined_to_the_logical_area() {
        // The cursor is positioned in the space windows live in, so a scale-2
        // output confines it to half the pixels it has — the other half is
        // reached by the same logical coordinates, drawn twice as large.
        let output = output_of(800, 600, 2);
        assert_eq!(cursor_bounds(&output), Some((0.0, 0.0, 399.0, 299.0)));

        let unscaled = output_of(800, 600, 1);
        assert_eq!(cursor_bounds(&unscaled), Some((0.0, 0.0, 799.0, 599.0)));
    }

    #[test]
    fn a_point_is_on_an_output_by_its_logical_area() {
        let output = output_of(800, 600, 2);
        assert!(output_contains(&output, 399, 299));
        assert!(
            !output_contains(&output, 400, 0),
            "past the logical right edge, even though the display has pixels there",
        );
        assert!(!output_contains(&output, 0, 300));
    }

    #[test]
    fn an_output_with_no_size_confines_nothing() {
        assert_eq!(cursor_bounds(&output_of(0, 0, 1)), None);
    }
}
