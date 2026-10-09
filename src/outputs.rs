//! How a backend describes the displays it has, and the compositor describes them onward as `wl_output`

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

/// A display scale, in 120ths of one — the unit `wp_fractional_scale_v1` uses
///
/// 120 is one physical pixel per logical pixel, 180 is 1.5×, 240 is 2×. Never below 1×.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scale(u32);

impl Default for Scale {
    fn default() -> Self {
        Self::ONE
    }
}

impl Scale {
    /// One physical pixel per logical pixel
    pub const ONE: Scale = Scale(120);

    /// From a whole-number factor (1×, 2×, …), clamped to at least 1×
    #[must_use]
    pub fn from_integer(factor: i32) -> Self {
        Self(factor.max(1).unsigned_abs().saturating_mul(120))
    }

    /// From a count of 120ths, clamped to at least 1× — the form `wp_fractional_scale_v1` sends
    #[must_use]
    pub fn from_120ths(ths: u32) -> Self {
        Self(ths.max(120))
    }

    /// From a floating factor a host reports (1.5, 2.0, …), rounded to the nearest 120th
    /// and clamped to at least 1×. A non-finite input falls back to 1×.
    #[must_use]
    pub fn from_f64(factor: f64) -> Self {
        if !factor.is_finite() {
            return Self::ONE;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let ths = (factor * 120.0).round().max(120.0) as u32;
        Self(ths)
    }

    /// The factor as a count of 120ths, for `wp_fractional_scale_v1`
    #[must_use]
    pub fn as_120ths(self) -> u32 {
        self.0
    }

    /// The factor itself, for the one place that scales pixels: the renderer
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / 120.0
    }

    /// The integer `wl_output.scale` must carry: the ceiling of the real factor, so a
    /// client without fractional scale allocates a buffer large enough
    #[must_use]
    pub fn wl_output_scale(self) -> i32 {
        self.0.div_ceil(120).cast_signed()
    }

    /// Divide a physical length by this scale, rounded to the nearest logical pixel
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
    /// The scale of this output, possibly fractional
    pub scale: Scale,
    /// The name for this output
    pub name: String,
    /// The description for this output
    pub description: String,
}

impl Output {
    /// The size the compositor lays windows out in: the physical size divided by the scale
    ///
    /// Everything above the renderer works in this space — window positions, maximised
    /// sizes, the cursor's range, hit testing.
    #[must_use]
    pub fn logical_size(&self) -> (i32, i32) {
        (
            self.scale.logical(self.geometry.physical_width),
            self.scale.logical(self.geometry.physical_height),
        )
    }

    /// How many physical pixels one logical pixel covers, possibly fractional
    #[must_use]
    pub fn effective_scale(&self) -> Scale {
        self.scale
    }
}

/// The positions a cursor may occupy on an output, as an inclusive rectangle in logical pixels
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

/// Whether an output's area contains a point, in global logical coordinates
#[must_use]
pub fn output_contains(output: &Output, x: i32, y: i32) -> bool {
    let g = &output.geometry;
    let (width, height) = output.logical_size();
    x >= g.x && x < g.x + width && y >= g.y && y < g.y + height
}
