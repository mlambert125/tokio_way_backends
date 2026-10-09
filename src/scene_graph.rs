use crate::{
    dma::{AcquireFence, DmabufImage, ReleaseFence},
    outputs::{OutputId, Scale},
    shm::UploadPixels,
};
use std::sync::Arc;

/// An axis-aligned rectangle in texture pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRect {
    /// Location X
    pub x: i32,
    /// Location Y
    pub y: i32,
    /// Width of the texture
    pub width: i32,
    /// Height of the texture
    pub height: i32,
}

/// The pointer cursor, carried alongside the scenes rather than inside one
#[derive(Debug, Default, Clone)]
pub struct Cursor {
    /// The output the pointer is over, or `None` when it is over none
    pub output: Option<OutputId>,
    /// The cursor quads, in `output`'s logical coordinates. Empty when there is nothing to draw
    pub elements: Vec<SceneElement>,
    /// Rises whenever the cursor's position or appearance changes
    pub serial: u64,
}

/// One frame: the newest scene for every output, and where the cursor sits
#[derive(Debug, Default, Clone)]
pub struct SceneGraph {
    /// The newest scene for every output
    pub scenes: Vec<Arc<Scene>>,
    /// The pointer cursor
    pub cursor: Cursor,
}

/// Everything to draw for one output, back to front
#[derive(Debug)]
pub struct Scene {
    /// The target output
    pub output_id: OutputId,
    /// What the output is cleared to, as straight (not premultiplied) `0xAARRGGBB`
    pub background: u32,
    /// Distinguishes this scene from the last one composed for the same output, and rises with every one
    pub serial: u64,
    /// The elements to draw
    pub elements: Vec<SceneElement>,
    /// How many physical pixels one of this scene's logical pixels covers, possibly fractional
    pub scale: Scale,
    /// The scene `damage` is expressed against, or `None` when there was nothing to diff against
    pub damage_from: Option<u64>,
    /// What changed since [`Self::damage_from`], in logical pixels. Empty means the whole output
    pub damage: Vec<TextureRect>,
}

/// A projective (2D homography) transform an element applies to its own quad
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementTransform {
    /// The homography, column-major: `matrix[column][row]`
    pub matrix: [[f32; 3]; 3],
}

impl Default for ElementTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ElementTransform {
    /// Leaves the quad untouched
    pub const IDENTITY: Self = Self {
        matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    /// A rotation about an element-local pivot, in radians. Positive turns clockwise on screen
    #[must_use]
    pub fn rotate_about(radians: f32, cx: f32, cy: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self {
            matrix: [
                [cos, sin, 0.0],
                [-sin, cos, 0.0],
                // The pivot must land on itself: offset = pivot - R · pivot
                [cx - (cos * cx - sin * cy), cy - (sin * cx + cos * cy), 1.0],
            ],
        }
    }

    /// A card flip: rotation about the vertical axis through `x = cx`, seen in perspective from `depth` logical pixels away
    #[must_use]
    pub fn perspective_rotate_y(radians: f32, cx: f32, cy: f32, depth: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        let inv_depth = if depth > 0.0 && depth.is_finite() {
            sin / depth
        } else {
            0.0
        };
        // Each numerator · w written as a linear form, which is what makes it a homography
        Self {
            matrix: [
                [cos - cx * inv_depth, -cy * inv_depth, -inv_depth],
                [0.0, 1.0, 0.0],
                [
                    cx * (1.0 - cos) + cx * cx * inv_depth,
                    cy * cx * inv_depth,
                    1.0 + cx * inv_depth,
                ],
            ],
        }
    }

    /// The x-axis twin of [`Self::perspective_rotate_y`]: rotation about the horizontal axis through `y = cy`
    #[must_use]
    pub fn perspective_rotate_x(radians: f32, cx: f32, cy: f32, depth: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        let inv_depth = if depth > 0.0 && depth.is_finite() {
            sin / depth
        } else {
            0.0
        };
        Self {
            matrix: [
                [1.0, 0.0, 0.0],
                [-cx * inv_depth, cos - cy * inv_depth, -inv_depth],
                [
                    cx * cy * inv_depth,
                    cy * (1.0 - cos) + cy * cy * inv_depth,
                    1.0 + cy * inv_depth,
                ],
            ],
        }
    }

    /// Where an element-local point lands under this transform
    #[must_use]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.matrix;
        let hx = m[0][0] * x + m[1][0] * y + m[2][0];
        let hy = m[0][1] * x + m[1][1] * y + m[2][1];
        let hw = m[0][2] * x + m[1][2] * y + m[2][2];
        let hw = if hw.abs() < 1e-6 { 1e-6 } else { hw };
        (hx / hw, hy / hw)
    }

    /// Whether this is exactly the identity
    #[must_use]
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// Whether the bottom row is `(0, 0, 1)`: affine, no perspective, nothing ever divides
    // Exact on purpose: the question is whether the transform was built with no
    // perspective — the constructors write literal zeros — not whether it is nearly flat.
    #[allow(clippy::float_cmp)]
    #[must_use]
    pub fn is_affine(&self) -> bool {
        self.matrix[0][2] == 0.0 && self.matrix[1][2] == 0.0 && self.matrix[2][2] == 1.0
    }
}

/// One quad, in output logical coordinates
#[derive(Debug, Clone)]
pub struct SceneElement {
    /// What fills the quad
    pub content: SceneContent,
    /// Destination rectangle in the output's *logical* pixels: (x, y, width, height), fractional
    pub dst: (f64, f64, f64, f64),
    /// The element's own transform over that rectangle — see [`ElementTransform`]
    pub transform: ElementTransform,
    /// A fragment effect to run on this element's pixels — see [`ShaderEffect`]
    pub effect: Option<Arc<ShaderEffect>>,
    /// Whole-element opacity, `0.0..=1.0`, multiplied over the content's own alpha
    pub alpha: f32,
    /// The client has promised this quad is fully opaque, so the blend can be skipped
    pub opaque: bool,
}

impl SceneElement {
    /// The texture this element samples, if it is a textured one
    #[must_use]
    pub fn texture(&self) -> Option<&Arc<TextureImage>> {
        match &self.content {
            SceneContent::Texture { image, .. } => Some(image),
            SceneContent::Color(_) | SceneContent::Group(_) => None,
        }
    }
}

/// What fills a scene element's rectangle
#[derive(Debug, Clone)]
pub enum SceneContent {
    /// A texture, cropped and stretched into the destination
    Texture {
        /// The texture to sample
        image: Arc<TextureImage>,
        /// Source rectangle in texture pixels: (x, y, width, height)
        src: (f64, f64, f64, f64),
        /// How the client transformed its buffer, which the draw has to undo
        transform: BufferTransform,
    },
    /// A solid fill, as straight (not premultiplied) `0xAARRGGBB`
    Color(u32),
    /// A sub-scene composed offscreen, then drawn as one quad — see [`SceneGroup`]
    Group(Arc<SceneGroup>),
}

/// A sub-scene composed offscreen and drawn as one quad
#[derive(Debug)]
pub struct SceneGroup {
    /// The canvas size in logical pixels, `(0, 0)` at its top-left
    pub size: (f64, f64),
    /// What the canvas starts as, straight `0xAARRGGBB` — usually 0, fully transparent
    pub background: u32,
    /// The elements, back to front, in canvas coordinates
    pub elements: Vec<SceneElement>,
}

/// A fragment-stage effect a compositor attaches to an element
///
/// `source` is GLES 3.0 code spliced in at global scope, defining
/// `vec4 effect(vec4 texel, vec2 uv)`, which takes the repaired premultiplied
/// texel and element-local `uv` and returns the premultiplied colour to draw.
#[derive(Debug)]
pub struct ShaderEffect {
    /// A short name for humans, identifying the effect in logs and failure messages
    pub name: String,
    /// The GLES 3.0 snippet defining `vec4 effect(vec4 texel, vec2 uv)`
    pub source: Arc<str>,
    /// Values for the uniforms the snippet declared, by name
    pub uniforms: Vec<(String, EffectUniform)>,
}

/// A value for a uniform a [`ShaderEffect`] snippet declared
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectUniform {
    /// A `float`
    Float(f32),
    /// A `vec2`
    Vec2(f32, f32),
    /// A `vec4`
    Vec4(f32, f32, f32, f32),
}

/// One texture to draw, and everything the backend needs to get hold of it
///
/// Bytes, where there are any, are little-endian `0xAARRGGBB`, i.e. `[B, G, R, A]` per pixel.
#[derive(Debug)]
pub struct TextureImage {
    /// Id
    pub id: TextureId,
    /// A serial number for this texture
    pub serial: u64,
    /// Width of the image
    pub width: i32,
    /// Height of the image
    pub height: i32,
    /// Pixel format
    pub format: PixelFormat,
    /// Where the pixels come from. Always addressable in full, even when an upload's damage is not
    pub source: TextureSource,
}

impl TextureImage {
    /// Row stride in pixels, for GL's `UNPACK_ROW_LENGTH`
    #[must_use]
    pub fn row_length(&self) -> i32 {
        match &self.source {
            TextureSource::Upload {
                pixels: UploadPixels::Mapped { stride, .. },
                ..
            } => i32::try_from(stride / 4).unwrap_or(self.width),
            _ => self.width,
        }
    }

    /// Whether the source bytes are `[B, G, R, A]` and need swizzling when sampled
    #[must_use]
    pub fn swizzle_bgra(&self) -> bool {
        matches!(self.source, TextureSource::Upload { .. })
    }

    /// The GPU buffer behind this image, if that is what it is
    #[must_use]
    pub fn dmabuf(&self) -> Option<&Arc<DmabufImage>> {
        match &self.source {
            TextureSource::Dmabuf { image, .. } => Some(image),
            TextureSource::Upload { .. } => None,
        }
    }

    /// The fence gating this image's content, if its producer supplied one
    #[must_use]
    pub fn acquire_fence(&self) -> Option<&Arc<AcquireFence>> {
        match &self.source {
            TextureSource::Dmabuf { acquire, .. } => acquire.as_ref(),
            TextureSource::Upload { .. } => None,
        }
    }

    /// The cell the backend fills with a fence covering its reads of this image
    #[must_use]
    pub fn release_fence(&self) -> Option<&Arc<ReleaseFence>> {
        match &self.source {
            TextureSource::Dmabuf { release, .. } => release.as_ref(),
            TextureSource::Upload { .. } => None,
        }
    }

    /// Borrow the pixels, from the first byte of the image to the last
    ///
    /// # Safety
    ///
    /// For a mapped image this borrows a client's shm mapping; the caller must not
    /// keep the slice past the life of this image. Returns `None` if the image does
    /// not fit its mapping.
    #[must_use]
    pub unsafe fn bytes(&self) -> Option<&[u8]> {
        // On the GPU already; there is nothing here to read
        let TextureSource::Upload { pixels, .. } = &self.source else {
            return None;
        };
        match pixels {
            UploadPixels::Owned(bytes) => Some(bytes),
            UploadPixels::Mapped {
                guard,
                offset,
                stride,
            } => {
                let rows = self.height.unsigned_abs() as usize;
                let width_bytes = self.width.unsigned_abs() as usize * 4;
                // The last row needs no stride padding, so the extent is shorter than `rows * stride`
                let len = rows.checked_sub(1)?.checked_mul(*stride)? + width_bytes;
                // SAFETY: delegated to this function's own contract
                unsafe { guard.mapping().slice(*offset, len) }
            }
        }
    }
}

/// How a client has already transformed its buffer, from `wl_surface.set_buffer_transform`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BufferTransform {
    /// No transform applied
    #[default]
    Normal,
    /// Rotated 90 degrees
    Rotate90,
    /// Rotated 180 degrees
    Rotate180,
    /// Rotated 270 degrees
    Rotate270,
    /// Flipped
    Flipped,
    /// Rotated 90 degrees and flipped
    FlippedRotate90,
    /// Rotated 180 degrees and flipped
    FlippedRotate180,
    /// Rotated 270 degrees and flipped
    FlippedRotate270,
}

impl BufferTransform {
    /// From the `wl_output.transform` value a client sends, or `None` if it is invalid
    #[must_use]
    pub fn from_wire(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Normal,
            1 => Self::Rotate90,
            2 => Self::Rotate180,
            3 => Self::Rotate270,
            4 => Self::Flipped,
            5 => Self::FlippedRotate90,
            6 => Self::FlippedRotate180,
            7 => Self::FlippedRotate270,
            _ => return None,
        })
    }

    /// Whether this transform exchanges the buffer's width and height
    #[must_use]
    pub fn swaps_axes(self) -> bool {
        matches!(
            self,
            Self::Rotate90 | Self::Rotate270 | Self::FlippedRotate90 | Self::FlippedRotate270
        )
    }

    /// Where a point on the surface reads from in the buffer, as an affine map over
    /// unit coordinates: `(origin, basis)` such that `source = origin + basis * destination`.
    /// `basis` is column-major, and the map is the inverse of what the client applied.
    #[must_use]
    pub fn uv_map(self) -> ((f32, f32), [[f32; 2]; 2]) {
        match self {
            Self::Normal => ((0.0, 0.0), [[1.0, 0.0], [0.0, 1.0]]),
            Self::Rotate90 => ((1.0, 0.0), [[0.0, 1.0], [-1.0, 0.0]]),
            Self::Rotate180 => ((1.0, 1.0), [[-1.0, 0.0], [0.0, -1.0]]),
            Self::Rotate270 => ((0.0, 1.0), [[0.0, -1.0], [1.0, 0.0]]),
            Self::Flipped => ((1.0, 0.0), [[-1.0, 0.0], [0.0, 1.0]]),
            Self::FlippedRotate90 => ((0.0, 0.0), [[0.0, 1.0], [1.0, 0.0]]),
            Self::FlippedRotate180 => ((0.0, 1.0), [[1.0, 0.0], [0.0, -1.0]]),
            Self::FlippedRotate270 => ((1.0, 1.0), [[0.0, -1.0], [-1.0, 0.0]]),
        }
    }
}

/// Identity of a texture, used by the backend to cache GPU textures across frames
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureId {
    /// A client `wl_buffer`, keyed by (`client_id`, `buffer_id`)
    Buffer(u32, u32),
    /// The cursor loaded from the system cursor theme
    DefaultCursor,
    /// The built-in cursor used when no theme is available
    FallbackCursor,
}

/// Pixel layout of a texture's source bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// `WL_SHM_FORMAT_ARGB8888` — premultiplied alpha
    Argb8888,
    /// `WL_SHM_FORMAT_XRGB8888` — the high byte is undefined, treat as opaque
    Xrgb8888,
}

/// Where a texture comes from, and what the backend has to do to get it
#[derive(Debug)]
pub enum TextureSource {
    /// Bytes the backend has to send to the GPU
    Upload {
        /// The bytes themselves
        pixels: UploadPixels,
        /// The serial this image was derived from, if any — `damage` is only meaningful relative to it
        previous_serial: Option<u64>,
        /// What changed since `previous_serial`. Empty means "all of it"
        damage: Vec<TextureRect>,
    },
    /// Already on the GPU, shared as dma-buf descriptors — the backend imports them
    /// as a texture and samples the client's own memory
    Dmabuf {
        /// The buffer
        image: Arc<DmabufImage>,
        /// Explicit sync: signals when the producer's writes have landed, or `None` for implicit sync
        acquire: Option<Arc<AcquireFence>>,
        /// The release half: a cell the backend fills with a fence covering its GPU reads
        release: Option<Arc<ReleaseFence>>,
    },
}
