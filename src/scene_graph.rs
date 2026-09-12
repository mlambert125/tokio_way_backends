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

/// The pointer cursor, kept out of the scene — see [`SceneGraph`].
///
/// A backend with a cursor plane programs it from this. One without composites
/// [`Self::elements`] over the scene of [`Self::output`], on top of everything
/// else, exactly where a scene element would have gone.
#[derive(Debug, Default, Clone)]
pub struct Cursor {
    /// The output the pointer is over, or `None` when it is over none — in
    /// which case there is nothing to draw and `elements` is empty.
    pub output: Option<OutputId>,
    /// The cursor quads, already positioned in `output`'s logical coordinates.
    /// Empty when there is nothing to draw: the pointer is off every output, or
    /// a client has hidden its cursor.
    pub elements: Vec<SceneElement>,
    /// Rises whenever the cursor's position or appearance changes. It lets a
    /// backend tell a cursor-only update from a content one, and so redraw the
    /// cursor without re-reporting a presentation that would fire clients'
    /// frame callbacks for content that did not change.
    pub serial: u64,
}

/// One frame: the newest scene for every output, and where the cursor sits.
///
/// Not a moment in time — the scenes in it were composed at whatever moment
/// their own output last asked for one. It is a slot holding the latest state
/// of every output at once, so that publishing a new scene for one output
/// cannot drop an unshown scene belonging to another. The serial on each scene
/// is what a backend uses to tell what is new to it.
///
/// The cursor rides alongside the scenes rather than inside one so that pointer
/// motion — the most frequent thing that happens — neither recomposes a scene
/// nor perturbs its serial or damage, and so a backend with a hardware cursor
/// plane can move it without touching the rest of the frame at all.
#[derive(Debug, Default, Clone)]
pub struct SceneGraph {
    /// The newest scene for every output.
    pub scenes: Vec<Arc<Scene>>,
    /// The pointer cursor.
    pub cursor: Cursor,
}

/// Everything to draw for one output, back to front.
#[derive(Debug)]
pub struct Scene {
    /// The target output
    pub output_id: OutputId,
    /// What the output is cleared to before the elements go down, as straight
    /// (not premultiplied) `0xAARRGGBB`: the pixels no element covers, and the
    /// surplus a resize exposes before a scene at the new size arrives. Policy
    /// belongs to the compositor composing the scene, so it rides in the scene
    /// rather than being the renderer's to choose.
    pub background: u32,
    /// Distinguishes this scene from the last one composed for the same
    /// output, and rises with every one.
    ///
    /// Outputs are paced apart — each is composed when the backend says it can
    /// show another frame for it — so a published frame is a mixture of scenes
    /// composed at different moments, most of which the backend has already
    /// drawn. This is what tells it which one it has not.
    pub serial: u64,
    /// The elements to draw
    pub elements: Vec<SceneElement>,
    /// How many physical pixels one of this scene's logical pixels covers,
    /// possibly fractional.
    ///
    /// The elements are in logical coordinates, because that is the space the
    /// compositor lays windows out in. The renderer is the one place that has
    /// to know the difference: it draws into a framebuffer measured in
    /// physical pixels, so it scales the projection rather than every quad.
    pub scale: Scale,
    /// The scene these damage rectangles are expressed against, if any. A
    /// backend can trust `damage` only if what it last drew on this output was
    /// exactly this serial; anything else — a scene it never drew, or one
    /// several behind — means the rectangles refer to pixels it does not have,
    /// so it must repaint the whole output. `None` whenever there was nothing
    /// to diff against. This is the scene-level twin of an upload's
    /// `previous_serial`, and serves the same purpose: output serials are not
    /// consecutive (one counter feeds every output), so "the previous one" has
    /// to be named rather than assumed to be `serial - 1`.
    pub damage_from: Option<u64>,
    /// What changed on the output since [`Self::damage_from`] — the regions a
    /// backend must repaint, in logical pixels. Empty means the whole output,
    /// the same convention texture damage uses, and is the honest answer
    /// whenever there is nothing to diff against.
    ///
    /// The composer's side of the contract: every rectangle has positive
    /// width and height and lies within the output. A backend may skip a
    /// rectangle that does not, and the pixels it covered go stale.
    ///
    /// A backend free to repaint everything (as the GL renderer does) may
    /// ignore this. One that pays for full repaints — a display using buffer
    /// age, a remote backend shipping pixels over a wire — patches only these
    /// rectangles, exactly as the texture-damage path already does one level
    /// down.
    pub damage: Vec<TextureRect>,
}

/// A projective (2D homography) transform an element applies to its own quad.
///
/// Maps element-local points — logical pixels measured from the element's
/// top-left, so the untransformed quad spans `(0, 0)` to `(width, height)` —
/// as homogeneous coordinates: `(hx, hy, hw) = matrix · (x, y, 1)`, landing
/// at `(hx / hw, hy / hw)`, anchored at [`SceneElement::dst`]'s origin as
/// before. Identity leaves the quad exactly where `dst` put it.
///
/// Affine transforms — rotation, scale, skew, translation — keep the bottom
/// row `(0, 0, 1)` and never divide. The bottom row is where perspective
/// lives: [`Self::perspective_rotate_y`] and its x twin build card-flip and
/// tilt maps whose foreshortening is real, not a flat squash. One contract
/// rides with that power: every corner of the quad must stay at `hw > 0`.
/// A corner at or behind zero is "behind the eye", and both the renderer
/// and [`Self::apply`] produce geometry-shaped nonsense for it — for the
/// rotation helpers that means keeping the turn comfortably short of 90°
/// at the chosen depth.
///
/// `matrix` is column-major, the same convention as
/// [`BufferTransform::uv_map`] and GL's `mat3`: `matrix[column][row]`. With
/// y growing downward, a positive [`Self::rotate_about`] angle turns
/// clockwise on screen.
///
/// Data rather than a shader on purpose: a backend can read it, and an
/// identity transform is one of the things that keeps an element eligible
/// for fast paths like plane offload — a rotated element visibly is not,
/// and [`Self::is_affine`] separates the merely-transformed from the
/// perspective-projected for reasoning finer than that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementTransform {
    /// The homography, column-major: `matrix[column][row]`.
    pub matrix: [[f32; 3]; 3],
}

impl Default for ElementTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ElementTransform {
    /// Leaves the quad untouched.
    pub const IDENTITY: Self = Self {
        matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    /// A rotation about an element-local pivot, in radians. Positive turns
    /// clockwise on screen — the y-down convention. Rotating about the
    /// element's centre is `rotate_about(angle, width / 2.0, height / 2.0)`.
    /// Affine: no foreshortening, bottom row untouched.
    #[must_use]
    pub fn rotate_about(radians: f32, cx: f32, cy: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self {
            matrix: [
                [cos, sin, 0.0],
                [-sin, cos, 0.0],
                // The pivot must land on itself: offset = pivot - R · pivot.
                [
                    cx - (cos * cx - sin * cy),
                    cy - (sin * cx + cos * cy),
                    1.0,
                ],
            ],
        }
    }

    /// A card flip: rotation about the vertical axis through `x = cx`, seen
    /// in perspective, with `(cx, cy)` as the centre of projection.
    ///
    /// Positive angles bring the element's right edge toward the viewer —
    /// it grows — and send the left away. `depth` is the viewing distance
    /// in logical pixels, CSS's `perspective()`: smaller is more dramatic,
    /// somewhere in 800–2000 reads naturally for window-sized elements. A
    /// depth that is zero, negative or non-finite degrades to the flat
    /// affine squash — the limit of infinite distance. The `hw > 0` rule
    /// means `|sin(angle)|` times the pivot's distance to the farthest
    /// corner must stay under `depth`.
    #[must_use]
    pub fn perspective_rotate_y(radians: f32, cx: f32, cy: f32, depth: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        let inv_depth = if depth > 0.0 && depth.is_finite() {
            sin / depth
        } else {
            0.0
        };
        // x' = cx + u·cos / w, y' = cy + v / w, w = 1 - u·sin/depth, with
        // u = x - cx, v = y - cy — cleared of the division by writing each
        // numerator·w as a linear form, which is what makes it a homography.
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

    /// The x-axis twin of [`Self::perspective_rotate_y`]: rotation about
    /// the horizontal axis through `y = cy`, seen in perspective.
    ///
    /// Positive angles bring the element's bottom edge toward the viewer
    /// and lay the top away — a card falling backward. Conventions and the
    /// `depth` parameter are as for the y version.
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

    /// Where an element-local point lands under this transform.
    ///
    /// Meaningful only on the `hw > 0` side of the horizon — see the type
    /// docs; a degenerate `hw` is clamped rather than divided by zero.
    #[must_use]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.matrix;
        let hx = m[0][0] * x + m[1][0] * y + m[2][0];
        let hy = m[0][1] * x + m[1][1] * y + m[2][1];
        let hw = m[0][2] * x + m[1][2] * y + m[2][2];
        let hw = if hw.abs() < 1e-6 { 1e-6 } else { hw };
        (hx / hw, hy / hw)
    }

    /// Whether this is exactly the identity — the test a backend makes when
    /// deciding whether an element can take a fast path.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// Whether the bottom row is `(0, 0, 1)`: no perspective, straight
    /// lines stay parallel, and nothing ever divides. What separates a
    /// rotation a plane could conceivably still scan out from a projection
    /// nothing but a sampler can draw.
    // Exact on purpose, like the derived equality `is_identity` relies on:
    // the question is whether the transform was *built* with no perspective
    // — the constructors write literal zeros — not whether it is nearly flat.
    #[allow(clippy::float_cmp)]
    #[must_use]
    pub fn is_affine(&self) -> bool {
        self.matrix[0][2] == 0.0 && self.matrix[1][2] == 0.0 && self.matrix[2][2] == 1.0
    }
}

/// One quad, in output logical coordinates.
#[derive(Debug, Clone)]
pub struct SceneElement {
    /// What fills the quad.
    pub content: SceneContent,
    /// Destination rectangle in the output's *logical* pixels:
    /// (x, y, width, height). The renderer scales it — see [`Scene::scale`].
    ///
    /// Fractional, because animation needs positions between pixels: a window
    /// sliding or zooming over a second lands off the integer grid on almost
    /// every frame, and snapping it there is visible as judder.
    pub dst: (f64, f64, f64, f64),
    /// The element's own transform over that rectangle — identity for the
    /// ordinary case, a rotation or skew when an animation calls for one, a
    /// perspective flip when it calls for that. See [`ElementTransform`].
    pub transform: ElementTransform,
    /// A fragment effect to run on this element's pixels, or `None` to draw
    /// it plain — see [`ShaderEffect`].
    pub effect: Option<Arc<ShaderEffect>>,
    /// Whole-element opacity, `0.0..=1.0`, multiplied over whatever the
    /// content's own alpha says. `1.0` draws the content as committed; less is
    /// how a fade happens without the client's involvement.
    pub alpha: f32,
    /// The client has promised this quad is fully opaque, so the blend can be
    /// skipped. A promise, not a measurement: it descends from the client's
    /// `wl_surface.set_opaque_region`, and keeping it is the client's problem.
    pub opaque: bool,
}

impl SceneElement {
    /// The texture this element samples, if it is a textured one. A group
    /// has no texture of its own — its sub-elements have theirs.
    #[must_use]
    pub fn texture(&self) -> Option<&Arc<TextureImage>> {
        match &self.content {
            SceneContent::Texture { image, .. } => Some(image),
            SceneContent::Color(_) | SceneContent::Group(_) => None,
        }
    }
}

/// What fills a scene element's rectangle.
///
/// The fields only a texture can have — a crop, and a transform to undo —
/// live inside that arm rather than on the element, where a colour would have
/// to carry meaningless values for them.
#[derive(Debug, Clone)]
pub enum SceneContent {
    /// A texture, cropped and stretched into the destination.
    Texture {
        /// The texture to sample.
        image: Arc<TextureImage>,
        /// Source rectangle in texture pixels: (x, y, width, height).
        src: (f64, f64, f64, f64),
        /// How the client transformed its buffer, which the draw has to undo.
        transform: BufferTransform,
    },
    /// A solid fill, as straight (not premultiplied) `0xAARRGGBB`.
    ///
    /// First-class rather than a 1×1 texture stretched to fit, so a scrim, a
    /// border or the bell flash costs no upload and no cache entry.
    Color(u32),
    /// A sub-scene composed offscreen, then drawn as one quad — see
    /// [`SceneGroup`]. Shared, so the clone a frame publish makes is a
    /// pointer rather than a subtree.
    Group(Arc<SceneGroup>),
}

/// A sub-scene composed offscreen and drawn as one quad.
///
/// The primitive behind three things the flat element list cannot say
/// alone:
///
/// - **Group opacity.** Overlapping elements faded one by one show seams —
///   the lower shows through the upper's translucency. Composed here at
///   full opacity and faded as one quad, the stack fades as a unit.
/// - **Whole-output passes.** A group holding everything on the output,
///   stretched over it with an [`effect`](SceneElement::effect), is a
///   post-process pass: a workspace dissolve, a screen ripple.
/// - **Backdrop effects.** A group holding what lies beneath a translucent
///   surface is a backdrop that surface's own pixels cannot see. The
///   element's effect samples the group as its texture — multiple taps
///   included, which is how a blur kernel reads — and the surface then
///   draws over the result.
///
/// The group composes onto its own canvas, in its own logical space at the
/// scene's scale, and the result behaves exactly like a texture: `dst`
/// places and stretches it, `alpha` fades it without seams, `transform`
/// rotates it, `effect` shades it. Groups nest, to a depth the renderer
/// bounds rather than trusts.
///
/// Composed fresh on every draw — the price of the flexibility. A group is
/// for the moments that need one (the fade, the transition, the panel with
/// a blurred backdrop), not a container to leave around scene-wide when
/// nothing is animating.
#[derive(Debug)]
pub struct SceneGroup {
    /// The canvas size in logical pixels: the space `elements` are
    /// positioned in, `(0, 0)` at the canvas's top-left.
    pub size: (f64, f64),
    /// What the canvas starts as, straight `0xAARRGGBB` — usually 0, fully
    /// transparent, so the group is only its own elements. An opaque colour
    /// works too; a translucent one is premultiplied before it is written,
    /// so it composes correctly under the elements.
    pub background: u32,
    /// The elements, back to front, in canvas coordinates.
    pub elements: Vec<SceneElement>,
}

/// A fragment-stage effect a compositor attaches to an element.
///
/// The escape hatch of the declarative vocabulary: rounded corners, dimming,
/// desaturation — per-pixel looks the fixed element fields cannot say. Kept
/// to a snippet rather than a whole program so the pipeline's own contract —
/// channel repair, premultiplied blending — stays in the backend's hands.
///
/// `source` is GLES 3.0 fragment code, spliced into the backend's shader at
/// global scope. It must define
///
/// ```glsl
/// vec4 effect(vec4 texel, vec2 uv)
/// ```
///
/// which receives the element's repaired texel — premultiplied, channels in
/// order — and the element-local `uv`, `(0, 0)` at the quad's top-left to
/// `(1, 1)` at its bottom-right regardless of crop or buffer transform, and
/// returns the premultiplied colour to draw. The element's alpha and tint
/// apply after, so a solid-colour element's effect sees opaque white:
/// shaping alpha works, inspecting the colour does not.
///
/// The snippet may declare its own uniforms at global scope and may read
/// `u_element_size`, the destination size in logical pixels. Values for
/// declared uniforms ride in [`Self::uniforms`] and are set on every draw.
///
/// Programs are cached by a hash of `source`: keep the source stable and
/// animate through uniforms — a source edited per frame is a program compile
/// per frame. A snippet that fails to compile draws its elements plain, and
/// the failure is reported once as
/// [`EffectCompileFailed`](crate::messages::BackendMessage::EffectCompileFailed).
///
/// An effect is opaque to backend reasoning in the way [`ElementTransform`]
/// data is not: an element carrying one can never take a scanout shortcut.
#[derive(Debug)]
pub struct ShaderEffect {
    /// A short name for humans, identifying the effect in logs and in
    /// [`EffectCompileFailed`](crate::messages::BackendMessage::EffectCompileFailed).
    pub name: String,
    /// The GLES 3.0 snippet defining `vec4 effect(vec4 texel, vec2 uv)`.
    pub source: Arc<str>,
    /// Values for the uniforms the snippet declared, by name. A name the
    /// snippet did not declare is ignored.
    pub uniforms: Vec<(String, EffectUniform)>,
}

/// A value for a uniform a [`ShaderEffect`] snippet declared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectUniform {
    /// A `float`.
    Float(f32),
    /// A `vec2`.
    Vec2(f32, f32),
    /// A `vec4`.
    Vec4(f32, f32, f32, f32),
}

/// One texture to draw, and everything the backend needs to get hold of it.
///
/// The fields here are the ones every kind of texture has, and they are what
/// the drawing path reads: what to cache it under, how big it is, and whether
/// its alpha means anything. Where the pixels actually come from — and what,
/// if anything, has to be sent to the GPU — is [`TextureSource`], because that
/// is the only part that differs.
///
/// Bytes, where there are any, are little-endian `0xAARRGGBB`, i.e.
/// `[B, G, R, A]` per pixel.
///
/// `serial` changes whenever the contents change under a stable `id`, which is
/// how the backend knows a cached texture needs re-uploading. For a dma-buf it
/// says only that the description changed, not the pixels: a client drawing
/// into a buffer the GPU already holds changes what is sampled without
/// anything crossing this boundary.
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
    /// Where the pixels come from. Always addressable in full, even when the
    /// damage on an upload is not, so a backend that cannot use the damage can
    /// fall back.
    pub source: TextureSource,
}

impl TextureImage {
    /// Row stride in pixels, for GL's `UNPACK_ROW_LENGTH`.
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

    /// Whether the source bytes are `[B, G, R, A]` and need swizzling when
    /// sampled.
    ///
    /// True for everything that goes up through `tex_image_2d`: GLES has no
    /// guaranteed BGRA upload format, so those bytes are uploaded as RGBA
    /// untouched and put right in the shader. An imported dma-buf is not
    /// uploaded at all — the driver is told the real format and samples it
    /// correctly — so swizzling one would undo what the import got right.
    #[must_use]
    pub fn swizzle_bgra(&self) -> bool {
        matches!(self.source, TextureSource::Upload { .. })
    }

    /// The GPU buffer behind this image, if that is what it is.
    #[must_use]
    pub fn dmabuf(&self) -> Option<&Arc<DmabufImage>> {
        match &self.source {
            TextureSource::Dmabuf { image, .. } => Some(image),
            TextureSource::Upload { .. } => None,
        }
    }

    /// The fence gating this image's content, if its producer supplied one.
    /// The backend must not sample before it signals — see [`AcquireFence`].
    #[must_use]
    pub fn acquire_fence(&self) -> Option<&Arc<AcquireFence>> {
        match &self.source {
            TextureSource::Dmabuf { acquire, .. } => acquire.as_ref(),
            TextureSource::Upload { .. } => None,
        }
    }

    /// The cell the backend fills with a fence covering its reads of this
    /// image, if the producer asked for one — see [`ReleaseFence`].
    #[must_use]
    pub fn release_fence(&self) -> Option<&Arc<ReleaseFence>> {
        match &self.source {
            TextureSource::Dmabuf { release, .. } => release.as_ref(),
            TextureSource::Upload { .. } => None,
        }
    }

    /// Borrow the pixels, from the first byte of the image to the last.
    ///
    /// # Safety
    /// For a mapped image this borrows a client's shm mapping, which the client
    /// may write to whenever it is allowed to. What makes reading it sound is
    /// the protocol: a client must not touch a committed buffer until it is
    /// released, and the compositor holds `wl_buffer.release` back until every
    /// `TextureImage` borrowing it has been dropped. The caller must therefore
    /// not keep the slice past the life of this image.
    ///
    /// Returns `None` if the image does not fit its mapping, which should be
    /// impossible — the extent is checked when the image is built — but is
    /// worth failing softly rather than reading out of bounds.
    #[must_use]
    pub unsafe fn bytes(&self) -> Option<&[u8]> {
        // On the GPU already; there is nothing here to read.
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
                // The last row needs no stride padding, so the extent is
                // shorter than `rows * stride`.
                let len = rows.checked_sub(1)?.checked_mul(*stride)? + width_bytes;
                // SAFETY: delegated to this function's own contract.
                unsafe { guard.mapping().slice(*offset, len) }
            }
        }
    }
}

/// How a client has already transformed its buffer, from `wl_surface.set_buffer_transform`.
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

    /// Whether this transform exchanges the buffer's width and height.
    #[must_use]
    pub fn swaps_axes(self) -> bool {
        matches!(
            self,
            Self::Rotate90 | Self::Rotate270 | Self::FlippedRotate90 | Self::FlippedRotate270
        )
    }

    /// Where a point on the surface reads from in the buffer, as an affine map
    /// over unit coordinates: `(origin, basis)` such that
    /// `source = origin + basis * destination`.
    ///
    /// `basis` is column-major, matching how GL reads a `mat2`: `basis[0]`
    /// scales the destination's x and `basis[1]` its y. Writing it row-major
    /// transposes the two quarter turns and leaves the symmetric transforms
    /// looking correct, which is a bug that hides well.
    ///
    /// This is the *inverse* of what the client did, because the client's value
    /// describes the transform it already applied and the compositor's job is
    /// to undo it. The flipped variants are a mirror about the vertical axis
    /// followed by the rotation, so their inverses are the rotation's inverse
    /// followed by the mirror — which is where the axis swaps come from.
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

/// Identity of a texture, used by the backend to cache GPU textures across
/// frames. Stable for as long as the underlying resource lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureId {
    /// A client `wl_buffer`, keyed by (`client_id`, `buffer_id`).
    Buffer(u32, u32),
    /// The cursor loaded from the system cursor theme.
    DefaultCursor,
    /// The built-in cursor used when no theme is available.
    FallbackCursor,
}

/// Pixel layout of a texture's source bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// `WL_SHM_FORMAT_ARGB8888` — premultiplied alpha.
    Argb8888,
    /// `WL_SHM_FORMAT_XRGB8888` — the high byte is undefined, treat as opaque.
    Xrgb8888,
}

/// Where a texture comes from, and what the backend has to do to get it.
///
/// The two arms differ in more than storage: an upload can be patched from the
/// copy the backend already holds, which is what `previous_serial` and
/// `damage` describe. An imported buffer has no such notion — the client draws
/// into memory the texture already samples, and nothing crosses this boundary
/// when it does. Those fields therefore live here rather than on
/// [`TextureImage`], where they would have to be given a meaningless value for
/// half the images that exist.
#[derive(Debug)]
pub enum TextureSource {
    /// Bytes the backend has to send to the GPU.
    Upload {
        /// The bytes themselves.
        pixels: UploadPixels,
        /// The serial this image was derived from, if it was derived from one.
        ///
        /// `damage` is only meaningful relative to this. A backend holding a
        /// texture at exactly this serial can patch it; anything else — a
        /// texture it never had, or one several serials behind — has to take
        /// the whole image. This is what keeps partial uploads safe even
        /// though the backend evicts textures without telling anyone.
        previous_serial: Option<u64>,
        /// What changed since `previous_serial`. Empty means "all of it".
        ///
        /// The composer's side of the contract: every rectangle has positive
        /// width and height and lies within the image — clamped when the
        /// damage is recorded, not checked again at upload. A backend may
        /// skip a rectangle that breaks it, leaving those pixels stale.
        damage: Vec<TextureRect>,
    },
    /// Already on the GPU, shared as dma-buf descriptors. There are no bytes
    /// to send: the backend imports the descriptors as a texture and samples
    /// the client's own memory.
    ///
    /// Shared rather than owned so that the same buffer redrawn across frames
    /// is the same image to the backend, which is how it knows the import it
    /// already has is still good.
    ///
    /// Built when a client draws with a buffer it created through
    /// `zwp_linux_dmabuf_v1` — a global advertised only once the backend has
    /// said it can import one, so this variant exists exactly when there is a
    /// backend able to do something with it — or by the compositor itself,
    /// submitting content it rendered on its own context.
    Dmabuf {
        /// The buffer.
        image: Arc<DmabufImage>,
        /// Explicit sync: signals when the producer's writes have landed,
        /// and the backend waits on it before sampling. `None` means
        /// implicit sync — the kernel's ordering is trusted, which is
        /// correct on Mesa and is what plain `wl_buffer` commits get. Per
        /// submission rather than per buffer: the same buffer recommitted
        /// carries a new fence each time.
        acquire: Option<Arc<AcquireFence>>,
        /// The release half: a cell the backend fills with a fence covering
        /// its GPU reads, refreshed on every frame that samples the buffer.
        /// The producer waits on it — or forwards it to a client's syncobj
        /// release point — once the buffer is structurally free, before
        /// writing again. `None` when nobody needs one: CPU-side release
        /// (this image's `Arc` coming free) plus implicit sync is the
        /// default story. See [`ReleaseFence`].
        release: Option<Arc<ReleaseFence>>,
    },
}

#[cfg(test)]
mod tests {
    //! Tests for buffer transforms: how a client's declared rotation/flip maps
    //! destination coordinates back onto its buffer.

    // Every coordinate here is 0.0 or 1.0, so the casts to `i32` are exact.
    #![allow(clippy::cast_possible_truncation)]

    use super::BufferTransform;

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

    use super::ElementTransform;

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
}
