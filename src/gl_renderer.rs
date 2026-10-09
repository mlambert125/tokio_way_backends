//! GL scene rasteriser
//!
//! Draws a `Scene` as textured quads through GLES 3.0, caching one GPU texture per
//! `TextureId`. Must be used only on the thread that owns the GL context.
//!
//! Client pixels arrive as little-endian `0xAARRGGBB`, i.e. `[B, G, R, A]`. GLES has no
//! guaranteed BGRA upload format, so the bytes go up as `RGBA` and the fragment shader
//! swizzles. A dma-buf is the exception — the driver is told the real format — hence `u_swizzle`.

use std::collections::HashMap;

use glow::HasContext;
use std::os::fd::AsFd;
use tracing::{debug, warn};

use crate::{
    dma::{AcquireFence, DmabufImage, ReleaseFence},
    dmabuf_import::{DmabufCapabilities, DmabufImportProbeResult},
    scene_graph::{
        BufferTransform, EffectUniform, PixelFormat, Scene, SceneContent, SceneElement, SceneGraph,
        SceneGroup, ShaderEffect, TextureId, TextureImage, TextureRect, TextureSource,
    },
};

use super::dmabuf_import::{DmabufImporter, EglImage};

/// Vertex shader: stretches the unit quad into its destination rectangle and hands
/// the fragment shader the matching corner of the source crop
const VERTEX_SHADER: &str = r"#version 300 es
precision highp float;

// A unit quad, scaled into place by the destination rectangle
layout(location = 0) in vec2 a_unit;

// Destination rect in output pixels: (x, y, width, height)
uniform vec4 u_dst;
// Output size in pixels, for the pixels-to-clip-space conversion
uniform vec2 u_viewport;
// Source rect in normalised texture coordinates: (u0, v0, u1, v1)
uniform vec4 u_src;
// Undoes the client's `wl_surface.set_buffer_transform`, as an affine map over the
// unit quad: source = u_uv_origin + u_uv_basis * destination
uniform vec2 u_uv_origin;
uniform mat2 u_uv_basis;
// The element's own homography, over element-local logical pixels:
// (hx, hy, hw) = u_el_matrix * (local, 1), landing at dst.xy + h.xy / h.w
uniform mat3 u_el_matrix;

out vec2 v_texcoord;
// The untransformed 0..1 position within the destination quad, for effects: unlike
// v_texcoord it ignores crop and buffer transform
out vec2 v_unit;

void main() {
    vec2 local = a_unit * u_dst.zw;
    vec3 h = u_el_matrix * vec3(local, 1.0);
    // The anchor rides in homogeneous space (scaled by h.z) so everything stays linear
    // per vertex; the one perspective divide is the hardware's, through gl_Position.w.
    // Dividing here would interpolate v_texcoord linearly and make the texture swim.
    vec2 anchored = h.xy + u_dst.xy * h.z;
    // Wayland's y axis grows downward, GL's grows upward
    gl_Position = vec4(2.0 * anchored.x / u_viewport.x - h.z,
                       h.z - 2.0 * anchored.y / u_viewport.y,
                       0.0, h.z);
    vec2 unit = u_uv_origin + u_uv_basis * a_unit;
    v_texcoord = mix(u_src.xy, u_src.zw, unit);
    v_unit = a_unit;
}
";

/// Fragment shader source, with an optional [`ShaderEffect`] snippet spliced in
///
/// Samples the surface texture, repairs the channel order and XRGB8888's undefined
/// high byte, then runs the effect on the repaired texel. Output stays premultiplied,
/// which is what the `ONE, ONE_MINUS_SRC_ALPHA` blend set up in `new` expects.
pub(crate) fn fragment_source(effect: Option<&str>) -> String {
    let declarations = effect.unwrap_or("");
    let apply = if effect.is_some() {
        "    texel = effect(texel, v_unit);\n"
    } else {
        ""
    };
    format!(
        r"#version 300 es
precision highp float;

in vec2 v_texcoord;
in vec2 v_unit;
uniform sampler2D u_texture;
// 1.0 when the source has no alpha channel (XRGB8888), 0.0 otherwise
uniform float u_ignore_alpha;
// 1.0 for a texture uploaded as RGBA but laid out [B, G, R, A], 0.0 for an imported dma-buf
uniform float u_swizzle;
// Premultiplied colour the repaired texel is multiplied by: element alpha, and — against
// the renderer's white texture — solid colours
uniform vec4 u_tint;
// The element's destination size in logical pixels, for snippets working in pixels
uniform vec2 u_element_size;

out vec4 f_color;

{declarations}

void main() {{
    vec4 sampled = texture(u_texture, v_texcoord);
    vec4 texel = mix(sampled, sampled.bgra, u_swizzle);
    // XRGB8888's high byte is undefined; force it opaque. Before the tint, so a faded
    // XRGB surface fades from opaque.
    texel = vec4(texel.rgb, mix(texel.a, 1.0, u_ignore_alpha));
{apply}    f_color = texel * u_tint;
}}
"
    )
}

/// Unit quad as a triangle strip: top-left, top-right, bottom-left, bottom-right
const QUAD: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

/// What `resolve_content` settles for one element before any GL is touched
#[derive(Clone, Copy)]
struct ResolvedContent {
    /// The texture the element samples
    texture: glow::Texture,
    /// Source rectangle in normalised texture coordinates, (u0, v0, u1, v1)
    src: [f32; 4],
    /// The client buffer transform to undo
    transform: BufferTransform,
    /// 1.0 when the sampled alpha is not to be believed
    ignore_alpha: f32,
    /// 1.0 when the bytes are `[B, G, R, A]` and need the shader swizzle
    swizzle: f32,
    /// Premultiplied tint: element alpha, and solid colours
    tint: [f32; 4],
    /// A group's canvas and the framebuffer it was composed through, deleted as soon as the element's draw is issued
    scratch: Option<(glow::Texture, glow::Framebuffer)>,
}

/// Where a pass is drawing: which framebuffer, and its physical extent
#[derive(Clone, Copy)]
struct Target {
    /// The framebuffer, `None` being the window's own
    framebuffer: Option<glow::Framebuffer>,
    /// Physical width in pixels
    width: u32,
    /// Physical height in pixels
    height: u32,
}

/// How deep groups may nest before the renderer refuses to recurse further
const MAX_GROUP_DEPTH: u32 = 8;

/// A group canvas extent in physical pixels: logical size at the scene's scale.
pub(crate) fn group_extent(logical: f64, scale: f64, max_texture_size: i32) -> u32 {
    let max = u32::try_from(max_texture_size.max(1)).unwrap_or(1);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let extent = (logical * scale).ceil().max(1.0) as u32;
    extent.min(max)
}

/// A cached texture
struct CachedTexture {
    /// The texture
    texture: glow::Texture,
    /// The imported `EGLImage` the texture samples, for a dma-buf.
    #[allow(dead_code)]
    imported: Option<EglImage>,
    /// Serial of the image last uploaded, so unchanged content is not re-sent
    serial: u64,
    /// Width of the texture
    width: i32,
    /// Height of the texture
    height: i32,
}

/// One linked program variant - the plain pipeline or one effect - and its uniform
/// locations. Each location is an `Option` because GL returns none for a uniform the
/// linker optimised out.
struct ProgramHandles {
    /// The linked program
    program: glow::Program,
    /// Destination rect, in output logical pixels
    u_dst: Option<glow::UniformLocation>,
    /// Output size in logical pixels, for the pixels-to-clip conversion
    u_viewport: Option<glow::UniformLocation>,
    /// Source crop, in normalised texture coordinates
    u_src: Option<glow::UniformLocation>,
    /// Buffer-transform map, undoing `wl_surface.set_buffer_transform`
    u_uv_origin: Option<glow::UniformLocation>,
    u_uv_basis: Option<glow::UniformLocation>,
    /// The element's own homography — see `ElementTransform`
    u_el_matrix: Option<glow::UniformLocation>,
    /// Opaque-alpha flag, 1.0 for XRGB8888
    u_ignore_alpha: Option<glow::UniformLocation>,
    /// Channel-order flag, 1.0 for anything uploaded from CPU bytes
    u_swizzle: Option<glow::UniformLocation>,
    /// Premultiplied tint: element alpha, and solid colours
    u_tint: Option<glow::UniformLocation>,
    /// Destination size in logical pixels, for effect snippets
    u_element_size: Option<glow::UniformLocation>,
    /// Locations of the uniforms an effect snippet declared, cached by name on first set
    custom: HashMap<String, Option<glow::UniformLocation>>,
}

impl ProgramHandles {
    /// Link the pipeline with an optional effect snippet spliced in
    ///
    /// # Safety
    /// The GL context must be current on this thread.
    ///
    /// # Errors
    /// Compile or link failure; for an effect snippet, the driver's log is returned.
    unsafe fn link(gl: &glow::Context, effect: Option<&str>) -> anyhow::Result<Self> {
        unsafe {
            let program = link_program(gl, &fragment_source(effect))?;
            gl.use_program(Some(program));
            if let Some(location) = gl.get_uniform_location(program, "u_texture") {
                gl.uniform_1_i32(Some(&location), 0);
            }
            Ok(Self {
                program,
                u_dst: gl.get_uniform_location(program, "u_dst"),
                u_viewport: gl.get_uniform_location(program, "u_viewport"),
                u_src: gl.get_uniform_location(program, "u_src"),
                u_uv_origin: gl.get_uniform_location(program, "u_uv_origin"),
                u_uv_basis: gl.get_uniform_location(program, "u_uv_basis"),
                u_el_matrix: gl.get_uniform_location(program, "u_el_matrix"),
                u_ignore_alpha: gl.get_uniform_location(program, "u_ignore_alpha"),
                u_swizzle: gl.get_uniform_location(program, "u_swizzle"),
                u_tint: gl.get_uniform_location(program, "u_tint"),
                u_element_size: gl.get_uniform_location(program, "u_element_size"),
                custom: HashMap::new(),
            })
        }
    }
}

/// Cache key for an effect: a hash of its source, so the same snippet in different `Arc`s shares one program
fn effect_key(source: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

/// The GL render pipeline: the programs, one quad, and the texture cache
///
/// Every field is a handle into the GL context, valid only while that context is
/// current on this thread — which is why `new` is unsafe, and why the context must
/// still be current when the renderer is dropped.
pub struct GlRenderer {
    /// Loaded GL entry points, tying every handle below to the context they came from
    gl: glow::Context,
    /// The plain pipeline: every element without an effect draws through it
    plain: ProgramHandles,
    /// One program per distinct effect snippet, keyed by [`effect_key`]. `None` records
    /// a snippet that failed to compile, so it is reported once and never retried.
    effects: HashMap<u64, Option<ProgramHandles>>,
    /// Compile failures not yet reported: (effect name, driver log)
    failures: Vec<(String, String)>,
    /// Release cells of the images sampled by the frame being drawn, all signalled with
    /// one end-of-frame fence. Per draw, so it is empty between them.
    pending_releases: Vec<std::sync::Arc<ReleaseFence>>,
    /// How deep in nested group passes the current draw is, checked against [`MAX_GROUP_DEPTH`]
    group_depth: u32,
    /// `GL_MAX_TEXTURE_SIZE`, read once: the ceiling on a group's canvas
    max_texture_size: i32,
    /// Vertex array holding the attribute layout for `QUAD`
    vao: glow::VertexArray,
    /// Buffer holding `QUAD` itself, kept so `Drop` can delete it
    vbo: glow::Buffer,
    /// One white pixel, kept forever: a solid-colour element is this texture under a tint
    white: glow::Texture,
    /// One GPU texture per `TextureId`, pruned by [`Self::prune_caches`]
    textures: HashMap<TextureId, CachedTexture>,
    /// The dma-buf import path, if this driver has one. `None` means client GPU buffers cannot be drawn.
    dmabuf_importer: Option<DmabufImporter>,
}

impl GlRenderer {
    /// Build the pipeline. `loader` resolves GL function pointers, and the caller must
    /// have made the context current first. `dmabuf` is the import path for client GPU
    /// buffers, built by the backend from its own EGL display.
    ///
    /// # Safety
    /// `loader` must return valid GL entry points for the current context, and that
    /// context must stay current for as long as the renderer is used.
    ///
    /// # Errors
    /// Return an error if gl could not be initialized
    pub unsafe fn new(
        loader: impl FnMut(&std::ffi::CStr) -> *const std::ffi::c_void,
        dmabuf: Option<DmabufImporter>,
    ) -> anyhow::Result<Self> {
        let gl = unsafe { glow::Context::from_loader_function_cstr(loader) };

        unsafe {
            let plain = ProgramHandles::link(&gl, None)?;

            let white = gl
                .create_texture()
                .map_err(|e| anyhow::anyhow!("failed to create white texture: {e}"))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(white));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST.cast_signed(),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST.cast_signed(),
            );
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8.cast_signed(),
                1,
                1,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&[0xff, 0xff, 0xff, 0xff])),
            );

            let vao = gl
                .create_vertex_array()
                .map_err(|e| anyhow::anyhow!("failed to create vertex array: {e}"))?;
            gl.bind_vertex_array(Some(vao));

            let vbo = gl
                .create_buffer()
                .map_err(|e| anyhow::anyhow!("failed to create vertex buffer: {e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck_cast(&QUAD), glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);

            gl.enable(glow::BLEND);
            gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
            // Rows are tightly packed at width * 4, so the default unpack alignment of 4 is wrong
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);

            debug!(
                "GL renderer ready: {} / {}",
                gl.get_parameter_string(glow::RENDERER),
                gl.get_parameter_string(glow::VERSION)
            );

            Ok(Self {
                group_depth: 0,
                max_texture_size: gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE),
                gl,
                plain,
                effects: HashMap::new(),
                failures: Vec::new(),
                pending_releases: Vec::new(),
                vao,
                vbo,
                white,
                textures: HashMap::new(),
                dmabuf_importer: dmabuf,
            })
        }
    }

    /// What this renderer can do with dma-bufs: the device imports land on, the formats
    /// it will take, and what came of actually trying one
    pub fn dmabuf_support(&self) -> DmabufCapabilities {
        let Some(importer) = self.dmabuf_importer.as_ref() else {
            return DmabufCapabilities {
                device: None,
                formats: Vec::new(),
                probe: DmabufImportProbeResult::Unsupported(
                    "no EGL dma-buf import path on this driver".into(),
                ),
            };
        };
        // SAFETY: the context is current on this thread for the life of the renderer
        let probe = unsafe { importer.self_test(&self.gl) };
        let formats = match probe {
            // Nothing importable is worth advertising: a client would only find out at commit time
            DmabufImportProbeResult::Failed(_) | DmabufImportProbeResult::Unsupported(_) => {
                Vec::new()
            }
            DmabufImportProbeResult::Passed | DmabufImportProbeResult::Untested(_) => {
                importer.formats()
            }
        };
        DmabufCapabilities {
            device: importer.render_node().cloned(),
            formats,
            probe,
        }
    }

    /// Whether this driver will take a client's buffer. The import is thrown away again;
    /// the buffer is imported for real when it is first drawn.
    pub fn verify_import(&self, image: &DmabufImage) -> bool {
        self.dmabuf_importer
            .as_ref()
            .is_some_and(|importer| importer.import(image).is_ok())
    }

    /// Draw a scene into the window's framebuffer
    ///
    /// `width`/`height` are the real drawable size, which can differ from the scene's own
    /// size while a resize is in flight; any surplus shows background.
    pub fn draw(&mut self, scene: &Scene, cursor: &[SceneElement], width: u32, height: u32) {
        self.draw_into(scene, cursor, None, width, height);
    }

    /// Draw a scene into a target - the window's framebuffer, or an offscreen one for
    /// readback. Groups compose through their own targets and restore this one behind them.
    fn draw_into(
        &mut self,
        scene: &Scene,
        cursor: &[SceneElement],
        framebuffer: Option<glow::Framebuffer>,
        width: u32,
        height: u32,
    ) {
        let scale = scene.scale.as_f64();
        let logical = (logical_extent(width, scale), logical_extent(height, scale));
        let target = Target {
            framebuffer,
            width,
            height,
        };
        unsafe {
            let gl = &self.gl;
            gl.bind_framebuffer(glow::FRAMEBUFFER, framebuffer);
            gl.viewport(0, 0, width.cast_signed(), height.cast_signed());
            let [r, g, b, a] = unpack_color(scene.background);
            gl.clear_color(r, g, b, a);
            gl.clear(glow::COLOR_BUFFER_BIT);

            gl.bind_vertex_array(Some(self.vao));
            gl.active_texture(glow::TEXTURE0);
        }

        // The scene, then the cursor on top of it, in one pass with no clear between
        self.draw_elements(&scene.elements, logical, scale, target);
        self.draw_elements(cursor, logical, scale, target);
        self.signal_release_fences();

        unsafe { self.gl.bind_vertex_array(None) };
    }

    /// Give every release cell the frame touched a fence covering its reads
    ///
    /// One end-of-frame fence serves them all, and each cell keeps only its newest.
    /// Best-effort: without the sync extensions the cells stay empty and producers
    /// fall back to implicit sync.
    fn signal_release_fences(&mut self) {
        let pending = std::mem::take(&mut self.pending_releases);
        if pending.is_empty() {
            return;
        }
        let Some(importer) = self.dmabuf_importer.as_ref() else {
            return;
        };
        let Some(fence) = (unsafe { importer.export_native_fence(&self.gl) }) else {
            return;
        };
        for release in pending {
            if let Ok(duplicate) = fence.try_clone() {
                release.replace(duplicate);
            }
        }
    }

    /// Draw a run of elements into the already-set-up pipeline
    ///
    /// Assumes the caller has bound the target, vertex array and viewport. `scale` and
    /// `target` ride along for the groups.
    fn draw_elements(
        &mut self,
        elements: &[SceneElement],
        viewport: (f64, f64),
        scale: f64,
        target: Target,
    ) {
        for element in elements {
            // Uploading, composing a group and compiling all happen before the borrows used to draw
            let Some(resolved) = self.resolve_content(element, scale, target) else {
                continue;
            };
            if let Some(effect) = &element.effect {
                self.ensure_effect_program(effect);
            }
            // A failed effect falls back to the plain pipeline
            let handles = match element.effect.as_ref().map(|e| effect_key(&e.source)) {
                Some(key) => match self.effects.get_mut(&key) {
                    Some(Some(handles)) => handles,
                    _ => &mut self.plain,
                },
                None => &mut self.plain,
            };
            let scratch = resolved.scratch;

            unsafe {
                draw_one(&self.gl, handles, element, viewport, resolved);

                if let Some((texture, framebuffer)) = scratch {
                    self.gl.delete_framebuffer(framebuffer);
                    self.gl.delete_texture(texture);
                }
            }
        }
    }

    /// Everything the kinds of content disagree about, settled before touching GL
    fn resolve_content(
        &mut self,
        element: &SceneElement,
        scale: f64,
        target: Target,
    ) -> Option<ResolvedContent> {
        match &element.content {
            SceneContent::Texture {
                image,
                src,
                transform,
            } => {
                let texture = self.upload(image)?;
                // Explicit sync: the wait must be queued before the draw that samples the
                // buffer. A producer that supplied no fence still has the kernel's, so
                // those are exported and waited on the same way.
                if let Some(fence) = image.acquire_fence() {
                    self.wait_acquire(fence);
                } else if let Some(dmabuf) = image.dmabuf() {
                    // Waited on the CPU, briefly and boundedly, never on the GPU: a queued
                    // GPU wait cannot be revoked, so a fence that never signals would wedge
                    // this context's queue and everything behind it.
                    if let Some(fence) = dmabuf.export_implicit_fence() {
                        wait_sync_file_blocking(fence.as_fd());
                    }
                }
                // The reverse promise: a producer that asked to hear when the reads are done
                // gets this frame's end fence.
                if let Some(release) = image.release_fence() {
                    self.pending_releases.push(std::sync::Arc::clone(release));
                }
                // Normalise the source crop against the texture it came from
                let (tw, th) = (f64::from(image.width), f64::from(image.height));
                let (sx, sy, sw, sh) = *src;
                // No alpha channel, or a client promising the surface is opaque. Either way
                // the sampled alpha is not to be believed.
                let opaque = image.format == PixelFormat::Xrgb8888 || element.opaque;
                let a = element.alpha;
                Some(ResolvedContent {
                    texture,
                    src: [
                        as_f32(sx / tw),
                        as_f32(sy / th),
                        as_f32((sx + sw) / tw),
                        as_f32((sy + sh) / th),
                    ],
                    transform: *transform,
                    ignore_alpha: f32::from(u8::from(opaque)),
                    swizzle: f32::from(u8::from(image.swizzle_bgra())),
                    tint: [a, a, a, a],
                    scratch: None,
                })
            }
            // The white texture under a premultiplied tint: the colour is the tint
            SceneContent::Color(argb) => Some(ResolvedContent {
                texture: self.white,
                src: [0.0, 0.0, 1.0, 1.0],
                transform: BufferTransform::Normal,
                ignore_alpha: 0.0,
                swizzle: 0.0,
                tint: premultiplied_tint(*argb, element.alpha),
                scratch: None,
            }),
            // A sub-scene, composed into its own canvas first. The canvas is then a texture
            // like any other.
            SceneContent::Group(group) => {
                let (texture, framebuffer) = self.render_group(group, scale, target)?;
                let a = element.alpha;
                Some(ResolvedContent {
                    texture,
                    // The offscreen pass rendered with the same y-flip the screen gets, so
                    // sampling v inverted puts its rows right way round.
                    src: [0.0, 1.0, 1.0, 0.0],
                    transform: BufferTransform::Normal,
                    ignore_alpha: f32::from(u8::from(element.opaque)),
                    swizzle: 0.0,
                    tint: [a, a, a, a],
                    scratch: Some((texture, framebuffer)),
                })
            }
        }
    }

    /// Compose a group onto its own canvas, and hand back the canvas
    ///
    /// A full sub-pass with its own framebuffer and viewport, restoring the parent's
    /// target behind it. Sized at the scene's scale, clamped to what the driver allows.
    fn render_group(
        &mut self,
        group: &SceneGroup,
        scale: f64,
        parent: Target,
    ) -> Option<(glow::Texture, glow::Framebuffer)> {
        if self.group_depth >= MAX_GROUP_DEPTH {
            warn!("scene group nested deeper than {MAX_GROUP_DEPTH}; not composing");
            return None;
        }
        let width = group_extent(group.size.0, scale, self.max_texture_size);
        let height = group_extent(group.size.1, scale, self.max_texture_size);
        // SAFETY: the context is current on this thread for the life of the renderer
        let (texture, framebuffer) = unsafe { create_offscreen_target(&self.gl, width, height)? };
        unsafe {
            let gl = &self.gl;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            gl.viewport(0, 0, width.cast_signed(), height.cast_signed());
            // Premultiplied, unlike the top-level clear: the canvas is blended over and sampled
            let [r, g, b, a] = unpack_color(group.background);
            gl.clear_color(r * a, g * a, b * a, a);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }

        self.group_depth += 1;
        self.draw_elements(
            &group.elements,
            group.size,
            scale,
            Target {
                framebuffer: Some(framebuffer),
                width,
                height,
            },
        );
        self.group_depth -= 1;

        // The pass around this one carries on where it left off
        unsafe {
            let gl = &self.gl;
            gl.bind_framebuffer(glow::FRAMEBUFFER, parent.framebuffer);
            gl.viewport(
                0,
                0,
                parent.width.cast_signed(),
                parent.height.cast_signed(),
            );
        }
        Some((texture, framebuffer))
    }

    /// Compile an effect's program if this is the first time it is seen. A failure is
    /// cached as `None`, so a broken snippet is compiled and reported once.
    fn ensure_effect_program(&mut self, effect: &ShaderEffect) {
        let key = effect_key(&effect.source);
        if self.effects.contains_key(&key) {
            return;
        }

        let handles = match unsafe { ProgramHandles::link(&self.gl, Some(&effect.source)) } {
            Ok(handles) => Some(handles),
            Err(e) => {
                warn!("shader effect {:?} failed to compile: {e}", effect.name);
                self.failures.push((effect.name.clone(), e.to_string()));
                None
            }
        };
        self.effects.insert(key, handles);
    }

    /// Compile failures since the last call: (effect name, driver log)
    pub fn take_effect_failures(&mut self) -> Vec<(String, String)> {
        std::mem::take(&mut self.failures)
    }

    /// Make sure the GPU will not sample a fenced image before its producer finishes writing it
    ///
    /// Queued once per fence. A GPU-side wait through EGL when the driver has one; a
    /// driver without gets a bounded CPU wait on this thread instead.
    fn wait_acquire(&self, fence: &AcquireFence) {
        if !fence.needs_wait() {
            return;
        }
        let gpu_waited = self
            .dmabuf_importer
            .as_ref()
            .is_some_and(|importer| unsafe { importer.wait_native_fence(fence.fd()) });
        if !gpu_waited && !fence.wait_blocking(1000) {
            warn!("an acquire fence did not signal within 1s; sampling anyway");
        }
    }

    /// Return the GPU texture for an image, uploading whatever part of it the GPU does not already have
    ///
    /// A full upload happens when there is nothing to patch: no cached texture, one at a
    /// different serial, a change of dimensions, or damage that could not be described.
    /// Otherwise only the damaged rectangles go up.
    fn upload(&mut self, image: &TextureImage) -> Option<glow::Texture> {
        let cached = self.textures.get(&image.id);
        if let Some(cached) = cached
            && cached.serial == image.serial
        {
            return Some(cached.texture);
        }

        let TextureSource::Upload {
            previous_serial,
            damage,
            ..
        } = &image.source
        else {
            // A GPU buffer is imported, not uploaded
            return self.import(image);
        };

        let patchable = cached.is_some_and(|cached| {
            Some(cached.serial) == *previous_serial
                && cached.width == image.width
                && cached.height == image.height
        }) && !damage.is_empty();

        let gl = &self.gl;
        unsafe {
            if patchable {
                let texture = self.textures[&image.id].texture;
                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                for rect in damage {
                    upload_region(gl, image, *rect);
                }
                self.textures.insert(
                    image.id,
                    CachedTexture {
                        texture,
                        imported: None,
                        serial: image.serial,
                        width: image.width,
                        height: image.height,
                    },
                );
                return Some(texture);
            }

            if let Some(old) = self.textures.remove(&image.id) {
                gl.delete_texture(old.texture);
            }

            let texture = match gl.create_texture() {
                Ok(t) => t,
                Err(e) => {
                    warn!("failed to create texture: {e}");
                    return None;
                }
            };
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR.cast_signed(),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR.cast_signed(),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE.cast_signed(),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE.cast_signed(),
            );

            let Some(bytes) = image.bytes() else {
                warn!("image {:?} does not fit its mapping", image.id);
                gl.delete_texture(texture);
                return None;
            };
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, image.row_length());
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8.cast_signed(),
                image.width,
                image.height,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(bytes)),
            );
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);

            self.textures.insert(
                image.id,
                CachedTexture {
                    texture,
                    imported: None,
                    serial: image.serial,
                    width: image.width,
                    height: image.height,
                },
            );
            Some(texture)
        }
    }

    /// Import a client's GPU buffer and give back a texture that samples it
    fn import(&mut self, image: &TextureImage) -> Option<glow::Texture> {
        let dmabuf = image.dmabuf()?;
        let Some(importer) = self.dmabuf_importer.as_ref() else {
            warn!("no dma-buf import path: cannot draw texture {:?}", image.id);
            return None;
        };
        let egl_image = match importer.import(dmabuf) {
            Ok(image) => image,
            Err(e) => {
                warn!("failed to import dma-buf for texture {:?}: {e}", image.id);
                return None;
            }
        };

        let gl = &self.gl;

        unsafe {
            if let Some(old) = self.textures.remove(&image.id) {
                gl.delete_texture(old.texture);
            }
            let texture = match gl.create_texture() {
                Ok(t) => t,
                Err(e) => {
                    warn!("failed to create texture: {e}");
                    return None;
                }
            };
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR.cast_signed(),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR.cast_signed(),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE.cast_signed(),
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE.cast_signed(),
            );
            importer.bind_to_texture(&egl_image);

            self.textures.insert(
                image.id,
                CachedTexture {
                    texture,
                    imported: Some(egl_image),
                    serial: image.serial,
                    width: image.width,
                    height: image.height,
                },
            );
            Some(texture)
        }
    }

    /// Render a scene offscreen and read the pixels back to the CPU
    ///
    /// Rows come back top-down, `[R, G, B, A]` per pixel, premultiplied. Rebinds the
    /// default framebuffer before returning, so a caller mid-frame must redraw. `None`
    /// when the extent is empty or the driver refuses the offscreen target.
    pub fn render_to_cpu(
        &mut self,
        scene: &Scene,
        cursor: &[SceneElement],
        width: u32,
        height: u32,
    ) -> Option<Vec<u8>> {
        if width == 0 || height == 0 {
            return None;
        }
        let (texture, framebuffer) = unsafe { create_offscreen_target(&self.gl, width, height)? };

        self.draw_into(scene, cursor, Some(framebuffer), width, height);

        let stride = width as usize * 4;
        let mut pixels = vec![0u8; stride * height as usize];
        unsafe {
            let gl = &self.gl;
            gl.read_pixels(
                0,
                0,
                width.cast_signed(),
                height.cast_signed(),
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut pixels)),
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.delete_framebuffer(framebuffer);
            gl.delete_texture(texture);
        }

        // GL reads rows bottom-up; an image is top-down
        let rows = height as usize;
        for y in 0..rows / 2 {
            let (top, bottom) = pixels.split_at_mut((rows - 1 - y) * stride);
            top[y * stride..(y + 1) * stride].swap_with_slice(&mut bottom[..stride]);
        }
        Some(pixels)
    }

    /// Drop the cached texture and effect program for everything the frame did not reference
    ///
    /// Takes the whole frame rather than one scene: with more than one output the same
    /// texture can appear in one scene and not another.
    pub fn prune_caches(&mut self, frame: &SceneGraph) {
        let mut live_textures = std::collections::HashSet::new();
        let mut live_effects = std::collections::HashSet::new();
        for scene in &frame.scenes {
            collect_live(&scene.elements, &mut live_textures, &mut live_effects);
        }
        collect_live(
            &frame.cursor.elements,
            &mut live_textures,
            &mut live_effects,
        );
        let gl = &self.gl;
        self.textures.retain(|id, cached| {
            if live_textures.contains(id) {
                return true;
            }
            unsafe { gl.delete_texture(cached.texture) };
            false
        });
        // Failed entries go too. A fixed snippet hashes differently anyway
        self.effects.retain(|key, handles| {
            if live_effects.contains(key) {
                return true;
            }
            if let Some(handles) = handles {
                unsafe { gl.delete_program(handles.program) };
            }
            false
        });
    }
}

/// Gather every texture id and effect key a run of elements references, descending into groups
fn collect_live(
    elements: &[SceneElement],
    textures: &mut std::collections::HashSet<TextureId>,
    effects: &mut std::collections::HashSet<u64>,
) {
    for element in elements {
        if let Some(image) = element.texture() {
            textures.insert(image.id);
        }
        if let Some(effect) = &element.effect {
            effects.insert(effect_key(&effect.source));
        }
        if let SceneContent::Group(group) = &element.content {
            collect_live(&group.elements, textures, effects);
        }
    }
}

/// A texture and the framebuffer rendering into it, completeness-checked and left bound.
/// The canvas for a group, and the target for a readback.
///
/// # Safety
/// The GL context must be current on this thread.
unsafe fn create_offscreen_target(
    gl: &glow::Context,
    width: u32,
    height: u32,
) -> Option<(glow::Texture, glow::Framebuffer)> {
    unsafe {
        let texture = gl.create_texture().ok()?;
        let framebuffer = match gl.create_framebuffer() {
            Ok(framebuffer) => framebuffer,
            Err(e) => {
                warn!("failed to create offscreen framebuffer: {e}");
                gl.delete_texture(texture);
                return None;
            }
        };
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR.cast_signed(),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR.cast_signed(),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE.cast_signed(),
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE.cast_signed(),
        );
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8.cast_signed(),
            width.cast_signed(),
            height.cast_signed(),
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(None),
        );
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(texture),
            0,
        );
        if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
            warn!("offscreen framebuffer is incomplete");
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.delete_framebuffer(framebuffer);
            gl.delete_texture(texture);
            return None;
        }
        Some((texture, framebuffer))
    }
}

/// Bind one element's program, texture and uniforms, and draw its quad
///
/// Free of `GlRenderer` so the borrow of one program's handles can coexist with the
/// borrow of the context.
///
/// # Safety
/// The GL context must be current on this thread, and `handles` must belong to a live
/// program of that context.
unsafe fn draw_one(
    gl: &glow::Context,
    handles: &mut ProgramHandles,
    element: &SceneElement,
    viewport: (f64, f64),
    resolved: ResolvedContent,
) {
    let ResolvedContent {
        texture,
        src,
        transform,
        ignore_alpha,
        swizzle,
        tint,
        scratch: _,
    } = resolved;
    let program = handles.program;

    unsafe {
        gl.use_program(Some(program));
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));

        gl.uniform_2_f32(
            handles.u_viewport.as_ref(),
            as_f32(viewport.0),
            as_f32(viewport.1),
        );
        let (dx, dy, dw, dh) = element.dst;
        gl.uniform_4_f32(
            handles.u_dst.as_ref(),
            as_f32(dx),
            as_f32(dy),
            as_f32(dw),
            as_f32(dh),
        );

        gl.uniform_4_f32(handles.u_src.as_ref(), src[0], src[1], src[2], src[3]);

        let ((ox, oy), basis) = transform.uv_map();
        gl.uniform_2_f32(handles.u_uv_origin.as_ref(), ox, oy);
        // Column-major, which is what GL expects: each inner array is one column
        gl.uniform_matrix_2_f32_slice(
            handles.u_uv_basis.as_ref(),
            false,
            &[basis[0][0], basis[0][1], basis[1][0], basis[1][1]],
        );

        // The element's own homography, in the same column-major convention
        let el = element.transform;
        gl.uniform_matrix_3_f32_slice(
            handles.u_el_matrix.as_ref(),
            false,
            &[
                el.matrix[0][0],
                el.matrix[0][1],
                el.matrix[0][2],
                el.matrix[1][0],
                el.matrix[1][1],
                el.matrix[1][2],
                el.matrix[2][0],
                el.matrix[2][1],
                el.matrix[2][2],
            ],
        );

        gl.uniform_1_f32(handles.u_ignore_alpha.as_ref(), ignore_alpha);
        gl.uniform_1_f32(handles.u_swizzle.as_ref(), swizzle);
        gl.uniform_4_f32(handles.u_tint.as_ref(), tint[0], tint[1], tint[2], tint[3]);
        gl.uniform_2_f32(handles.u_element_size.as_ref(), as_f32(dw), as_f32(dh));

        // The snippet's own uniforms, locations cached by name per program
        if let Some(effect) = &element.effect {
            for (name, value) in &effect.uniforms {
                let location = handles
                    .custom
                    .entry(name.clone())
                    .or_insert_with(|| gl.get_uniform_location(program, name));
                match *value {
                    EffectUniform::Float(v) => gl.uniform_1_f32(location.as_ref(), v),
                    EffectUniform::Vec2(x, y) => gl.uniform_2_f32(location.as_ref(), x, y),
                    EffectUniform::Vec4(x, y, z, w) => {
                        gl.uniform_4_f32(location.as_ref(), x, y, z, w);
                    }
                }
            }
        }

        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
    }
}

impl Drop for GlRenderer {
    /// Drop implementation for this `GlRenderer`, cleans up Gl resources
    fn drop(&mut self) {
        unsafe {
            for cached in self.textures.values() {
                self.gl.delete_texture(cached.texture);
            }
            for handles in self.effects.values().filter_map(Option::as_ref) {
                self.gl.delete_program(handles.program);
            }
            self.gl.delete_texture(self.white);
            self.gl.delete_buffer(self.vbo);
            self.gl.delete_vertex_array(self.vao);
            self.gl.delete_program(self.plain.program);
        }
    }
}

/// Creates and links a GL program from the fixed vertex shader and the given fragment source
unsafe fn link_program(gl: &glow::Context, fragment: &str) -> anyhow::Result<glow::Program> {
    unsafe {
        let program = gl
            .create_program()
            .map_err(|e| anyhow::anyhow!("failed to create program: {e}"))?;

        let mut shaders = Vec::new();
        for (kind, source) in [
            (glow::VERTEX_SHADER, VERTEX_SHADER),
            (glow::FRAGMENT_SHADER, fragment),
        ] {
            let shader = gl
                .create_shader(kind)
                .map_err(|e| anyhow::anyhow!("failed to create shader: {e}"))?;
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                return Err(anyhow::anyhow!("shader failed to compile: {log}"));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            return Err(anyhow::anyhow!("program failed to link: {log}"));
        }
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        Ok(program)
    }
}

/// Upload a damaged rectangle out of an image's pixel buffer
unsafe fn upload_region(gl: &glow::Context, image: &TextureImage, rect: TextureRect) {
    if rect.width <= 0 || rect.height <= 0 {
        return;
    }
    let row_length = image.row_length().unsigned_abs() as usize;
    let Some(bytes) = (unsafe { image.bytes() }) else {
        warn!("image {:?} does not fit its mapping", image.id);
        return;
    };
    let start = (rect.y.unsigned_abs() as usize * row_length + rect.x.unsigned_abs() as usize) * 4;
    let len = ((rect.height.unsigned_abs() as usize - 1) * row_length
        + rect.width.unsigned_abs() as usize)
        * 4;
    let Some(pixels) = bytes.get(start..start + len) else {
        // Should be unreachable: the rectangle was clamped when the damage was recorded.
        // Skipping leaves the region stale, which beats reading out of bounds.
        warn!("damage rectangle {rect:?} outside image {:?}", image.id);
        return;
    };

    unsafe {
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, image.row_length());
        gl.tex_sub_image_2d(
            glow::TEXTURE_2D,
            0,
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(pixels)),
        );
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
    }
}

/// A physical framebuffer extent divided by the output scale, in logical pixels, never below one
/// Wait CPU-side for a sync file to signal, briefly and boundedly
///
/// The cap is half a frame: a producer whose fence does not signal by then is sampled
/// anyway. Deliberately never a GPU-side wait; see the caller.
fn wait_sync_file_blocking(fence: std::os::fd::BorrowedFd<'_>) {
    let mut poll = libc::pollfd {
        fd: std::os::fd::AsRawFd::as_raw_fd(&fence),
        events: libc::POLLIN,
        revents: 0,
    };
    unsafe {
        libc::poll(&raw mut poll, 1, 8);
    }
}

fn logical_extent(physical: u32, scale: f64) -> f64 {
    (f64::from(physical) / scale).max(1.0)
}

/// Narrow a coordinate to the `f32` that GL uniforms take
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn as_f32(value: impl Into<f64>) -> f32 {
    value.into() as f32
}

/// Split an `0xAARRGGBB` word into GL's normalised float channels
fn unpack_color(argb: u32) -> [f32; 4] {
    let channel = |shift: u32| f32::from(((argb >> shift) & 0xff) as u8) / 255.0;
    [channel(16), channel(8), channel(0), channel(24)]
}

/// Turn a straight `0xAARRGGBB` colour and an element alpha into the premultiplied tint
/// the shader multiplies in
fn premultiplied_tint(argb: u32, alpha: f32) -> [f32; 4] {
    let [r, g, b, a] = unpack_color(argb);
    let a = a * alpha;
    [r * a, g * a, b * a, a]
}

/// Reinterpret the quad's floats as the bytes `buffer_data_u8_slice` wants
fn bytemuck_cast(floats: &[f32; 8]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(floats.as_ptr().cast::<u8>(), std::mem::size_of_val(floats))
    }
}
