use crate::{geometry, spherical::{RadianceField, Filtering, CellValue}};
use serde::{Serialize, Deserialize};
use bytemuck::{Pod, Zeroable};
use glam::*;
use rayon::prelude::*;



pub fn spherical_to_cartesian(radius: f32, theta: f32, phi: f32) -> Vec3 {
    radius * Vec3::new(
        phi.sin() * theta.sin(),
        phi.cos(),
        phi.sin() * theta.cos(),
    )
}



#[derive(Clone, Debug, PartialEq, Copy, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct RaymarchSettings {
    pub n_steps: usize,
}

impl Default for RaymarchSettings {
    fn default() -> Self {
        Self { n_steps: 300 }
    }
}

pub fn raymarch(
    ro: Vec3, rd: Vec3, near: f32, far: f32,
    mut get_info: impl FnMut(Vec3, Vec3) -> CellValue,
    settings: RaymarchSettings,
) -> Vec3 {
    let step_size = (far - near) / settings.n_steps as f32;

    let positions = (0..settings.n_steps).map(|i| ro + rd * near.lerp(
        far, i as f32 / (settings.n_steps - 1) as f32,
    ));

    let mut color = Vec3::ZERO;
    let mut density_sum = 0.0;

    for pos in positions {
        let CellValue { color: cell_color, density } = get_info(pos, rd);

        color += cell_color
            * f32::exp(-density_sum)
            * (1.0 - f32::exp(-density * step_size));

        density_sum += step_size * density;
    }

    color
}



#[repr(C)]
#[derive(Clone, Debug, PartialEq, Copy)]
#[derive(Serialize, Deserialize)]
#[derive(Pod, Zeroable)]
pub struct Camera {
    pub distance: f32,
    pub theta: f32,
    pub phi: f32,
    pub vfov: f32,
    pub target_pos: Vec3,
}

impl Camera {
    pub const DEFAULT_DISTANCE: f32 = 1.3;
    pub const VALID_THETAS: [f32; 7] = [0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4];
    pub const DEFAULT_VFOV: f32 = std::f32::consts::FRAC_PI_4;
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            distance: Self::DEFAULT_DISTANCE,
            theta: Self::VALID_THETAS[0],
            phi: std::f32::consts::FRAC_PI_3,
            target_pos: Vec3::ZERO,
            vfov: Self::DEFAULT_VFOV,
        }
    }
}



#[derive(Clone, Debug, PartialEq, Default, Copy)]
#[derive(Serialize, Deserialize)]
pub struct RenderConfiguration {
    pub camera: Camera,
    pub rm_settings: RaymarchSettings,
}



pub fn get_color(
    screen_coord: Vec2, screen_width: usize, screen_height: usize,
    field: &RadianceField, cfg: &RenderConfiguration,
) -> Vec3 {
    let aspect_ratio = screen_height as f32 / screen_width as f32;
    
    let camera_pos = cfg.camera.target_pos + spherical_to_cartesian(
        cfg.camera.distance, cfg.camera.theta, cfg.camera.phi,
    );
    let camera_direction = Vec3::normalize(cfg.camera.target_pos - camera_pos);
    let camera_tangent = -cfg.camera.theta.sin() * Vec3::Z + cfg.camera.theta.cos() * Vec3::X;
    let camera_bitangent = Vec3::cross(camera_direction, camera_tangent);

    let fov_tan = f32::tan(0.5 * cfg.camera.vfov);
    let ray_direction = Vec3::normalize(camera_direction
        + (screen_coord.x / aspect_ratio) * fov_tan * camera_tangent
        + screen_coord.y * fov_tan * camera_bitangent
    );
    let ray_origin = camera_pos;

    let Some((mut near, far)) = geometry::intersect_ray_box(
        ray_origin, ray_direction, -0.5 * Vec3::ONE, 0.5 * Vec3::ONE,
    ) else { return Vec3::ZERO };

    near = near.max(0.0);

    let color_fn = |ro: Vec3, rd: Vec3| -> CellValue {
        let mut value = field.eval(ro + 0.5, rd, Filtering::Trilinear).unwrap_or_default();

        value.density = value.density.max(0.0);
        value.color = Vec3::clamp(value.color * 0.5 + 0.5, Vec3::ZERO, Vec3::ONE);

        value
    };

    raymarch(ray_origin, ray_direction, near, far, color_fn, cfg.rm_settings)
}

pub fn compact_color(mut color: Vec4) -> [u8; 4] {
    color = Vec4::clamp(255.0 * color, Vec4::ZERO, Vec4::splat(255.0));

    [
        color.x as u8,
        color.y as u8,
        color.z as u8,
        color.w as u8,
    ]
}

pub fn render_image_cpu(
    screen_width: usize, screen_height: usize,
    field: &RadianceField, cfg: &RenderConfiguration,
) -> Vec<u8> {
    let mut image = Vec::with_capacity(screen_width * screen_height);

    kdam::par_tqdm!((0..screen_width * screen_height).into_par_iter(), desc = "Rendering", position = 1)
        .map(|i| (i % screen_width, i / screen_width))
        .map(|(x, y)| vec2(
            ((2 * x) as f32 + 0.5) / (screen_width  - 1) as f32 - 1.0,
            ((2 * y) as f32 + 0.5) / (screen_height - 1) as f32 - 1.0,
        ))
        .map(|coord| get_color(
            coord, screen_width, screen_height, field, cfg,
        ).extend(1.0))
        .map(compact_color)
        .collect_into_vec(&mut image);

    bytemuck::allocation::cast_vec(image)
}