use serde::{Serialize, Deserialize};
use glam::*;
use crate::{geometry, spherical::{RadianceField, Filtering, CellValue}};



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
    n_steps: usize,
}

impl Default for RaymarchSettings {
    fn default() -> Self {
        Self { n_steps: 100 }
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
        let CellValue { color: cur_color, density } = get_info(pos, rd);

        if density <= 0.0 { continue }

        let cur_color = cur_color.clamp(Vec3::ZERO, Vec3::ONE);

        color += cur_color * (1.0 - f32::exp(-density * step_size)) * f32::exp(-density_sum);
        density_sum += step_size * density;
    }

    color
}



#[derive(Clone, Debug, PartialEq, Copy)]
#[derive(Serialize, Deserialize)]
pub struct Camera {
    distance: f32,
    theta: f32,
    phi: f32,
    target_pos: Vec3,
    vfov: f32,
}

impl Camera {
    pub const DEFAULT_DISTANCE: f32 = 1.3;
    pub const VALID_THETAS: [f32; 7] = [0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4];
}

impl Default for Camera {
    fn default() -> Self {
        let distance = Self::DEFAULT_DISTANCE;

        Self {
            distance,
            theta: Self::VALID_THETAS[0],
            phi: 1.0 * std::f32::consts::FRAC_PI_2,
            target_pos: Vec3::ZERO,
            vfov: (1.0 / distance) * std::f32::consts::FRAC_PI_3,
        }
    }
}



#[derive(Clone, Debug, PartialEq, Default, Copy)]
#[derive(Serialize, Deserialize)]
pub struct RenderConfiguration {
    camera: Camera,
    rm_settings: RaymarchSettings,
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

    let Some((near, far)) = geometry::intersect_ray_box(
        ray_origin, ray_direction, -0.5 * Vec3::ONE, 0.5 * Vec3::ONE,
    ) else { return Vec3::ZERO };

    let color_fn = |ro: Vec3, rd: Vec3| -> CellValue {
        field.eval(ro + 0.5, rd, Filtering::Trilinear).unwrap_or_default()
    };

    let color = raymarch(ray_origin, ray_direction, near, far, color_fn, cfg.rm_settings);

    color.powf(0.4545)
}