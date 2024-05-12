use crate::{
    benchmark::Bench, geometry::Intersect as _, graphics::{
        Color, RaymarchSettings, RenderConfiguration, RENDER_TARGET_COLOR, RENDER_TARGET_DENSITY
    }, spherical::{CellValue, Filtering, RadianceField}
};
use glam::*;
use rayon::prelude::*;



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
        let cell = get_info(pos, rd);

        color += cell.color
            * f32::exp(-density_sum)
            * (1.0 - f32::exp(-cell.density * step_size));

        density_sum += step_size * cell.density;
    }

    color
}



pub fn get_color(
    screen_coord: Vec2, screen_width: usize, screen_height: usize,
    field: &RadianceField, cfg: &RenderConfiguration,
) -> Vec3 {
    let aspect_ratio = screen_height as f32 / screen_width as f32;

    let ray = cfg.camera.shoot_ray(screen_coord, aspect_ratio);

    let Some((near, far)) = cfg.bounding_box.intersect(&ray) else {
        return Vec3::ZERO;
    };

    let color_fn = |ro: Vec3, rd: Vec3| -> CellValue {
        let mut value = field.eval(ro + 0.5, rd, Filtering::Trilinear).unwrap_or_default();

        value.density = value.density.max(0.0);
        value.color = match cfg.render_target {
            RENDER_TARGET_COLOR
                => value.color.clamp(Vec3::ZERO, Vec3::ONE),
            RENDER_TARGET_DENSITY
                => Vec3::splat(value.density),
            _ => panic!("Invalid render target '{}'", cfg.render_target),
        };

        value
    };

    raymarch(ray.origin, ray.direction, near.max(0.0), far, color_fn, cfg.rm_settings)
}

pub fn render_multicpu(
    screen_width: usize, screen_height: usize,
    field: &RadianceField, cfg: &RenderConfiguration, bench: &mut Bench,
) -> Vec<u8> {
    let mut image = Vec::with_capacity(screen_width * screen_height);

    bench.render.start();

    kdam::par_tqdm!((0..screen_width * screen_height).into_par_iter(), desc = "Rendering")
        .map(|i| (i % screen_width, i / screen_width))
        .map(|(x, y)| vec2(
            ((2 * x) as f32 + 0.5) / (screen_width  - 1) as f32 - 1.0,
            ((2 * y) as f32 + 0.5) / (screen_height - 1) as f32 - 1.0,
        ))
        .map(|coord| get_color(
            coord, screen_width, screen_height, field, cfg,
        ).extend(1.0))
        .map(Color::from_vec4)
        .collect_into_vec(&mut image);
    
    println!();

    bench.render.end();

    bytemuck::allocation::cast_vec(image)
}

pub fn render_singlecpu(
    screen_width: usize, screen_height: usize,
    field: &RadianceField, cfg: &RenderConfiguration, bench: &mut Bench,
) -> Vec<u8> {
    bench.render.start();

    let image = kdam::tqdm!(0..screen_width * screen_height, desc = "Rendering")
        .map(|i| (i % screen_width, i / screen_width))
        .map(|(x, y)| vec2(
            ((2 * x) as f32 + 0.5) / (screen_width  - 1) as f32 - 1.0,
            ((2 * y) as f32 + 0.5) / (screen_height - 1) as f32 - 1.0,
        ))
        .map(|coord| get_color(
            coord, screen_width, screen_height, field, cfg,
        ).extend(1.0))
        .map(Color::from_vec4)
        .collect();

    println!();

    bench.render.end();

    bytemuck::allocation::cast_vec(image)
}