#![allow(unused)]

pub mod geometry;
pub mod render;
pub mod spherical;

use anyhow::{Result as AnyResult, Context as _};
use rayon::prelude::*;
use glam::*;
use render::{Camera, RenderConfiguration};
use spherical::RadianceField;



pub fn compact_color(mut color: Vec3) -> [u8; 4] {
    color = Vec3::clamp(255.0 * color, Vec3::ZERO, 255.0 * Vec3::ONE);

    [
        color.x as u8,
        color.y as u8,
        color.z as u8,
        u8::MAX,
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
        .map(|coord| render::get_color(
            coord, screen_width, screen_height, field, cfg,
        ))
        .map(compact_color)
        .collect_into_vec(&mut image);

    bytemuck::allocation::cast_vec(image)
}



#[tokio::main]
async fn main() -> AnyResult<()> {
    let screen_width = 1024;
    let screen_height = 1024;

    let field = bincode::deserialize::<RadianceField>(
        &tokio::fs::read("assets/model.bin").await?,
    )?;

    let mut video_file = std::fs::File::create("output/result.gif")?;
    let mut encoder = gif::Encoder::new(
        &mut video_file,
        screen_width as u16,
        screen_height as u16,
        &[],
    )?;

    encoder.set_repeat(gif::Repeat::Infinite)?;

    const N_VIEWS: usize = 168;

    for i in kdam::tqdm!(0..N_VIEWS, desc = "Rotating model") {
        let mut image = render_image_cpu(
            screen_width, screen_height, &field,
            &RenderConfiguration {
                camera: Camera {
                    theta: -(i as f32 / N_VIEWS as f32 + 0.5) * 4.0 * std::f32::consts::PI,
                    phi: (i as f32 / N_VIEWS as f32 - 0.25) * 2.0 * std::f32::consts::PI,
                    distance: 1.0 / ((i as f32 / N_VIEWS as f32) * 2.0 * std::f32::consts::PI).powi(2),
                    target_pos: vec3(0.0, 0.15, 0.2),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let mut frame = gif::Frame::from_rgba(
            screen_width as u16, screen_height as u16, &mut image,
        );

        frame.delay = 3;

        encoder.write_frame(&frame)?;
    }

    println!("\n");

    Ok(())
}