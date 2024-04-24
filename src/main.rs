#![allow(unused)]

pub mod geometry;
pub mod render;
pub mod spherical;

use anyhow::{Result as AnyResult, Context as _};
use rayon::prelude::*;
use glam::*;
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

pub fn render_image_cpu(screen_width: usize, screen_height: usize, field: &RadianceField) -> Vec<u8> {
    let mut image = Vec::with_capacity(screen_width * screen_height);

    kdam::par_tqdm!((0..screen_width * screen_height).into_par_iter(), desc = "Rendering")
        .map(|i| (i % screen_width, i / screen_width))
        .map(|(x, y)| vec2(
            ((2 * x) as f32 + 0.5) / (screen_width  - 1) as f32 - 1.0,
            ((2 * y) as f32 + 0.5) / (screen_height - 1) as f32 - 1.0,
        ))
        .map(|coord| render::get_color(
            coord, screen_width, screen_height, field, &Default::default(),
        ))
        .map(compact_color)
        .collect_into_vec(&mut image);

    let image = image.into_boxed_slice();
    let image: Box<[u8]> = bytemuck::allocation::cast_slice_box(image);
    
    image.into_vec()
}


#[tokio::main]
async fn main() -> AnyResult<()> {
    let screen_width = 2520;
    let screen_height = 1680;

    let field = bincode::deserialize::<RadianceField>(
        &tokio::fs::read("assets/model.bin").await?,
    )?;

    let image = render_image_cpu(screen_width, screen_height, &field);

    let file = std::fs::File::create("target/image.png")?;

    let buf_writer = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(
        buf_writer, screen_width as u32, screen_height as u32,
    );

    encoder.set_color(png::ColorType::Rgba);

    let mut writer = encoder.write_header()?;
    writer.write_image_data(bytemuck::cast_slice(&image))?; 

    Ok(())
}