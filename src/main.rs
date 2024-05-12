pub mod geometry;
pub mod render_cpu;
pub mod spherical;
pub mod render_gpu;
pub mod graphics;

use anyhow::Result as AnyResult;
use glam::*;
use graphics::{Camera, Color, RenderConfiguration};
use spherical::RadianceField;



pub fn make_textures(field: &RadianceField) -> [Vec<u8>; 9] {
    std::array::from_fn(|i| {
        let colors = field.cells.iter()
            .map(|cell| vec4(
                cell.density,
                cell.sh_r[i],
                cell.sh_g[i],
                cell.sh_b[i],
            ))
            .map(Color::from_vec4)
            .collect::<Vec<_>>();

        bytemuck::allocation::cast_vec(colors)
    })
}



#[tokio::main]
async fn main() -> AnyResult<()> {
    let screen_width = 512;
    let screen_height = 512;

    let field = bincode::deserialize::<RadianceField>(
        &tokio::fs::read("assets/model.bin").await?,
    )?;

    let ctx = render_gpu::GpuContext::new(render_gpu::GpuContextMode::Debug).await?;

    let cfg = RenderConfiguration {
        camera: Camera {
            theta: 3.0 * std::f32::consts::PI / 4.0,
            ..Default::default()
        },
        ..Default::default()
    };

    let image = render_gpu::render_gpu(screen_width, screen_height, &ctx, &field, &cfg);
    // let image = render_cpu::render_cpu(screen_width, screen_height, &field, &cfg);

    let file = std::fs::File::create("output/result.png")?;

    let buf_writer = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(
        buf_writer, screen_width as u32, screen_height as u32,
    );

    encoder.set_color(png::ColorType::Rgba);

    let mut writer = encoder.write_header()?;
    writer.write_image_data(bytemuck::cast_slice(&image))?; 

    Ok(())
}