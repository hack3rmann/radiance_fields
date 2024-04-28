pub mod geometry;
pub mod render_cpu;
pub mod spherical;
pub mod render_gpu;

use anyhow::Result as AnyResult;
use array_init::array_init;
use glam::*;
use render_cpu::{Camera, RenderConfiguration};
use spherical::RadianceField;



pub fn make_textures(field: &RadianceField) -> [Vec<u8>; 9] {
    array_init(|i| {
        let colors = field.cells.iter()
            .map(|cell| render_cpu::compact_color(vec4(
                cell.density,
                cell.sh_r[i],
                cell.sh_g[i],
                cell.sh_b[i],
            )))
            .collect::<Vec<_>>();

        bytemuck::allocation::cast_vec(colors)
    })
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
        let mut image = render_cpu::render_image_cpu(
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