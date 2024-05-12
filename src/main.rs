pub mod geometry;
pub mod render_cpu;
pub mod spherical;
pub mod render_gpu;
pub mod graphics;
pub mod benchmark;

use anyhow::Result as AnyResult;
use glam::*;
use graphics::{Color, RenderTarget};
use render_gpu::GpuContextMode;
use spherical::RadianceField;
use clap::Parser;
use thiserror::Error;
use benchmark::Bench;



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
    const SCREEN_WIDTH: usize = 256;
    const SCREEN_HEIGHT: usize = 256;

    let args = Args::parse();

    eprintln!("Reading rendering configuration from file...");

    let cfg = toml::from_str(
        &tokio::fs::read_to_string("assets/render_configuration.toml").await?,
    )?;

    eprintln!("Reading model from file...");

    let field = bincode::deserialize::<RadianceField>(
        &tokio::fs::read("assets/model.bin").await?,
    )?;

    let mut bench = Bench::new();

    let image = match args.r#type {
        MethodType::Gpu => {
            let ctx = render_gpu::GpuContext::new(render_gpu::GpuContextMode::Debug).await?;
            render_gpu::render_gpu(SCREEN_WIDTH, SCREEN_HEIGHT, &ctx, &field, &cfg, &mut bench)
        },
        MethodType::MultiCpu
            => render_cpu::render_multicpu(SCREEN_WIDTH, SCREEN_HEIGHT, &field, &cfg, &mut bench),
        MethodType::SingleCpu
            => render_cpu::render_singlecpu(SCREEN_WIDTH, SCREEN_HEIGHT, &field, &cfg, &mut bench),
    };

    if args.bench {
        println!("{}", bench.total());
    }

    let file = std::fs::File::create("output/result.png")?;

    let buf_writer = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(
        buf_writer, SCREEN_WIDTH as u32, SCREEN_HEIGHT as u32,
    );

    encoder.set_color(png::ColorType::Rgba);

    let mut writer = encoder.write_header()?;
    writer.write_image_data(bytemuck::cast_slice(&image))?; 

    Ok(())
}



/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about = "Radiance field volume renderer", long_about = None)]
struct Args {
    /// Name of the output file
    #[arg(short, long, default_value_t = String::from("output/result.png"))]
    out: String,

    /// Set render target
    #[arg(long, default_value_t = RenderTarget::Color)]
    target: RenderTarget,

    /// Compute context
    #[arg(long, default_value_t = GpuContextMode::Debug)]
    mode: GpuContextMode,

    /// Enables benchmarking
    #[arg(long, short)]
    bench: bool,

    /// Computation method. Valid values are: singlecpu, multicpu, gpu.
    #[arg(long, short, default_value_t = MethodType::Gpu)]
    r#type: MethodType,
}



#[derive(Clone, Debug, PartialEq, Default, Copy, Eq, PartialOrd, Ord, Hash)]
pub enum MethodType {
    SingleCpu,
    MultiCpu,
    #[default]
    Gpu,
}

impl std::str::FromStr for MethodType {
    type Err = MethodTypeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "singlecpu" => Self::SingleCpu,
            "multicpu" => Self::MultiCpu,
            "gpu" => Self::Gpu,
            _ => return Err(MethodTypeParseError(s.to_owned())),
        })
    }
}

impl std::fmt::Display for MethodType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::SingleCpu => "singlecpu",
            Self::MultiCpu => "multicpu",
            Self::Gpu => "gpu",
        })
    }
}



#[derive(Debug, Error)]
#[error("invalid method-type '{0}', valid values are: \
         'singlecpu', 'multicpu' and 'gpu'")]
pub struct MethodTypeParseError(pub String);