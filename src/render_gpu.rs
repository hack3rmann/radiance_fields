use crate::{render_cpu::{compact_color, RenderConfiguration}, spherical::RadianceField};
use std::sync::Arc;
use array_init::array_init;
use glam::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wgpu::util::DeviceExt;
use rayon::prelude::*;



#[derive(Clone, Debug, PartialEq, Default, Copy, Eq, PartialOrd, Ord, Hash)]
#[derive(Serialize, Deserialize)]
pub enum GpuContextMode {
    #[default]
    #[serde(rename = "debug")]
    Debug,
    #[serde(rename = "validation")]
    ReleaseValidation,
    #[serde(rename = "silent")]
    ReleaseSilent,
}

impl std::fmt::Display for GpuContextMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Debug => "debug",
            Self::ReleaseValidation => "validation",
            Self::ReleaseSilent => "silent",
        })
    }
}

#[derive(Clone, Debug, Error)]
pub enum ParseGpuContextModeError {
    #[error("invalid GPU context mode '{0}', valid values are: 'debug', 'validation', 'silent'")]
    InvalidArg(String),
}

impl From<GpuContextMode> for wgpu::InstanceFlags {
    fn from(value: GpuContextMode) -> Self {
        use GpuContextMode::*;

        match value {
            Debug => Self::DEBUG | Self::VALIDATION,
            ReleaseValidation => Self::VALIDATION,
            ReleaseSilent => Self::empty(),
        }
    }
}



#[derive(Clone, Debug)]
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter: Arc<wgpu::Adapter>,
    instance: Arc<wgpu::Instance>,
}

impl GpuContext {
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub async fn new(mode: GpuContextMode)
        -> Result<Self, wgpu::RequestDeviceError>
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: mode.into(),
            ..Default::default()
        });

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }).await.expect("failed to request the adapter");

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                label: None,
                required_limits: adapter.limits(),
            },
            None,
        ).await?;

        device.set_device_lost_callback(|reason, msg| {
            if msg != "Device dropped." {
                eprintln!("the device is lost: '{msg}', because: {reason:?}");
            }
        });

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter: Arc::new(adapter),
            instance: Arc::new(instance),
        })
    }
}

pub fn textures_from_radiance(field: &RadianceField) -> [Vec<u8>; 9] {
    array_init(|i| {
        let mut colors = Vec::with_capacity(field.size().pow(3));

        field.cells.par_iter()
            .map(|cell| compact_color(vec4(
                cell.sh_r[i],
                cell.sh_g[i],
                cell.sh_b[i],
                cell.density,
            )))
            .collect_into_vec(&mut colors);

        bytemuck::allocation::cast_vec(colors)
    })
}

pub fn render_gpu(
    screen_width: usize, screen_height: usize, ctx: &GpuContext,
    field: &RadianceField, cfg: &RenderConfiguration,
) -> Vec<u8> {
    let shader = ctx.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Glsl {
            shader: include_str!("radiance.comp").into(),
            stage: wgpu::naga::ShaderStage::Compute,
            defines: Default::default(),
        },
    });

    const N_TEXTURES: usize = 9;

    let field_textures_data = textures_from_radiance(field);

    let textures: [_; N_TEXTURES] = array_init(|i| {
        ctx.device().create_texture_with_data(
            ctx.queue(),
            &wgpu::TextureDescriptor {
                label: Some("model_texture"),
                size: wgpu::Extent3d {
                    width: field.size() as u32,
                    height: field.size() as u32,
                    depth_or_array_layers: field.size() as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &field_textures_data[i],
        )
    });

    let texture_array_uniform_layout = ctx.device().create_bind_group_layout(
        &wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_array"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: Some(std::num::NonZeroU32::new(N_TEXTURES as u32).unwrap()),
                },
            ],
        },
    );

    let sampler = ctx.device().create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let texture_views: [_; N_TEXTURES] = array_init(|i| {
        textures[i].create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        })
    });

    let texture_array_uniform = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("texture_array"),
        layout: &texture_array_uniform_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(
                    &array_init::<_, _, N_TEXTURES>(|i| &texture_views[i]),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    todo!()
}