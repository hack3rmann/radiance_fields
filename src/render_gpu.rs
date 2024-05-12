use crate::{
    graphics::RenderConfiguration, spherical::RadianceField
};
use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use glam::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
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
                required_features: wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::PUSH_CONSTANTS
                    | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
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



pub fn radiance_field_to_textures(field: &RadianceField) -> Vec<Vec<[f32; 4]>> {
    const BATCH_SIZE: usize = 64;
    
    assert!(field.size() % BATCH_SIZE == 0, "radiance field size should be divisible by {BATCH_SIZE}");

    (0..field.size() / BATCH_SIZE)
        .map(|batch_index| {
            let volume = field.size() * field.size() * BATCH_SIZE;
            let batch = &field.cells[batch_index * volume..(batch_index + 1) * volume];

            const N_TEXTURE_SLICES: usize = 9;

            (0..N_TEXTURE_SLICES)
                .into_par_iter()
                .flat_map_iter(|i| {
                    batch.iter()
                        .map(move |cell| [
                            cell.sh_r[i],
                            cell.sh_g[i],
                            cell.sh_b[i],
                            cell.density,
                        ])
                })
                .collect::<Vec<_>>()
        })
        .collect()
}



#[repr(C)]
#[derive(Clone, Debug, PartialEq, Copy)]
#[derive(Pod, Zeroable)]
pub struct GpuRenderCfg {
    pub camera_distance: f32,
    pub camera_theta: f32,
    pub camera_phi: f32,
    pub camera_vfov: f32,
    pub camera_target_pos: Vec4,
    pub bounding_box_lo: Vec4,
    pub bounding_box_hi: Vec4,
    pub rm_settings_n_steps: u32,
    pub _pad: Vec3,
}

impl From<&RenderConfiguration> for GpuRenderCfg {
    fn from(value: &RenderConfiguration) -> Self {
        Self {
            camera_distance: value.camera.distance,
            camera_theta: value.camera.theta,
            camera_phi: value.camera.phi,
            camera_vfov: value.camera.vfov,
            camera_target_pos: value.camera.target_pos.extend(0.0),
            rm_settings_n_steps: value.rm_settings.n_steps,
            bounding_box_lo: value.bounding_box.lo.extend(0.0),
            bounding_box_hi: value.bounding_box.hi.extend(0.0),
            _pad: Vec3::default(),
        }
    }
}



pub fn render_gpu(
    screen_width: usize, screen_height: usize, ctx: &GpuContext,
    field: &RadianceField, cfg: &RenderConfiguration,
) -> Vec<u8> {
    use wgpu::*;
    use wgpu::util::*;

    let cfg = GpuRenderCfg::from(cfg);

    assert!(screen_width % 8 == 0);
    assert!(screen_height % 8 == 0);

    let shader = ctx.device().create_shader_module(ShaderModuleDescriptor {
        label: Some("model_shader"),
        source: ShaderSource::Glsl {
            shader: include_str!("radiance.comp").into(),
            stage: naga::ShaderStage::Compute,
            defines: Default::default(),
        },
    });

    let screen_buffer_len = screen_width * screen_height;
    let screen_buffer_size = std::mem::size_of::<[f32; 4]>() * screen_buffer_len;

    let cpu_screen_buffer = ctx.device().create_buffer(&BufferDescriptor {
        label: Some("screen_buffer"),
        size: screen_buffer_size as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let screen_image = ctx.device().create_texture_with_data(
        ctx.queue(),
        &TextureDescriptor {
            label: Some("screen_image_texture"),
            size: Extent3d {
                width: screen_width as u32,
                height: screen_height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        },
        TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(
            &vec![Vec4::ZERO; screen_width * screen_height]
        ),
    );

    let screen_view = screen_image.create_view(&Default::default());

    const BATCH_SIZE: usize = 64;

    let field_texture_data = radiance_field_to_textures(field);

    let field_texture_size = Extent3d {
        width: field.size() as u32,
        height: field.size() as u32,
        depth_or_array_layers: (BATCH_SIZE * 9) as u32,
    };

    let model_textures = field_texture_data.iter().map(|texture_data| {
        ctx.device().create_texture_with_data(
            ctx.queue(),
            &TextureDescriptor {
                label: Some("model_texture"),
                size: field_texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
            util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(texture_data),
        )
    }).collect::<Vec<_>>();

    let model_views = model_textures.iter().map(|texture| {
        texture.create_view(&TextureViewDescriptor {
            label: Some("mode_view"),
            format: Some(TextureFormat::Rgba32Float),
            dimension: Some(TextureViewDimension::D3),
            ..Default::default()
        })
    }).collect::<Vec<_>>();

    let render_cfg_buffer = ctx.device().create_buffer_init(&BufferInitDescriptor {
        label: Some("render_configuration_uniform"),
        contents: bytemuck::bytes_of(&cfg),
        usage: BufferUsages::UNIFORM,
    });

    #[repr(C)]
    #[derive(Debug, Clone, PartialEq, Default, Copy, Eq, PartialOrd, Ord, Hash)]
    #[derive(Pod, Zeroable)]
    struct PassConfiguration {
        screen_width: u32,
        screen_height: u32,
    }

    let pass_cfg_buffer = ctx.device().create_buffer_init(&BufferInitDescriptor {
        label: Some("pass_configuration_uniform"),
        contents: bytemuck::bytes_of(&PassConfiguration {
            screen_width: screen_width as u32,
            screen_height: screen_height as u32,
        }),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group_layout = ctx.device().create_bind_group_layout(
        &BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        },
    );

    #[repr(C)]
    #[derive(Clone, Debug, PartialEq, Default, Copy)]
    #[derive(Pod, Zeroable)]
    struct PushConst {
        bounds_lo: Vec4,
        bounds_hi: Vec4,
        index: u32,
        n_passes: u32,
        _pad: [u32; 2],
    }

    let pipeline_layout = ctx.device().create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[
            PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConst>() as u32,
            },
        ],
    });

    let pipeline = ctx.device().create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
    });

    let n_passes = model_views.len();

    for (i, model_view) in model_views.iter().enumerate() {
        let bind_group = ctx.device().create_bind_group(&BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&screen_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(model_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: render_cfg_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pass_cfg_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device().create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());

            let push = PushConst {
                bounds_lo: Vec4::new(
                    -0.5, -0.5, i as f32 / n_passes as f32 - 0.5, 0.0,
                ),
                bounds_hi: Vec4::new(
                    0.5, 0.5, (i + 1) as f32 / n_passes as f32 - 0.5, 0.0,
                ),
                index: i as u32,
                n_passes: n_passes as u32,
                _pad: [0; 2],
            };

            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(&pipeline);
            pass.set_push_constants(0, bytemuck::bytes_of(&push));
            pass.dispatch_workgroups(
                screen_width as u32 / 8,
                screen_height as u32 / 8,
                1,
            );
        }

        ctx.device().poll(MaintainBase::wait_for(
            ctx.queue().submit([encoder.finish()]),
        ));
    }

    let mut encoder = ctx.device().create_command_encoder(&Default::default());

    encoder.copy_texture_to_buffer(
        screen_image.as_image_copy(),
        ImageCopyBufferBase {
            buffer: &cpu_screen_buffer,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some((std::mem::size_of::<[f32; 4]>() * screen_width) as u32),
                rows_per_image: Some(screen_height as u32),
            },
        },
        Extent3d {
            width: screen_width as u32,
            height: screen_height as u32,
            depth_or_array_layers: 1,
        },
    );

    ctx.device().poll(MaintainBase::wait_for(
        ctx.queue().submit([encoder.finish()]),
    ));

    cpu_screen_buffer.slice(..).map_async(MapMode::Read, Result::unwrap);

    ctx.device().poll(MaintainBase::Wait);

    let range = cpu_screen_buffer.slice(..).get_mapped_range();

    let mut result = Vec::with_capacity(screen_width * screen_height);

    range.par_chunks_exact(std::mem::size_of::<[f32; 4]>())
        .map(|mut chunk| {
            let (r, g, b, a);
            (r, chunk) = chunk.split_first_chunk().unwrap();
            (g, chunk) = chunk.split_first_chunk().unwrap();
            (b, chunk) = chunk.split_first_chunk().unwrap();
            (a, _) = chunk.split_first_chunk().unwrap();

            vec4(
                f32::from_le_bytes(*r),
                f32::from_le_bytes(*g),
                f32::from_le_bytes(*b),
                f32::from_le_bytes(*a),
            )
        })
        .map(crate::graphics::Color::from_vec4)
        .collect_into_vec(&mut result);

    bytemuck::allocation::cast_vec(result)
}