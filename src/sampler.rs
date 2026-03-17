use ash::vk;
use crate::{device::GpuDevice, types::GpuResult};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter { Nearest, Linear }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode { Repeat, MirroredRepeat, ClampToEdge, ClampToBorder }

pub struct SamplerDesc {
    pub filter:       Filter,
    pub wrap:         WrapMode,
    pub anisotropy:   Option<f32>,  // None = wyłączona
    pub mip_levels:   u32,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self { filter: Filter::Linear, wrap: WrapMode::Repeat, anisotropy: Some(16.0), mip_levels: 1 }
    }
}

pub struct Sampler {
    pub(crate) inner: vk::Sampler,
    device: Arc<GpuDevice>,
}

impl Sampler {
    pub fn new(device: Arc<GpuDevice>, desc: SamplerDesc) -> GpuResult<Self> {
        let vk_filter = match desc.filter {
            Filter::Nearest => vk::Filter::NEAREST,
            Filter::Linear  => vk::Filter::LINEAR,
        };
        let vk_wrap = match desc.wrap {
            WrapMode::Repeat         => vk::SamplerAddressMode::REPEAT,
            WrapMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            WrapMode::ClampToEdge    => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            WrapMode::ClampToBorder  => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        };

        let (anisotropy_enable, max_anisotropy) = match desc.anisotropy {
            Some(v) => (vk::TRUE, v.min(device.max_anisotropy())),
            None    => (vk::FALSE, 1.0),
        };

        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk_filter)
            .min_filter(vk_filter)
            .mipmap_mode(if desc.filter == Filter::Linear {
                vk::SamplerMipmapMode::LINEAR
            } else {
                vk::SamplerMipmapMode::NEAREST
            })
            .address_mode_u(vk_wrap)
            .address_mode_v(vk_wrap)
            .address_mode_w(vk_wrap)
            .anisotropy_enable(anisotropy_enable != 0)
            .max_anisotropy(max_anisotropy)
            .min_lod(0.0)
            .max_lod(desc.mip_levels as f32)
            .mip_lod_bias(0.0)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS);

        let inner = unsafe { device.logical.create_sampler(&info, None)? };
        Ok(Self { inner, device })
    }

    /// Sampler najbardziej przydatny do UI / sprite'ów
    pub fn nearest(device: Arc<GpuDevice>) -> GpuResult<Self> {
        Self::new(device, SamplerDesc { filter: Filter::Nearest, anisotropy: None, ..Default::default() })
    }

    /// Sampler do geometrii 3D z anizotropią
    pub fn linear_aniso(device: Arc<GpuDevice>) -> GpuResult<Self> {
        Self::new(device, SamplerDesc::default())
    }
}

impl Drop for Sampler {
    fn drop(&mut self) { unsafe { self.device.logical.destroy_sampler(self.inner, None) } }
}
