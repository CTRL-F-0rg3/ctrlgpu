use ash::vk;
use crate::{device::GpuDevice, renderpass::RenderPass, swapchain::Swapchain, types::GpuResult};
use std::sync::Arc;

pub struct Framebuffer {
    pub(crate) inner: vk::Framebuffer,
    device: Arc<GpuDevice>,
}

pub struct FramebufferSet {
    pub(crate) framebuffers: Vec<Framebuffer>,
}

impl Framebuffer {
    pub fn new(
        device: Arc<GpuDevice>,
        render_pass: &RenderPass,
        attachments: &[vk::ImageView],
        width: u32,
        height: u32,
    ) -> GpuResult<Self> {
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass.inner)
            .attachments(attachments)
            .width(width)
            .height(height)
            .layers(1);

        let inner = unsafe { device.logical.create_framebuffer(&info, None)? };
        Ok(Self { inner, device })
    }
}

impl FramebufferSet {
    /// Tworzy po jednym framebufferze na każdy image view swapchaina
    pub fn from_swapchain(
        device: Arc<GpuDevice>,
        render_pass: &RenderPass,
        swapchain: &Swapchain,
    ) -> GpuResult<Self> {
        let framebuffers = swapchain.image_views.iter()
            .map(|&view| Framebuffer::new(
                device.clone(),
                render_pass,
                &[view],
                swapchain.extent.width,
                swapchain.extent.height,
            ))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self { framebuffers })
    }

    /// Tworzy z depth attachmentem (color + depth)
    pub fn from_swapchain_with_depth(
        device: Arc<GpuDevice>,
        render_pass: &RenderPass,
        swapchain: &Swapchain,
        depth_view: vk::ImageView,
    ) -> GpuResult<Self> {
        let framebuffers = swapchain.image_views.iter()
            .map(|&color_view| Framebuffer::new(
                device.clone(),
                render_pass,
                &[color_view, depth_view],
                swapchain.extent.width,
                swapchain.extent.height,
            ))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self { framebuffers })
    }

    pub fn get(&self, index: usize) -> Option<&Framebuffer> {
        self.framebuffers.get(index)
    }

    pub fn len(&self) -> usize {
        self.framebuffers.len()
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) { unsafe { self.device.logical.destroy_framebuffer(self.inner, None) } }
}
