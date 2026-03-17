use ash::vk;
use crate::{
    buffer::{Buffer, BufferUsage},
    command::CommandPool,
    device::GpuDevice,
    texture::Texture,
    types::GpuResult,
};
use std::sync::Arc;

/// Pomocnik do przesyłania danych CPU → GPU przez staging buffer
pub struct Uploader {
    pool: CommandPool,
    device: Arc<GpuDevice>,
}

impl Uploader {
    pub fn new(device: Arc<GpuDevice>) -> GpuResult<Self> {
        let pool = CommandPool::new(device.clone())?;
        Ok(Self { pool, device })
    }

    /// Kopiuje dane do bufora na GPU (device-local)
    pub fn upload_buffer<T: Copy>(&self, data: &[T], usage: BufferUsage) -> GpuResult<Buffer> {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;

        // Staging buffer (CPU visible)
        let staging = Buffer::new(
            self.device.clone(), size,
            BufferUsage::TRANSFER_SRC, true,
        )?;
        staging.write(data)?;

        // Docelowy bufor (device local)
        let dst = Buffer::new(
            self.device.clone(), size,
            usage | BufferUsage::TRANSFER_DST, false,
        )?;

        self.copy_buffer(&staging, &dst, size)?;
        Ok(dst)
    }

    /// Przesyła piksele do tekstury i zmienia jej layout na SHADER_READ_ONLY
    pub fn upload_texture(&self, texture: &Texture, pixels: &[u8]) -> GpuResult<()> {
        let size = pixels.len() as u64;
        let staging = Buffer::new(self.device.clone(), size, BufferUsage::TRANSFER_SRC, true)?;
        staging.write(pixels)?;

        let cmd = self.pool.alloc()?;
        cmd.begin()?;

        // Przejście: UNDEFINED → TRANSFER_DST
        Self::transition_image(&self.device, cmd.inner, texture.image,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        // Kopiowanie z bufora do obrazka
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0, base_array_layer: 0, layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: texture.extent.width, height: texture.extent.height, depth: 1 },
        };
        unsafe {
            self.device.logical.cmd_copy_buffer_to_image(
                cmd.inner, staging.inner, texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region],
            );
        }

        // Przejście: TRANSFER_DST → SHADER_READ_ONLY
        Self::transition_image(&self.device, cmd.inner, texture.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        cmd.end()?;
        self.submit_and_wait(cmd.inner)
    }

    fn copy_buffer(&self, src: &Buffer, dst: &Buffer, size: u64) -> GpuResult<()> {
        let cmd = self.pool.alloc()?;
        cmd.begin()?;
        let region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size };
        unsafe { self.device.logical.cmd_copy_buffer(cmd.inner, src.inner, dst.inner, &[region]) }
        cmd.end()?;
        self.submit_and_wait(cmd.inner)
    }

    fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> GpuResult<()> {
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        unsafe {
            self.device.logical.queue_submit(self.device.graphics_queue, &[submit], vk::Fence::null())?;
            self.device.logical.queue_wait_idle(self.device.graphics_queue)?;
        }
        Ok(())
    }

    fn transition_image(device: &GpuDevice, cmd: vk::CommandBuffer, image: vk::Image,
        old: vk::ImageLayout, new: vk::ImageLayout) {
        let (src_mask, dst_mask, src_stage, dst_stage) = match (old, new) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) =>
                (vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE,
                 vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) =>
                (vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ,
                 vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER),
            _ => return,
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old).new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image).src_access_mask(src_mask).dst_access_mask(dst_mask)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0,
                level_count: 1, base_array_layer: 0, layer_count: 1,
            });
        unsafe {
            device.logical.cmd_pipeline_barrier(cmd, src_stage, dst_stage,
                vk::DependencyFlags::empty(), &[], &[], &[barrier]);
        }
    }
}
