use std::sync::Arc;
use raw_window_handle::{HasWindowHandle, HasDisplayHandle};
use crate::{
    debug::DebugMessenger,
    device::GpuDevice,
    instance::GpuInstance,
    surface::Surface,
    swapchain::Swapchain,
    types::{Extent2D, GpuResult},
    upload::Uploader,
};

/// Główny entry point biblioteki — inicjalizuje cały stos Vulkana
pub struct GpuContext {
    pub instance:  Arc<GpuInstance>,
    pub device:    Arc<GpuDevice>,
    pub surface:   Surface,
    pub swapchain: Swapchain,
    pub uploader:  Uploader,
    _debug:        Option<DebugMessenger>,
}

pub struct ContextDesc<'a> {
    pub app_name:   &'a str,
    pub size:       Extent2D,
    pub validation: bool,
}

impl GpuContext {
    pub fn new<W: HasWindowHandle + HasDisplayHandle>(
        window: &W,
        desc: ContextDesc,
    ) -> GpuResult<Self> {
        let instance = Arc::new(GpuInstance::new(desc.app_name, desc.validation)?);

        let debug = if desc.validation {
            Some(DebugMessenger::new(instance.clone())?)
        } else {
            None
        };

        let device   = Arc::new(GpuDevice::new(instance.clone())?);
        let surface  = Surface::new(instance.clone(), window)?;
        let swapchain = Swapchain::new(device.clone(), &surface, desc.size)?;
        let uploader  = Uploader::new(device.clone())?;

        Ok(Self { instance, device, surface, swapchain, uploader, _debug: debug })
    }

    /// Odtwarza swapchain po resize okna
    pub fn resize(&mut self, new_size: Extent2D) -> GpuResult<()> {
        self.device.wait_idle()?;
        self.swapchain = Swapchain::new(self.device.clone(), &self.surface, new_size)?;
        Ok(())
    }

    /// Skrót do czekania na idle GPU — przydatne przed shutdownem
    pub fn wait_idle(&self) -> GpuResult<()> {
        self.device.wait_idle()
    }

    pub fn swapchain_format(&self) -> ash::vk::Format {
        self.swapchain.format
    }

    pub fn swapchain_extent(&self) -> Extent2D {
        Extent2D {
            width:  self.swapchain.extent.width,
            height: self.swapchain.extent.height,
        }
    }
}
