use ash::{vk, Device};
use crate::{instance::GpuInstance, types::{GpuError, GpuResult}};
use std::sync::Arc;

pub struct QueueFamilies {
    pub graphics: u32,
    pub present:  u32,
    pub compute:  u32,
    /// Dedykowana kolejka transferu (opcjonalna — fallback: graphics)
    pub transfer: u32,
}

pub struct GpuDevice {
    pub(crate) physical:        vk::PhysicalDevice,
    pub(crate) logical:         Device,
    pub(crate) queues:          QueueFamilies,
    pub(crate) graphics_queue:  vk::Queue,
    pub(crate) compute_queue:   vk::Queue,
    pub(crate) transfer_queue:  vk::Queue,
    pub(crate) instance:        Arc<GpuInstance>,
}

impl GpuDevice {
    pub fn new(instance: Arc<GpuInstance>) -> GpuResult<Self> {
        let physical = Self::pick_physical(&instance)?;
        let queues   = Self::find_queue_families(&instance, physical)?;
        let logical  = Self::create_logical(&instance, physical, &queues)?;

        let graphics_queue = unsafe { logical.get_device_queue(queues.graphics, 0) };
        let compute_queue  = unsafe { logical.get_device_queue(queues.compute,  0) };
        let transfer_queue = unsafe { logical.get_device_queue(queues.transfer, 0) };

        Ok(Self { physical, logical, queues, graphics_queue, compute_queue, transfer_queue, instance })
    }

    /// Wybiera GPU — preferuje discrete, fallback na integrated
    fn pick_physical(instance: &GpuInstance) -> GpuResult<vk::PhysicalDevice> {
        let devices = unsafe { instance.inner.enumerate_physical_devices()? };

        devices.into_iter()
            .max_by_key(|&dev| {
                let props = unsafe { instance.inner.get_physical_device_properties(dev) };
                match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU   => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::VIRTUAL_GPU    => 1,
                    _                                       => 0,
                }
            })
            .ok_or(GpuError::NoSuitableDevice)
    }

    fn find_queue_families(instance: &GpuInstance, physical: vk::PhysicalDevice) -> GpuResult<QueueFamilies> {
        let props = unsafe { instance.inner.get_physical_device_queue_family_properties(physical) };

        let graphics = props.iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or(GpuError::NoSuitableDevice)? as u32;

        // Szukamy dedykowanej kolejki compute (bez graphics)
        let compute = props.iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE)
                       && !p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap_or(graphics as usize) as u32;

        // Dedykowana kolejka transferu (DMA) — bez graphics i compute
        let transfer = props.iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::TRANSFER)
                       && !p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                       && !p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .unwrap_or(graphics as usize) as u32;

        Ok(QueueFamilies { graphics, present: graphics, compute, transfer })
    }

    fn create_logical(instance: &GpuInstance, physical: vk::PhysicalDevice, queues: &QueueFamilies) -> GpuResult<Device> {
        let priority = [1.0f32];

        // Zbieramy unikalne indeksy kolejek
        let mut unique_families = vec![queues.graphics];
        if !unique_families.contains(&queues.compute)  { unique_families.push(queues.compute); }
        if !unique_families.contains(&queues.transfer) { unique_families.push(queues.transfer); }

        let queue_infos: Vec<_> = unique_families.iter().map(|&idx| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(idx)
                .queue_priorities(&priority)
        }).collect();

        let extensions = [
            ash::khr::swapchain::NAME.as_ptr(),
        ];

        let mut features_1_2 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true);

        let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let base_features = vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(true)
            .fill_mode_non_solid(true);

        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&base_features)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_3);

        Ok(unsafe { instance.inner.create_device(physical, &info, None)? })
    }

    /// Blokuje CPU do momentu aż GPU skończy wszystkie operacje
    pub fn wait_idle(&self) -> GpuResult<()> {
        unsafe { self.logical.device_wait_idle()? }
        Ok(())
    }

    /// Zwraca właściwości fizycznego urządzenia (limity, nazwa karty, itp.)
    pub fn properties(&self) -> vk::PhysicalDeviceProperties {
        unsafe { self.instance.inner.get_physical_device_properties(self.physical) }
    }

    /// Zwraca maksymalny poziom anizotropii wspierany przez kartę
    pub fn max_anisotropy(&self) -> f32 {
        self.properties().limits.max_sampler_anisotropy
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) { unsafe { self.logical.destroy_device(None) } }
}
