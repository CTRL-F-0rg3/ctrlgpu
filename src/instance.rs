use ash::{vk, Entry, Instance};
use crate::types::{GpuError, GpuResult};

pub struct GpuInstance {
    pub(crate) entry: Entry,
    pub(crate) inner: Instance,
    pub(crate) validation: bool,
}

impl GpuInstance {
    pub fn new(app_name: &str, validation: bool) -> GpuResult<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| GpuError::Init(e.to_string()))?;

        let app_name_cstr = std::ffi::CString::new(app_name)
            .map_err(|e| GpuError::Init(e.to_string()))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name_cstr)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"gpu_abstraction")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let mut layers: Vec<*const i8> = vec![];
        if validation {
            layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        }

        let mut extensions = Self::required_extensions();
        if validation {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        // MoltenVK na macOS wymaga portability extension
        #[cfg(target_os = "macos")]
        {
            extensions.push(ash::vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
        }

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);

        #[cfg(target_os = "macos")]
        {
            create_info = create_info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }

        let inner = unsafe { entry.create_instance(&create_info, None)? };
        Ok(Self { entry, inner, validation })
    }

    fn required_extensions() -> Vec<*const i8> {
        let mut exts = vec![
            ash::khr::surface::NAME.as_ptr(),
        ];

        #[cfg(windows)]
        exts.push(ash::khr::win32_surface::NAME.as_ptr());

        #[cfg(target_os = "linux")]
        {
            exts.push(ash::khr::xcb_surface::NAME.as_ptr());
            // lub xlib/wayland — ash-window wybierze automatycznie
        }

        #[cfg(target_os = "macos")]
        exts.push(ash::ext::metal_surface::NAME.as_ptr());

        exts
    }

    /// Sprawdza czy dana extension jest dostępna
    pub fn supports_extension(&self, name: &std::ffi::CStr) -> bool {
        unsafe { self.entry.enumerate_instance_extension_properties(None) }
            .unwrap_or_default()
            .iter()
            .any(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) } == name)
    }
}

impl Drop for GpuInstance {
    fn drop(&mut self) { unsafe { self.inner.destroy_instance(None) } }
}
