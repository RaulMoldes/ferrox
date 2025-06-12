pub mod device;
pub mod numeric;

mod tests;

pub use device::Device;
pub use device::cpu;
pub use device::default_device;

pub use numeric::Float;
pub use numeric::Numeric;
