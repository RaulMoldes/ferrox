#[cfg(test)]
mod tests {
    use crate::backend::device::cpu;

    #[test]
    fn test_device_operations() {
        let device = cpu();

        let zeros = device.zeros::<f64>(&[2, 3]);
        assert_eq!(zeros.shape(), &[2, 3]);
        assert!(zeros.iter().all(|&x| x == 0.0));

        let ones = device.ones::<f64>(&[2, 3]);
        assert_eq!(ones.shape(), &[2, 3]);
        assert!(ones.iter().all(|&x| x == 1.0));

        let full = device.full::<f64>(&[2, 2], 5.0);
        assert_eq!(full.shape(), &[2, 2]);
        assert!(full.iter().all(|&x| x == 5.0));
    }
}
