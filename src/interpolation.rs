use nalgebra::Vector3;

pub fn interpolate(t0: f64, t1: f64, t: f64, w0: &Vector3<f64>, w1: &Vector3<f64>) -> Vector3<f64> {
    assert!(t1 != t0);
    ((t1 - t) * w0 + (t - t0) * w1) / (t1 - t0)
}
