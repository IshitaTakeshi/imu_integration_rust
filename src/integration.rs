use nalgebra::{UnitQuaternion, Vector3};

pub fn integrate_euler(
    timestamps: &[f64],
    angular_velocities: &[Vector3<f64>],
    bias_correction: &Vector3<f64>,
) -> UnitQuaternion<f64> {
    let mut q = UnitQuaternion::identity();
    for i in 0..timestamps.len() - 1 {
        let dt = timestamps[i + 1] - timestamps[i + 0];
        let w = angular_velocities[i] - bias_correction;
        let dq = UnitQuaternion::from_scaled_axis(w * dt);
        q = q * dq;
    }
    q
}

fn integrate_midpoint(
    timestamps: &[f64],
    angular_velocities: &[Vector3<f64>],
    bias_correction: &Vector3<f64>,
) -> UnitQuaternion<f64> {
    let mut q = UnitQuaternion::identity();
    for i in 0..timestamps.len() - 1 {
        let dt = timestamps[i + 1] - timestamps[i + 0];
        let w0 = angular_velocities[i + 0];
        let w1 = angular_velocities[i + 1];
        let w = 0.5 * (w0 + w1) - bias_correction;
        let dq = UnitQuaternion::from_scaled_axis(w * dt);
        q = q * dq;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::generator::GyroscopeGenerator;
    use core::f64::consts::PI;
    use nalgebra::Quaternion;

    fn quat(t: f64) -> UnitQuaternion<f64> {
        let x = f64::sin(2. * PI * 1. * t);
        let y = f64::sin(2. * PI * 2. * t);
        let z = f64::sin(2. * PI * 3. * t);
        let w2 = 1.0 - (1. / 3.) * (x * x + y * y + z * z);
        assert!(w2 > 0.0);
        let w = f64::sqrt(w2);

        UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z))
    }

    const DELTA_T: f64 = 0.0001;
    fn time(i: usize) -> f64 {
        DELTA_T * (i as f64)
    }

    #[test]
    fn test_integrate_euler() {
        let generator = GyroscopeGenerator::new(time, quat);
        let a = 0;
        let b = 10000;
        let (_ta, qa) = generator.rotation(a);
        let (_tb, qb) = generator.rotation(b);

        let mut timestamps = vec![];
        let mut angular_velocities = vec![];
        for i in (a..=b).step_by(10) {
            // NOTE b is inclusive
            let (t, w) = generator.angular_velocity(i);
            timestamps.push(t);
            angular_velocities.push(w);
        }
        let dq = integrate_euler(&timestamps, &angular_velocities, &Vector3::zeros());
        assert!(f64::abs(((qa * dq).inverse() * qb).angle()) < 1e-4);
    }

    #[test]
    fn test_integrate_with_bias() {
        let generator = GyroscopeGenerator::new(time, quat);

        let bias = Vector3::new(0.3, 0.5, 0.1);

        let a = 0;
        let b = 10000;
        let (_ta, qa) = generator.rotation(a);
        let (_tb, qb) = generator.rotation(b);

        let mut timestamps = vec![];
        let mut angular_velocities = vec![];
        for i in a..=b {
            // NOTE b is inclusive
            let (t, w) = generator.angular_velocity(i);
            timestamps.push(t);
            angular_velocities.push(w + bias);
        }
        let dq = integrate_euler(&timestamps, &angular_velocities, &bias);
        assert!(f64::abs(((qa * dq).inverse() * qb).angle()) < 1e-4);
    }

    #[test]
    fn test_integrate_midpoint() {
        let generator = GyroscopeGenerator::new(time, quat);
        let a = 0;
        let b = 10000;
        let (_ta, qa) = generator.rotation(a);
        let (_tb, qb) = generator.rotation(b);

        let mut timestamps = vec![];
        let mut angular_velocities = vec![];
        for i in (a..=b).step_by(10) {
            // NOTE b is inclusive
            let (t, w) = generator.angular_velocity(i);
            timestamps.push(t);
            angular_velocities.push(w);
        }
        let dq = integrate_midpoint(&timestamps, &angular_velocities, &Vector3::zeros());
        assert!(f64::abs(((qa * dq).inverse() * qb).angle()) < 1e-4);
    }

    #[test]
    fn test_integrate_midpoint_with_bias() {
        let generator = GyroscopeGenerator::new(time, quat);

        let bias = Vector3::new(0.3, 0.5, 0.1);

        let a = 0;
        let b = 10000;
        let (_ta, qa) = generator.rotation(a);
        let (_tb, qb) = generator.rotation(b);

        let mut timestamps = vec![];
        let mut angular_velocities = vec![];
        for i in (a..=b).step_by(10) {
            // NOTE b is inclusive
            let (t, w) = generator.angular_velocity(i);
            timestamps.push(t);
            angular_velocities.push(w + bias);
        }
        let dq = integrate_midpoint(&timestamps, &angular_velocities, &bias);
        assert!(f64::abs(((qa * dq).inverse() * qb).angle()) < 1e-4);
    }
}
