use nalgebra::{UnitQuaternion, Vector3};

pub struct GyroscopeGenerator {
    t: fn(usize) -> f64,
    f: fn(f64) -> UnitQuaternion<f64>,
}

impl GyroscopeGenerator {
    pub fn new(t: fn(usize) -> f64, f: fn(f64) -> UnitQuaternion<f64>) -> Self {
        GyroscopeGenerator { t: t, f: f }
    }

    pub fn angular_velocity(&self, i: usize) -> (f64, Vector3<f64>) {
        let (t0, q0) = self.rotation(i + 0);
        let (t1, q1) = self.rotation(i + 1);
        let dq = q0.inverse() * q1;

        // Compute the numerical derivative of the rotation
        let dt = t1 - t0;
        let omega = dq.scaled_axis() / dt;
        (t0, omega)
    }

    pub fn rotation(&self, i: usize) -> (f64, UnitQuaternion<f64>) {
        let t = (self.t)(i);
        let q = (self.f)(t);
        (t, q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Quaternion;

    const PI: f64 = std::f64::consts::PI;

    fn quat(t: f64) -> UnitQuaternion<f64> {
        let x = f64::sin(2. * PI * 1. * t);
        let y = f64::sin(2. * PI * 2. * t);
        let z = f64::sin(2. * PI * 3. * t);
        let w2 = 1.0 - (1. / 3.) * (x * x + y * y + z * z);
        assert!(w2 > 0.0);
        let w = f64::sqrt(w2);

        UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z))
    }

    const DELTA_T: f64 = 0.01;
    fn time(i: usize) -> f64 {
        DELTA_T * (i as f64)
    }

    #[test]
    fn test_integration() {
        // Generate a trajectory from a to b on a unit sphere
        // (homomorphic to the set of unit quaternions).

        let generator = GyroscopeGenerator::new(time, quat);

        let a = 0;
        let b = 1000;

        let (_ta, qa) = generator.rotation(a);
        let (_tb, qb) = generator.rotation(b);

        let mut q = qa;
        for i in a..b {
            let dt = time(i + 1) - time(i + 0);
            let (_t, omega) = generator.angular_velocity(i);
            let dq = UnitQuaternion::from_scaled_axis(omega * dt);

            q = q * dq;
        }

        assert!(f64::abs((qb.inverse() * q).angle()) < 1e-8);
    }
}
