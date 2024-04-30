use crate::integratable::Integratable;
use nalgebra::{Quaternion, UnitQuaternion, Vector3};

struct GyroscopeResidual {
    r_wb_i: UnitQuaternion<f64>,
    r_wb_j: UnitQuaternion<f64>,
    integratable: Integratable,
}

impl GyroscopeResidual {
    fn new(
        r_wb_i: UnitQuaternion<f64>,
        r_wb_j: UnitQuaternion<f64>,
        integratable: Integratable,
    ) -> Self {
        GyroscopeResidual {
            r_wb_i,
            r_wb_j,
            integratable,
        }
    }

    fn residual(&self) -> Vector3<f64> {
        let dr = self.integratable.integrate_euler();
        let d = self.r_wb_j.inverse() * self.r_wb_i * dr;
        d.scaled_axis()
    }

    fn error(&self) -> f64 {
        self.residual().norm_squared()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::GyroscopeGenerator;
    use crate::integration::integrate_euler;
    use core::f64::consts::PI;

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
    fn test_gyro_residual() {
        let generator = GyroscopeGenerator::new(time, quat);
        let a = 0;
        let m = 8000;
        let b = 10000;
        let (ta, qa) = generator.rotation(a);
        let (tb, qb) = generator.rotation(b);

        let mut integratable_ts = vec![];
        let mut integratable_ws = vec![];
        for i in a..=m {
            let (t, w) = generator.angular_velocity(i);
            integratable_ts.push(t);
            integratable_ws.push(w);
        }
        let integratable =
            Integratable::new_interpolated(&integratable_ts, &integratable_ws, time(a), time(m));
        let gyro = GyroscopeResidual::new(qa, qb, integratable);

        let mut residual_ts = vec![];
        let mut residual_ws = vec![];
        for i in m..=b {
            let (t, w) = generator.angular_velocity(i);
            residual_ts.push(t);
            residual_ws.push(w);
        }
        let expected_q = integrate_euler(&residual_ts, &residual_ws).inverse();

        assert!((gyro.residual() - expected_q.scaled_axis()).norm() < 1e-8);
    }
}
