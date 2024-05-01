use crate::integratable::Integratable;
use nalgebra::{Quaternion, SMatrix, UnitQuaternion, Vector3};

use crate::{inv_right_jacobian, right_jacobian};

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

fn jacobian(
    ts: &[f64],
    ws: &[Vector3<f64>],
    qi: &UnitQuaternion<f64>,
    qj: &UnitQuaternion<f64>,
) -> SMatrix<f64, 3, 3> {
    let mut m = SMatrix::<f64, 3, 3>::zeros();
    let mut predcessor = UnitQuaternion::identity();
    for k in (0..ts.len() - 1).rev() {
        let dt = ts[k + 1] - ts[k + 0];
        let w = ws[k];
        let theta = w * dt;
        let r = predcessor.to_rotation_matrix();
        m += r.transpose() * right_jacobian(&theta) * dt;
        predcessor = UnitQuaternion::from_scaled_axis(theta) * predcessor;
    }

    let xi = (qj.inverse() * qi * predcessor).scaled_axis();
    inv_right_jacobian(&xi) * m
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
        let (_ta, qa) = generator.rotation(a);
        let (_tb, qb) = generator.rotation(b);

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

    #[test]
    fn test_approximate_gyro_residual() {
        let generator = GyroscopeGenerator::new(time, quat);
        let bias = Vector3::new(0.5, 0.3, 0.4);
        let dbias = Vector3::new(0.03, -0.15, 0.10);

        let i = 0;
        let j = 10000;
        let (_ti, qi) = generator.rotation(i);
        let (_tj, qj) = generator.rotation(j);

        let mut ts = vec![];
        let mut ws_observed = vec![];
        let mut ws_delta_affected = vec![];
        for k in i..=j {
            let (t, w) = generator.angular_velocity(k);
            ts.push(t);
            ws_observed.push(w - bias);
            ws_delta_affected.push(w - bias - dbias);
        }

        let integratable =
            Integratable::new_interpolated(&ts, &ws_delta_affected, time(i), time(j));
        let gyro = GyroscopeResidual::new(qi, qj, integratable);

        let mut q = qi.inverse() * qj;
        for k in 0..ts.len() - 1 {
            let dt = ts[k + 1] - ts[k + 0];
            let w = ws_observed[k];
            let theta = w * dt;
            q = q
                * UnitQuaternion::from_scaled_axis(theta)
                * UnitQuaternion::from_scaled_axis(-right_jacobian(&theta) * dbias * dt);
        }
        let residual = q.scaled_axis();

        assert!((gyro.residual() - residual).norm() < 1e-8);
    }

    #[test]
    fn test_approximate_gyro_residual2() {
        let generator = GyroscopeGenerator::new(time, quat);
        let bias = Vector3::new(0.5, 0.3, 0.4);
        let dbias = Vector3::new(0.03, -0.15, 0.10);

        let i = 0;
        let j = 4;
        let (_ti, qi) = generator.rotation(i);
        let (_tj, qj) = generator.rotation(j);

        let mut ts = vec![];
        let mut ws0 = vec![];
        let mut ws1 = vec![];
        for k in i..=j {
            let (t, w) = generator.angular_velocity(k);
            ts.push(t);
            ws0.push(w - bias);
            ws1.push(w - bias - dbias);
        }

        let integratable = Integratable::new_interpolated(&ts, &ws0, time(i), time(j));
        let gyro0 = GyroscopeResidual::new(qi, qj, integratable);

        let integratable = Integratable::new_interpolated(&ts, &ws1, time(i), time(j));
        let gyro1 = GyroscopeResidual::new(qi, qj, integratable);

        let jacobian = jacobian(&ts, &ws0, &qi, &qj);

        let r1 = gyro1.residual();
        let r0 = gyro0.residual();
        assert!((r1 - r0 + jacobian * dbias).norm() < 1e-8);
    }
}
