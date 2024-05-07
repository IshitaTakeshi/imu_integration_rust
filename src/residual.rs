use crate::integration;
use crate::interpolation;
use crate::right_jacobian;
use nalgebra::{SMatrix, UnitQuaternion, Vector3};

use crate::inv_right_jacobian;

#[derive(Debug, PartialEq)]
pub struct Integratable {
    ts: Vec<f64>,
    ws: Vec<Vector3<f64>>,
}

fn concat<T: Copy>(i: T, mid: &[T], j: T) -> Vec<T> {
    let mut array = vec![];
    array.push(i);
    for e in mid {
        array.push(*e);
    }
    array.push(j);
    array
}

fn check_timestamps(timestamps: &[f64], ti: f64, tj: f64) {
    let n = timestamps.len();
    assert!(n >= 3);
    assert!(timestamps[0] <= ti);
    assert!(ti < timestamps[1]);
    assert!(timestamps[n - 2] < tj);
    assert!(tj <= timestamps[n - 1]);
}

impl Integratable {
    pub fn new_interpolated(ts: &[f64], ws: &[Vector3<f64>], ti: f64, tj: f64) -> Self {
        check_timestamps(ts, ti, tj);
        assert_eq!(ts.len(), ws.len());
        let n = ws.len();
        let wi = interpolation::interpolate(ts[0], ts[1], ti, &ws[0], &ws[1]);
        let wj = interpolation::interpolate(ts[n - 2], ts[n - 1], tj, &ws[n - 2], &ws[n - 1]);
        let ts_new = concat(ti, &ts[1..n - 1], tj);
        let ws_new = concat(wi, &ws[1..n - 1], wj);
        Integratable {
            ts: ts_new,
            ws: ws_new,
        }
    }

    pub fn integrate_euler(&self, bias: &Vector3<f64>) -> UnitQuaternion<f64> {
        integration::integrate_euler(&self.ts, &self.ws, bias)
    }

    fn calc_m_and_predecessor(
        &self,
        bias: &Vector3<f64>,
    ) -> (SMatrix<f64, 3, 3>, UnitQuaternion<f64>) {
        let mut m = SMatrix::<f64, 3, 3>::zeros();
        let mut predecessor = UnitQuaternion::identity();
        for k in (0..self.ts.len() - 1).rev() {
            let dt = self.ts[k + 1] - self.ts[k + 0];
            let w = self.ws[k] - bias;
            let theta = w * dt;
            let r = predecessor.to_rotation_matrix();
            m += r.transpose() * right_jacobian(&theta) * dt;
            predecessor = UnitQuaternion::from_scaled_axis(theta) * predecessor;
        }
        (m, predecessor)
    }
}

#[derive(Debug, PartialEq)]
pub struct GyroscopeResidual {
    r_wb_i: UnitQuaternion<f64>,
    r_wb_j: UnitQuaternion<f64>,
    integratable: Integratable,
}

impl GyroscopeResidual {
    pub fn new(
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

    pub fn timestamps(&self) -> &Vec<f64> {
        &self.integratable.ts
    }

    pub fn angular_velocities(&self) -> &Vec<Vector3<f64>> {
        &self.integratable.ws
    }

    pub fn residual(&self, bias: &Vector3<f64>) -> Vector3<f64> {
        let dr = self.integratable.integrate_euler(bias);
        let d = self.r_wb_j.inverse() * self.r_wb_i * dr;
        d.scaled_axis()
    }

    pub fn error(&self, bias: &Vector3<f64>) -> f64 {
        self.residual(bias).norm_squared()
    }

    pub fn jacobian(&self, bias: &Vector3<f64>) -> SMatrix<f64, 3, 3> {
        let (m, predecessor) = self.integratable.calc_m_and_predecessor(bias);
        let xi = (self.r_wb_j.inverse() * self.r_wb_i * predecessor).scaled_axis();
        inv_right_jacobian(&xi) * m
    }
}

pub fn estimate_bias(gyro: &GyroscopeResidual) -> Vector3<f64> {
    let mut bias_pred = Vector3::zeros();
    for _ in 0..5 {
        let jacobian = gyro.jacobian(&bias_pred);
        let hessian = jacobian.transpose() * jacobian;
        let inv_hessian = hessian.try_inverse().unwrap();
        let dbias = inv_hessian * jacobian.transpose() * gyro.residual(&bias_pred);
        bias_pred = bias_pred + dbias;
    }
    bias_pred
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::GyroscopeGenerator;
    use crate::integration::integrate_euler;
    use core::f64::consts::PI;
    use nalgebra::Quaternion;

    #[test]
    fn test_new_interpolated() {
        let ts = [1.0, 1.1, 1.2, 1.3, 1.4];
        let ws = [
            Vector3::new(1.0, 0.0, 0.2),
            Vector3::new(2.0, 1.0, 0.2),
            Vector3::new(3.0, 1.0, 0.4),
            Vector3::new(4.0, 1.0, 0.5),
            Vector3::new(4.0, 1.5, 1.5),
        ];
        let integratable = Integratable::new_interpolated(&ts, &ws, 1.02, 1.36);
        assert_eq!(integratable.ts.len(), 5);
        assert_eq!(integratable.ws.len(), 5);

        let wi = Vector3::new(1.2, 0.2, 0.2);
        let wj = Vector3::new(4.0, 1.3, 1.1);
        assert!((integratable.ws[0] - wi).norm() < 1e-8);
        assert_eq!(integratable.ws[1], Vector3::new(2.0, 1.0, 0.2));
        assert_eq!(integratable.ws[2], Vector3::new(3.0, 1.0, 0.4));
        assert_eq!(integratable.ws[3], Vector3::new(4.0, 1.0, 0.5));
        assert!((integratable.ws[4] - wj).norm() < 1e-8);
    }

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
        // NOTE m is inclusive
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
        // NOTE b is inclusive
        for i in m..=b {
            let (t, w) = generator.angular_velocity(i);
            residual_ts.push(t);
            residual_ws.push(w);
        }
        let expected_q = integrate_euler(&residual_ts, &residual_ws, &Vector3::zeros()).inverse();

        assert!((gyro.residual(&Vector3::zeros()) - expected_q.scaled_axis()).norm() < 1e-8);
    }

    #[test]
    fn test_approximate_gyro_residual() {
        let generator = GyroscopeGenerator::new(time, quat);
        let bias = Vector3::new(0.5, 0.3, 0.4);
        let dbias = Vector3::new(0.03, -0.15, 0.10);

        let i = 0;
        let j = 1000;
        let (ti, qi) = generator.rotation(i);
        let (tj, qj) = generator.rotation(j);

        let mut ts = vec![];
        let mut ws = vec![];
        // NOTE j is inclusive
        for k in i..=j {
            let (t, w) = generator.angular_velocity(k);
            ts.push(t);
            ws.push(w - bias); // Assume that the observed angular velocities are already biased
        }

        let integratable = Integratable::new_interpolated(&ts, &ws, ti, tj);
        let gyro = GyroscopeResidual::new(qi, qj, integratable);

        let jacobian = gyro.jacobian(&Vector3::zeros());

        let r0 = gyro.residual(&Vector3::zeros());
        let r1 = gyro.residual(&dbias);
        assert!((r1 - r0 + jacobian * dbias).norm() < 1e-4);
    }

    #[test]
    fn test_gyro_error_minimization() {
        let generator = GyroscopeGenerator::new(time, quat);
        let bias_true = Vector3::new(0.15, -0.18, 0.24);

        let i = 0;
        let j = 1000;
        let (ti, qi) = generator.rotation(i);
        let (tj, qj) = generator.rotation(j);

        let mut ts = vec![];
        let mut ws = vec![];
        // NOTE j is inclusive
        for k in i..=j {
            let (t, w) = generator.angular_velocity(k);
            // Assume that the observed angular velocities are already biased and
            // we want to estimate the bias from the observed data.
            ts.push(t);
            ws.push(w - bias_true);
        }

        let integratable = Integratable::new_interpolated(&ts, &ws, ti, tj);
        let gyro = GyroscopeResidual::new(qi, qj, integratable);
        let bias_pred = estimate_bias(&gyro);
        assert!((bias_true + bias_pred).norm() < 1e-13);
    }
}
