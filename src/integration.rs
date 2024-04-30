use crate::integratable::Integratable;
use crate::{identity, propagate};

use core::ops::Mul;
use nalgebra::geometry::{Quaternion, UnitQuaternion};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::{error::Error, io, process};

#[derive(Debug, Deserialize)]
struct ImuData {
    timestamp: u128,
    wx: f64,
    wy: f64,
    wz: f64,
    ax: f64,
    ay: f64,
    az: f64,
}

impl ImuData {
    fn angular_velocity(&self) -> Vector3<f64> {
        Vector3::<f64>::new(self.wx, self.wy, self.wz)
    }
    fn acceleration(&self) -> Vector3<f64> {
        Vector3::<f64>::new(self.ax, self.ay, self.az)
    }
}

fn read_imu_data() -> Result<(Vec<u128>, Vec<Vector3<f64>>, Vec<Vector3<f64>>), Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b',')
        .comment(Some(b'#'))
        .from_path("dataset/data.csv")?;

    let mut timestamps = vec![];
    let mut accelerations = vec![];
    let mut angular_velocities = vec![];
    let mut records = vec![];
    for string_record in reader.records() {
        let record: ImuData = string_record?.deserialize(None)?;
        timestamps.push(record.timestamp);
        accelerations.push(record.acceleration());
        angular_velocities.push(record.angular_velocity());
        records.push(record);
    }
    Ok((timestamps, accelerations, angular_velocities))
}

#[derive(Debug, Deserialize)]
struct GroundTruth {
    timestamp: u128,
    px: f64,
    py: f64,
    pz: f64,
    qw: f64,
    qx: f64,
    qy: f64,
    qz: f64,
}

impl GroundTruth {
    fn position(&self) -> Vector3<f64> {
        Vector3::new(self.px, self.py, self.pz)
    }

    fn rotation(&self) -> UnitQuaternion<f64> {
        let q = Quaternion::new(self.qw, self.qx, self.qy, self.qz);
        UnitQuaternion::new_normalize(q)
    }
}

fn read_groundtruth(
) -> Result<(Vec<u128>, Vec<Vector3<f64>>, Vec<UnitQuaternion<f64>>), Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b',')
        .comment(Some(b'#'))
        .from_path("dataset/groundtruth.csv")?;
    let mut timestamps = vec![];
    let mut positions = vec![];
    let mut rotations = vec![];
    for string_record in reader.records() {
        let record: GroundTruth = string_record?.deserialize(None)?;
        timestamps.push(record.timestamp);
        positions.push(record.position());
        rotations.push(record.rotation());
    }
    Ok((timestamps, positions, rotations))
}

fn is_widthin<T: std::cmp::PartialOrd>(timestamps: &[T], timestamp: &T) -> bool {
    let n = timestamps.len();
    if n < 2 {
        return false;
    }
    let first = &timestamps[0];
    let last = &timestamps[n - 1];
    first <= timestamp && timestamp <= last
}

fn has_corresponding_imu_timestamp<T: std::cmp::PartialOrd + std::cmp::Ord>(
    imu_timestamps: &[T],
    query_timestamp: &T,
) -> bool {
    if !is_widthin(&imu_timestamps, query_timestamp) {
        return false;
    }
    imu_timestamps.binary_search(&query_timestamp).is_ok()
}

pub fn integrate_euler(
    timestamps: &[f64],
    angular_velocities: &[Vector3<f64>],
) -> UnitQuaternion<f64> {
    let mut q = UnitQuaternion::identity();
    for i in 0..timestamps.len() - 1 {
        let dt = timestamps[i + 1] - timestamps[i + 0];
        let w = angular_velocities[i];
        let dq = UnitQuaternion::from_scaled_axis(w * dt);
        q = q * dq;
    }
    q
}

fn integrate_midpoint(
    timestamps: &[f64],
    angular_velocities: &[Vector3<f64>],
) -> UnitQuaternion<f64> {
    let mut q = UnitQuaternion::identity();
    for i in 0..timestamps.len() - 1 {
        let dt = timestamps[i + 1] - timestamps[i + 0];
        let w0 = angular_velocities[i + 0];
        let w1 = angular_velocities[i + 1];
        let w = 0.5 * (w0 + w1);
        let dq = UnitQuaternion::from_scaled_axis(w * dt);
        q = q * dq;
    }
    q
}

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

fn jacobian() {}

fn nanosec_to_sec(nanosec: &u128) -> f64 {
    (*nanosec as f64) * 1e-9
}

fn main() -> Result<(), Box<dyn Error>> {
    let (imu_timestamps, accelerations, angular_velocities) = read_imu_data()?;
    // for d in imu_data.iter() {
    //     println!("d = {:?}", d);
    // }

    let (gt_timestamps, positons, rotations) = read_groundtruth()?;
    let index0 = 10001;
    let index1 = index0 + 800;
    if !has_corresponding_imu_timestamp(&imu_timestamps, &gt_timestamps[index0]) {
        println!(
            "index0 = {} does not have a corresponding imu timestamp",
            index0
        );
        return Ok(());
    }
    if !has_corresponding_imu_timestamp(&imu_timestamps, &gt_timestamps[index1]) {
        println!(
            "index1 = {} does not have a corresponding imu timestamp",
            index1
        );
        return Ok(());
    }

    // let imu_timestamps_sec = imu_timestamps[index0..=index1]
    //     .iter()
    //     .map(nanosec_to_sec)
    //     .collect::<Vec<f64>>();
    // let gyro = GyroscopeResidual::new(
    //     &rotations[index0],
    //     &rotations[index1],
    //     imu_timestamps_sec,
    //     angular_velocities[index0..index1].to_vec(),
    // );
    // let bias = Vector3::zeros();
    // let e = gyro.error(&bias);
    // println!("error = {}", e);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::GyroscopeGenerator;
    use core::f64::consts::PI;

    #[test]
    fn test_is_widthin() {
        assert!(!is_widthin(&[], &0));
        assert!(!is_widthin(&[0], &0));

        assert!(is_widthin(&[0, 1], &0));
        assert!(is_widthin(&[0, 1], &1));

        assert!(!is_widthin(&[0, 1], &-1));
        assert!(!is_widthin(&[0, 1], &2));
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
    fn test_integrate_euler() {
        let generator = GyroscopeGenerator::new(time, quat);
        let a = 0;
        let b = 10000;
        let (ta, qa) = generator.rotation(a);
        let (tb, qb) = generator.rotation(b);

        let mut timestamps = vec![];
        let mut angular_velocities = vec![];
        for i in (a..=b).step_by(10) {
            // NOTE b is inclusive
            let (t, w) = generator.angular_velocity(i);
            timestamps.push(t);
            angular_velocities.push(w);
        }
        let dq = integrate_euler(&timestamps, &angular_velocities);
        assert!(f64::abs(((qa * dq).inverse() * qb).angle()) < 1e-4);
    }

    #[test]
    fn test_integrate_midpoint() {
        let generator = GyroscopeGenerator::new(time, quat);
        let a = 0;
        let b = 10000;
        let (ta, qa) = generator.rotation(a);
        let (tb, qb) = generator.rotation(b);

        let mut timestamps = vec![];
        let mut angular_velocities = vec![];
        for i in (a..=b).step_by(10) {
            // NOTE b is inclusive
            let (t, w) = generator.angular_velocity(i);
            timestamps.push(t);
            angular_velocities.push(w);
        }
        let dq = integrate_midpoint(&timestamps, &angular_velocities);
        assert!(f64::abs(((qa * dq).inverse() * qb).angle()) < 1e-4);
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
