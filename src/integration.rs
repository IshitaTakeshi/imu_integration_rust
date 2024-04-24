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

struct GyroInterface {
    r_wb_i: Option<UnitQuaternion<f64>>,
    r_wb_j: Option<UnitQuaternion<f64>>,
    ti: Option<f64>,
    tj: Option<f64>,
    timestamps: Vec<f64>,
    angular_velocities: Vec<Vector3<f64>>,
}

fn interpolate(t0: f64, t1: f64, t: f64, w0: &Vector3<f64>, w1: &Vector3<f64>) -> Vector3<f64> {
    assert!(t1 != t0);
    ((t1 - t) * w0 + (t - t0) * w1) / (t1 - t0)
}

fn check_timestamps(timestamps: &[f64], ti: f64, tj: f64) {
    let n = timestamps.len();
    assert!(n >= 3);
    assert!(timestamps[0] <= ti);
    assert!(ti < timestamps[1]);
    assert!(timestamps[n - 2] < tj);
    assert!(tj <= timestamps[n - 1]);
}

fn compute_dts(timestamps: &[f64], ti: f64, tj: f64) -> Vec<f64> {
    check_timestamps(timestamps, ti, tj);

    let n = timestamps.len();

    let mut dts = Vec::<f64>::new();
    dts.push(timestamps[1] - ti);
    for k in 1..n - 2 {
        dts.push(timestamps[k + 1] - timestamps[k]);
    }
    dts.push(tj - timestamps[n - 2]);
    dts
}

impl GyroInterface {
    fn new() -> Self {
        GyroInterface {
            r_wb_i: None,
            r_wb_j: None,
            ti: None,
            tj: None,
            timestamps: Vec::<f64>::new(),
            angular_velocities: Vec::<Vector3<f64>>::new(),
        }
    }
    fn add_start_reference_pose(&mut self, t: f64, q: &UnitQuaternion<f64>) {
        self.ti = Some(t);
        self.r_wb_i = Some(*q);
    }

    fn add_end_reference_pose(&mut self, t: f64, q: &UnitQuaternion<f64>) {
        self.tj = Some(t);
        self.r_wb_j = Some(*q);
    }

    fn add_gyroscope(&mut self, t: f64, w: &Vector3<f64>) {
        match self.timestamps.last() {
            Some(last) => assert!(*last < t),
            None => (),
        }
        self.timestamps.push(t);
        self.angular_velocities.push(*w);
    }

    fn make(&mut self) -> Option<Gyro> {
        let n = self.angular_velocities.len();
        let tf0 = self.timestamps[0];
        let tf1 = self.timestamps[1];
        let tl0 = self.timestamps[n - 2];
        let tl1 = self.timestamps[n - 1];
        let wf0 = self.angular_velocities[0];
        let wf1 = self.angular_velocities[1];
        let wl0 = self.angular_velocities[n - 2];
        let wl1 = self.angular_velocities[n - 1];

        let mut angular_velocities = &mut self.angular_velocities;
        angular_velocities[0] = interpolate(tf0, tf1, self.ti?, &wf0, &wf1);
        angular_velocities[n - 1] = interpolate(tl0, tl1, self.tj?, &wl0, &wl1);

        let dts = compute_dts(&self.timestamps, self.ti?, self.tj?);
        let gyro = Gyro::new(
            &self.r_wb_i?,
            &self.r_wb_j?,
            dts,
            angular_velocities.to_vec(),
        );
        Some(gyro)
    }
}

struct Gyro {
    r_wb_i: UnitQuaternion<f64>,
    r_wb_j: UnitQuaternion<f64>,
    dts: Vec<f64>,
    angular_velocities: Vec<Vector3<f64>>,
}

impl Gyro {
    fn new(
        r_wb_i: &UnitQuaternion<f64>,
        r_wb_j: &UnitQuaternion<f64>,
        dts: Vec<f64>,
        angular_velocities: Vec<Vector3<f64>>,
    ) -> Gyro {
        Gyro {
            r_wb_i: *r_wb_i,
            r_wb_j: *r_wb_j,
            dts,
            angular_velocities,
        }
    }

    fn integrate(&self) {
        for i in 0..self.dts.len() {
            let dt = self.dts[i];
        }
    }

    // fn residual(&self, bias: &Vector3<f64>) -> Vector3<f64> {
    //     let mut q = self.r_wb_i.inverse() * self.r_wb_j;
    //     for i in 0..self.angular_velocities.len() {
    //         let w = self.angular_velocities[i];
    //         let dt = self.timestamps[i + 1] - self.timestamps[i];
    //         let dq = UnitQuaternion::from_scaled_axis((w - bias) * dt);
    //         q = q * dq;
    //     }
    //     q.scaled_axis()
    // }

    // fn error(&self, bias: &Vector3<f64>) -> f64 {
    //     let r = self.residual(bias);
    //     r.norm_squared()
    // }
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

    let imu_timestamps_sec = imu_timestamps[index0..=index1]
        .iter()
        .map(nanosec_to_sec)
        .collect::<Vec<f64>>();
    let gyro = Gyro::new(
        &rotations[index0],
        &rotations[index1],
        imu_timestamps_sec,
        angular_velocities[index0..index1].to_vec(),
    );
    // let bias = Vector3::zeros();
    // let e = gyro.error(&bias);
    // println!("error = {}", e);

    Ok(())
}

struct GyroscopeGenerator {
    t: fn(usize) -> f64,
    f: fn(f64) -> UnitQuaternion<f64>,
}

impl GyroscopeGenerator {
    fn new(t: fn(usize) -> f64, f: fn(f64) -> UnitQuaternion<f64>) -> Self {
        GyroscopeGenerator { t: t, f: f }
    }

    fn get(&self, i: usize) -> (Vector3<f64>, f64) {
        let t0 = (self.t)(i + 0);
        let t1 = (self.t)(i + 1);
        let q0 = (self.f)(t0);
        let q1 = (self.f)(t1);
        let dq = q0.inverse() * q1;

        // Compute the numerical derivative of the rotation
        let dt = t1 - t0;
        let omega = dq.scaled_axis() / dt;
        (omega, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exp_so3;

    #[test]
    fn test_is_widthin() {
        assert!(!is_widthin(&[], &0));
        assert!(!is_widthin(&[0], &0));

        assert!(is_widthin(&[0, 1], &0));
        assert!(is_widthin(&[0, 1], &1));

        assert!(!is_widthin(&[0, 1], &-1));
        assert!(!is_widthin(&[0, 1], &2));
    }

    const PI: f64 = std::f64::consts::PI;

    fn quat(t: f64) -> UnitQuaternion<f64> {
        let x = f64::sin(2. * PI * 1. * t);
        let y = f64::sin(2. * PI * 2. * t);
        let z = f64::sin(2. * PI * 3. * t);
        let w = 1.0 - (x * x + y * y + z * z);
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

        let a = 0;
        let b = 1000;
        let qa = quat(time(a));
        let qb = quat(time(b));

        let generator = GyroscopeGenerator::new(time, quat);
        let mut q = qa;
        for i in a..b {
            let (omega, dt) = generator.get(i);
            let dq = UnitQuaternion::from_scaled_axis(omega * dt);

            q = q * dq;
        }

        assert!(f64::abs((qb.inverse() * q).angle()) < 1e-8);
    }

    #[test]
    fn test_residual_without_time_offset() {
        let a = 0;
        let b = 1000;
        let ta = time(a);
        let tb = time(b);
        let qa = quat(ta);
        let qb = quat(tb);

        let mut interface = GyroInterface::new();
        interface.add_start_reference_pose(ta, &qa);

        let generator = GyroscopeGenerator::new(time, quat);
        for i in a..b {
            let (omega, dt) = generator.get(i);
            let t0 = time(i + 0);
            interface.add_gyroscope(t0, &omega);
        }

        interface.add_end_reference_pose(tb, &qb);
    }

    #[test]
    fn test_compute_dts() {
        let timestamps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let ti = 0.12;
        let tj = 0.57;
        let dts = compute_dts(&timestamps, ti, tj);
        let expected = [0.08, 0.1, 0.1, 0.1, 0.07];
        assert_eq!(dts.len(), expected.len());

        for (dt, e) in dts.iter().zip(expected.iter()) {
            assert!(f64::abs(dt - e) < 1e-16);
        }
    }

    // #[test]
    // fn test_residual_with_time_offset() {
    //     let gyro = Gyro::new();

    //     let q = quat(time(0));
    //     let q = quat(time(100));

    //     gyro.add_start_reference_pose();

    //     for i in 0..100 {
    //         let t = time(i);
    //         let q = quat(t);
    //         gyro.add_gyroscope(t, q);
    //     }

    //     gyro.add_end_reference_pose(tb, qb);
    // }
}
