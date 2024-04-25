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

fn compute_timestamps(timestamps: &[f64], ti: f64, tj: f64) -> Vec<f64> {
    check_timestamps(timestamps, ti, tj);

    let n = timestamps.len();

    let mut ts = Vec::<f64>::new();
    ts.push(ti);
    ts.extend(&timestamps[1..n - 1]);
    ts.push(tj);
    ts
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
            let omega = self.angular_velocities[i];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_widthin() {
        assert!(!is_widthin(&[], &0));
        assert!(!is_widthin(&[0], &0));

        assert!(is_widthin(&[0, 1], &0));
        assert!(is_widthin(&[0, 1], &1));

        assert!(!is_widthin(&[0, 1], &-1));
        assert!(!is_widthin(&[0, 1], &2));
    }

    #[test]
    fn test_compute_timestamps() {
        let timestamps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let ti = 0.12;
        let tj = 0.57;
        let timestamps = compute_timestamps(&timestamps, ti, tj);
        let expected = [0.12, 0.2, 0.3, 0.4, 0.5, 0.57];
        assert_eq!(timestamps.len(), expected.len());

        for (t, e) in timestamps.iter().zip(expected.iter()) {
            assert!(f64::abs(t - e) < 1e-16);
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
