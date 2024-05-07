use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use serde::Deserialize;
use std::error::Error;

use imu_preintegration::{estimate_bias, GyroInterface};

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

fn nanosec_to_sec(nanosec: u128) -> f64 {
    (nanosec as f64) * 1e-9
}

fn read_imu_data() -> Result<(Vec<f64>, Vec<Vector3<f64>>, Vec<Vector3<f64>>), Box<dyn Error>> {
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
        timestamps.push(nanosec_to_sec(record.timestamp));
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
) -> Result<(Vec<f64>, Vec<Vector3<f64>>, Vec<UnitQuaternion<f64>>), Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b',')
        .comment(Some(b'#'))
        .from_path("dataset/groundtruth.csv")?;
    let mut timestamps = vec![];
    let mut positions = vec![];
    let mut rotations = vec![];
    for string_record in reader.records() {
        let record: GroundTruth = string_record?.deserialize(None)?;
        timestamps.push(nanosec_to_sec(record.timestamp));
        positions.push(record.position());
        rotations.push(record.rotation());
    }
    Ok((timestamps, positions, rotations))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (imu_timestamps, _accelerations, angular_velocities) = read_imu_data()?;
    let (gt_timestamps, _positions, rotations) = read_groundtruth()?;

    let mut interface = GyroInterface::new();

    for (t, w) in imu_timestamps.iter().zip(angular_velocities.iter()) {
        interface.add_gyroscope(*t, w);
    }

    for (t, q) in gt_timestamps.iter().zip(rotations.iter()).step_by(10) {
        interface.add_reference_pose(*t, q);
    }

    for _ in 0..10 {
        if let Some(r) = interface.pop() {
            let bias_pred = estimate_bias(&r);
            println!("bias_pred = {:?}", bias_pred);
        };
    }

    Ok(())
}
