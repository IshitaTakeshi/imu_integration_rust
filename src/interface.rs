use crate::generator::GyroscopeGenerator;
use nalgebra::{UnitQuaternion, Vector3};
use std::collections::VecDeque;

struct GyroInterface {
    rotation_timestamps: VecDeque<f64>,
    rotations: VecDeque<UnitQuaternion<f64>>,
    gyroscope_timestamps: VecDeque<f64>,
    angular_velocities: VecDeque<Vector3<f64>>,
}

fn binary_search(target: &VecDeque<f64>, query: f64) -> Result<usize, usize> {
    target.binary_search_by(|t| t.total_cmp(&query))
}

fn interpolate(t0: f64, t1: f64, t: f64, w0: &Vector3<f64>, w1: &Vector3<f64>) -> Vector3<f64> {
    assert!(t1 != t0);
    ((t1 - t) * w0 + (t - t0) * w1) / (t1 - t0)
}

struct Integratable {
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

impl Integratable {
    fn new_interpolated(ts: &[f64], ws: &[Vector3<f64>], ti: f64, tj: f64) -> Self {
        assert_eq!(ts.len(), ws.len());
        let n = ws.len();
        let wi = interpolate(ts[0], ts[1], ti, &ws[0], &ws[1]);
        let wj = interpolate(ts[n - 2], ts[n - 1], tj, &ws[n - 2], &ws[n - 1]);
        let ts_new = concat(ti, &ts[1..n - 1], tj);
        let ws_new = concat(wi, &ws[1..n - 1], wj);
        Integratable {
            ts: ts_new,
            ws: ws_new,
        }
    }
}

impl GyroInterface {
    fn new() -> Self {
        GyroInterface {
            rotation_timestamps: VecDeque::<f64>::new(),
            rotations: VecDeque::<UnitQuaternion<f64>>::new(),
            gyroscope_timestamps: VecDeque::<f64>::new(),
            angular_velocities: VecDeque::<Vector3<f64>>::new(),
        }
    }

    fn add_reference_pose(&mut self, t: f64, q: &UnitQuaternion<f64>) {
        if let Some(&back) = self.rotation_timestamps.back() {
            assert!(t > back);
        };

        self.rotation_timestamps.push_back(t);
        self.rotations.push_back(*q);
    }

    fn add_gyroscope(&mut self, t: f64, w: &Vector3<f64>) {
        if let Some(&back) = self.gyroscope_timestamps.back() {
            assert!(t > back);
        };

        self.gyroscope_timestamps.push_back(t);
        self.angular_velocities.push_back(*w);
    }

    fn get(&mut self) -> Option<(Vec<f64>, Vec<Vector3<f64>>)> {
        if self.gyroscope_timestamps.len() < 2 {
            return None;
        }

        let n = self.gyroscope_timestamps.len();
        let rt0 = self.rotation_timestamps[0];
        let rt1 = self.rotation_timestamps[1];

        if rt0 < self.gyroscope_timestamps[0] {
            return None;
        }
        if rt1 > self.gyroscope_timestamps[n - 1] {
            return None;
        }

        match binary_search(&self.gyroscope_timestamps, rt0) {
            Ok(index) => {
                self.gyroscope_timestamps.drain(..index);
                self.angular_velocities.drain(..index);
            }
            Err(index) => {
                self.gyroscope_timestamps.drain(..index - 1);
                self.angular_velocities.drain(..index - 1);
            }
        }

        let rt1 = self.rotation_timestamps[1];
        match binary_search(&self.gyroscope_timestamps, rt1) {
            Ok(index) => {
                let ts = self
                    .gyroscope_timestamps
                    .drain(..=index)
                    .collect::<Vec<f64>>();
                let ws = self
                    .angular_velocities
                    .drain(..=index)
                    .collect::<Vec<Vector3<f64>>>();
                return Some((ts, ws));
            }
            Err(index) => {
                let mut ts = self
                    .gyroscope_timestamps
                    .drain(..index)
                    .collect::<Vec<f64>>();
                let mut ws = self
                    .angular_velocities
                    .drain(..index)
                    .collect::<Vec<Vector3<f64>>>();
                let t = self.gyroscope_timestamps[0];
                let w = self.angular_velocities[0];
                ts.push(t);
                ws.push(w);
                return Some((ts, ws));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Quaternion;

    const PI: f64 = std::f64::consts::PI;

    #[test]
    fn test_binary_search() {
        let mut v = VecDeque::new();
        v.extend(&[0.2, 0.4, 0.6, 0.8]);
        assert_eq!(binary_search(&v, 0.1), Err(0));
        assert_eq!(binary_search(&v, 0.5), Err(2));
        assert_eq!(binary_search(&v, 0.8), Ok(3));
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

    const DELTA_T: f64 = 0.01;
    fn time(i: usize) -> f64 {
        DELTA_T * (i as f64)
    }

    #[test]
    fn test_get_without_time_offset() {
        let mut interface = GyroInterface::new();

        let generator = GyroscopeGenerator::new(time, quat);

        for i in 0..10 {
            let (t, omega) = generator.angular_velocity(i);
            interface.add_gyroscope(t, &omega);
        }

        let (ta, qa) = generator.rotation(3);
        interface.add_reference_pose(ta, &qa);

        let (tb, qb) = generator.rotation(7);
        interface.add_reference_pose(tb, &qb);

        match interface.get() {
            Some((ts, _ws)) => assert_eq!(ts, [0.03, 0.04, 0.05, 0.06, 0.07]),
            None => assert!(false),
        }
    }

    #[test]
    fn test_get_start_reference_time_too_early() {
        let mut interface = GyroInterface::new();

        let generator = GyroscopeGenerator::new(time, quat);

        for i in 5..10 {
            let (t, omega) = generator.angular_velocity(i);
            interface.add_gyroscope(t, &omega);
        }

        let (ta, qa) = generator.rotation(4);
        interface.add_reference_pose(ta, &qa);

        let (tb, qb) = generator.rotation(7);
        interface.add_reference_pose(tb, &qb);

        assert_eq!(interface.get(), None);
    }

    #[test]
    fn test_get_end_reference_time_too_late() {
        let mut interface = GyroInterface::new();

        let generator = GyroscopeGenerator::new(time, quat);

        for i in 0..10 {
            let (t, omega) = generator.angular_velocity(i);
            interface.add_gyroscope(t, &omega);
        }

        let (ta, qa) = generator.rotation(4);
        interface.add_reference_pose(ta, &qa);

        let (tb, qb) = generator.rotation(10);
        interface.add_reference_pose(tb, &qb);

        assert_eq!(interface.get(), None);
    }

    #[test]
    fn test_get_start_timestamp_matches() {
        let mut interface = GyroInterface::new();

        let generator = GyroscopeGenerator::new(time, quat);

        for i in 5..10 {
            let (t, omega) = generator.angular_velocity(i);
            interface.add_gyroscope(t, &omega);
        }

        let (ta, qa) = generator.rotation(5);
        interface.add_reference_pose(ta, &qa);

        let (tb, qb) = generator.rotation(7);
        interface.add_reference_pose(tb, &qb);

        match interface.get() {
            Some((ts, ws)) => assert_eq!(ts, [0.05, 0.06, 0.07]),
            None => assert!(false),
        }
    }

    #[test]
    fn test_get_with_both_time_offsets() {
        let mut interface = GyroInterface::new();

        let generator = GyroscopeGenerator::new(time, quat);

        for i in [2, 4, 6, 8, 10, 12].iter() {
            let (t, omega) = generator.angular_velocity(*i);
            interface.add_gyroscope(t, &omega);
        }

        let (ta, qa) = generator.rotation(5);
        interface.add_reference_pose(ta, &qa);

        let (tb, qb) = generator.rotation(9);
        interface.add_reference_pose(tb, &qb);

        match interface.get() {
            Some((ts, ws)) => assert_eq!(ts, [0.04, 0.06, 0.08, 0.10]),
            None => assert!(false),
        }
    }

    #[test]
    fn test_get_only_start_time_offset() {
        let mut interface = GyroInterface::new();

        let generator = GyroscopeGenerator::new(time, quat);

        for i in [2, 4, 6, 8, 10, 12].iter() {
            let (t, omega) = generator.angular_velocity(*i);
            interface.add_gyroscope(t, &omega);
        }

        let (ta, qa) = generator.rotation(5);
        interface.add_reference_pose(ta, &qa);

        let (tb, qb) = generator.rotation(10);
        interface.add_reference_pose(tb, &qb);

        // Include 0.04 to generate the interpolated angular velocity for 0.05
        match interface.get() {
            Some((ts, ws)) => assert_eq!(ts, [0.04, 0.06, 0.08, 0.10]),
            None => assert!(false),
        }
    }

    #[test]
    fn test_get_only_end_time_offset() {
        let mut interface = GyroInterface::new();

        let generator = GyroscopeGenerator::new(time, quat);

        for i in [2, 4, 6, 8, 10, 12, 14].iter() {
            let (t, omega) = generator.angular_velocity(*i);
            interface.add_gyroscope(t, &omega);
        }

        let (ta, qa) = generator.rotation(4);
        interface.add_reference_pose(ta, &qa);

        let (tb, qb) = generator.rotation(9);
        interface.add_reference_pose(tb, &qb);

        // Include 0.10 to generate the interpolated angular velocity for 0.09
        match interface.get() {
            Some((ts, ws)) => assert_eq!(ts, [0.04, 0.06, 0.08, 0.10]),
            None => assert!(false),
        }
    }

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
}
