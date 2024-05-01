use crate::integration;
use crate::interpolation;
use nalgebra::{UnitQuaternion, Vector3};

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
