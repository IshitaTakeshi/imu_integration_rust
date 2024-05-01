#![feature(stmt_expr_attributes)]

mod generator;
mod integration;
mod interface;
mod interpolation;
mod residual;

use nalgebra::{ArrayStorage, Const, Matrix, SMatrix, Vector3};

pub type Vector9<T> = Matrix<T, Const<9>, Const<1>, ArrayStorage<T, 9, 1>>;

const GRAVITY: Vector3<f64> = Vector3::<f64>::new(0.0, 0.0, -9.8);

// fn get_rot(transform: &SMatrix::<f64, 5, 5>) -> SMatrix::<f64, 3, 3> {
//     transform.fixed_view::<3, 3>(0, 0)
// }

pub fn identity<const D: usize>() -> SMatrix<f64, D, D> {
    SMatrix::<f64, D, D>::identity()
}

fn gamma(t: f64) -> SMatrix<f64, 5, 5> {
    let mut gamma = identity::<5>();
    gamma.fixed_view_mut::<3, 1>(0, 3).copy_from(&(GRAVITY * t));
    gamma
        .fixed_view_mut::<3, 1>(0, 4)
        .copy_from(&(0.5 * GRAVITY * t * t));
    gamma
}

fn phi(transform: &SMatrix<f64, 5, 5>, dt: f64) -> SMatrix<f64, 5, 5> {
    let v = transform.fixed_view::<3, 1>(0, 3);
    let mut d = SMatrix::<f64, 5, 5>::zeros();
    d.fixed_view_mut::<3, 1>(0, 4).copy_from(&(v * dt));
    transform + d
}

fn make_f(t: f64) -> SMatrix<f64, 9, 9> {
    let identity3 = SMatrix::<f64, 3, 3>::identity();
    let mut f = SMatrix::<f64, 9, 9>::identity();
    f.fixed_view_mut::<3, 3>(6, 3).copy_from(&(identity3 * t));
    f
}

fn wedge_so3(v: &Vector3<f64>) -> SMatrix<f64, 3, 3> {
    #[rustfmt::skip]
    SMatrix::<f64, 3, 3>::new(
        0., -v[2], v[1],
        v[2], 0., -v[0],
        -v[1], v[0], 0.
    )
}

fn wedge_se23(xi: &Vector9<f64>) -> SMatrix<f64, 5, 5> {
    let phi = xi.fixed_view::<3, 1>(0, 0).into();
    let nu = xi.fixed_view::<3, 1>(3, 0);
    let rho = xi.fixed_view::<3, 1>(6, 0);

    let wedge_phi = wedge_so3(&phi);

    let mut wedge = SMatrix::<f64, 5, 5>::zeros();
    wedge.fixed_view_mut::<3, 3>(0, 0).copy_from(&wedge_phi);
    wedge.fixed_view_mut::<3, 1>(0, 3).copy_from(&nu);
    wedge.fixed_view_mut::<3, 1>(0, 4).copy_from(&rho);
    wedge
}

fn left_jacobian(rotvec: &Vector3<f64>) -> SMatrix<f64, 3, 3> {
    let epsilon = 1e-6;
    let norm = rotvec.norm();

    let identity3 = identity::<3>();
    let wedge = wedge_so3(&rotvec);

    if norm < epsilon {
        return identity3 + 0.5 * wedge;
    }

    identity3
        + ((1. - f64::cos(norm)) / (norm * norm)) * wedge
        + ((norm - f64::sin(norm)) / (norm * norm * norm)) * wedge * wedge
}

fn inv_left_jacobian(rotvec: &Vector3<f64>) -> SMatrix<f64, 3, 3> {
    let epsilon = 1e-6;
    let norm = rotvec.norm();

    let identity3 = identity::<3>();
    let wedge = wedge_so3(&rotvec);

    if norm < epsilon {
        return identity3 - 0.5 * wedge;
    }

    let k = 1. / (norm * norm) - (1. + f64::cos(norm)) / (2. * norm * f64::sin(norm));
    identity3 - 0.5 * wedge + k * wedge * wedge
}

fn right_jacobian(rotvec: &Vector3<f64>) -> SMatrix<f64, 3, 3> {
    left_jacobian(&(-rotvec))
}

fn inv_right_jacobian(rotvec: &Vector3<f64>) -> SMatrix<f64, 3, 3> {
    inv_left_jacobian(&(-rotvec))
}

fn exp_so3(rotvec: &Vector3<f64>) -> SMatrix<f64, 3, 3> {
    let epsilon = 1e-6;
    let norm = rotvec.norm();

    let identity3 = identity::<3>();
    if norm < epsilon {
        return identity3 + wedge_so3(&rotvec);
    }
    let wedge = wedge_so3(&(rotvec / norm));
    identity3 + f64::sin(norm) * wedge + (1. - f64::cos(norm)) * wedge * wedge
}

fn exp_se23(xi: &Vector9<f64>) -> SMatrix<f64, 5, 5> {
    let phi = xi.fixed_view::<3, 1>(0, 0).into();
    let nu = xi.fixed_view::<3, 1>(3, 0);
    let rho = xi.fixed_view::<3, 1>(6, 0);

    let jacobian = left_jacobian(&phi);
    let rot = exp_so3(&phi);

    let mut transform = SMatrix::<f64, 5, 5>::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rot);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&(jacobian * nu));
    transform
        .fixed_view_mut::<3, 1>(0, 4)
        .copy_from(&(jacobian * rho));
    transform
}

fn inverse_se23(transform: &SMatrix<f64, 5, 5>) -> SMatrix<f64, 5, 5> {
    let nu_rho = transform.fixed_view::<3, 2>(0, 3);
    let rot = transform.fixed_view::<3, 3>(0, 0);
    let mut inverse = identity::<5>();
    inverse
        .fixed_view_mut::<3, 2>(0, 3)
        .copy_from(&(-rot.transpose() * nu_rho));
    inverse
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rot.transpose());
    inverse
}

fn adjoint(transform: &SMatrix<f64, 5, 5>) -> SMatrix<f64, 9, 9> {
    let rot = transform.fixed_view::<3, 3>(0, 0);
    let v = transform.fixed_view::<3, 1>(0, 3).into();
    let p = transform.fixed_view::<3, 1>(0, 4).into();

    let mut adjoint = SMatrix::<f64, 9, 9>::zeros();
    adjoint.fixed_view_mut::<3, 3>(0, 0).copy_from(&(rot));
    adjoint.fixed_view_mut::<3, 3>(3, 3).copy_from(&(rot));
    adjoint.fixed_view_mut::<3, 3>(6, 6).copy_from(&(rot));
    adjoint
        .fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&(wedge_so3(&v) * rot));
    adjoint
        .fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&(wedge_so3(&p) * rot));
    adjoint
}

fn upsilon(
    angular_velocity: &Vector3<f64>,
    acceleration: &Vector3<f64>,
    dt: f64,
) -> SMatrix<f64, 5, 5> {
    let mut upsilon = identity::<5>();
    upsilon
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&exp_so3(&(angular_velocity * dt)));
    upsilon
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&(acceleration * dt));
    upsilon
        .fixed_view_mut::<3, 1>(0, 4)
        .copy_from(&(0.5 * acceleration * dt * dt));
    upsilon
}

pub fn propagate(
    transform: &SMatrix<f64, 5, 5>,
    angular_velocity: &Vector3<f64>,
    acceleration: &Vector3<f64>,
    dt: f64,
) -> SMatrix<f64, 5, 5> {
    let gamma = gamma(dt);
    let upsilon = upsilon(angular_velocity, acceleration, dt);
    gamma * phi(transform, dt) * upsilon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma() {
        let gamma = gamma(3.0);
        assert_eq!(gamma[(2, 3)], -9.8 * 3.0);
        assert_eq!(gamma[(2, 4)], -0.5 * 9.8 * 3.0 * 3.0);
    }

    #[test]
    fn test_make_f() {
        let f = make_f(2.);

        assert_eq!(f.nrows(), 9);
        assert_eq!(f.ncols(), 9);
        assert_eq!(f.fixed_view::<3, 3>(6, 3), 2. * identity::<3>());

        for i in 0..9 {
            assert_eq!(f[(i, i)], 1.);
        }
    }

    #[test]
    fn test_wedge_so3() {
        let mat = wedge_so3(&Vector3::new(0.3, 0.5, 0.7));

        #[rustfmt::skip]
        let expected = SMatrix::<f64, 3, 3>::new(
            0.0, -0.7, 0.5,
            0.7, 0.0, -0.3,
            -0.5, 0.3, 0.0
        );
        assert_eq!(mat, expected);
    }

    fn numerical_exp_se23(xi: &Vector9<f64>) -> SMatrix<f64, 5, 5> {
        let n_max = 200;
        let wedge = wedge_se23(xi);

        let mut exp = SMatrix::<f64, 5, 5>::identity();
        for n in 1..n_max {
            let mut power = SMatrix::<f64, 5, 5>::identity();
            for i in 1..=n {
                power = power * wedge / (i as f64);
            }
            exp += power;
        }
        exp
    }

    #[test]
    fn test_phi() {
        #[rustfmt::skip]
        let transform = SMatrix::<f64, 5, 5>::new(
            1., 0., 0., 2., 6.,
            0., 1., 0., 4., 5.,
            0., 0., 1., 6., 4.,
            0., 0., 0., 1., 0.,
            0., 0., 0., 0., 1.,
        );

        #[rustfmt::skip]
        let expected = SMatrix::<f64, 5, 5>::new(
            1., 0., 0., 2., 7.,
            0., 1., 0., 4., 7.,
            0., 0., 1., 6., 7.,
            0., 0., 0., 1., 0.,
            0., 0., 0., 0., 1.,
        );

        let phi = phi(&transform, 0.5);
        assert_eq!(phi, expected);
    }

    #[test]
    fn test_wedge_se23() {
        let xi = Vector9::<f64>::from_data(ArrayStorage([[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        ]]));

        let wedge = wedge_se23(&xi);

        #[rustfmt::skip]
        let expected = SMatrix::<f64, 5, 5>::new(
            0.0, -0.3, 0.2, 0.4, 0.7,
            0.3, 0.0, -0.1, 0.5, 0.8,
            -0.2, 0.1, 0.0, 0.6, 0.9,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        );

        assert_eq!(wedge, expected);
    }

    #[test]
    fn test_left_jacobian() {
        // test the definition of the exponential map of se(3)
        //
        // exp_se23([phi, nu, rho])
        //   [exp_so3(phi)  JL(phi) * nu  JL(phi) * rho ]
        // = [0    0    0         1             0       ]
        //   [0    0    0         0             1       ]
        let xi = Vector9::<f64>::from_data(ArrayStorage([[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        ]]));

        let exp = numerical_exp_se23(&xi);

        let phi = xi.fixed_view::<3, 1>(0, 0).into();
        let nu = xi.fixed_view::<3, 1>(3, 0);
        let rho = xi.fixed_view::<3, 1>(6, 0);

        let j = left_jacobian(&phi);

        assert!((j * nu - exp.fixed_view::<3, 1>(0, 3)).norm() < 1e-8);
        assert!((j * rho - exp.fixed_view::<3, 1>(0, 4)).norm() < 1e-8);
    }

    #[test]
    fn test_left_jacobian_small_phi() {
        let xi = Vector9::<f64>::from_data(ArrayStorage([[
            1e-8, 2e-8, 3e-8, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        ]]));

        let exp = numerical_exp_se23(&xi);

        let phi = xi.fixed_view::<3, 1>(0, 0).into();
        let nu = xi.fixed_view::<3, 1>(3, 0);
        let rho = xi.fixed_view::<3, 1>(6, 0);

        let j = left_jacobian(&phi);

        assert!((j * nu - exp.fixed_view::<3, 1>(0, 3)).norm() < 1e-10);
        assert!((j * rho - exp.fixed_view::<3, 1>(0, 4)).norm() < 1e-10);
    }

    #[test]
    fn test_inv_left_jacobian() {
        let theta = Vector3::new(0.3, 0.1, 0.2);
        let j = left_jacobian(&theta);
        let inv_j = inv_left_jacobian(&theta);

        assert!((j * inv_j - SMatrix::<f64, 3, 3>::identity()).norm() < 1e-12);
    }

    #[test]
    fn test_inv_left_jacobian_small_phi() {
        let theta = Vector3::new(1e-8, 1e-7, 2e-7);
        let j = left_jacobian(&theta);
        let inv_j = inv_left_jacobian(&theta);

        assert!((j * inv_j - SMatrix::<f64, 3, 3>::identity()).norm() < 1e-12);
    }

    #[test]
    fn test_inv_right_jacobian() {
        let theta = Vector3::new(0.3, 0.1, 0.2);
        let j = right_jacobian(&theta);
        let inv_j = inv_right_jacobian(&theta);

        assert!((j * inv_j - SMatrix::<f64, 3, 3>::identity()).norm() < 1e-12);
    }

    #[test]
    fn test_inv_right_jacobian_small_phi() {
        let theta = Vector3::new(1e-8, 1e-7, 2e-7);
        let j = right_jacobian(&theta);
        let inv_j = inv_right_jacobian(&theta);

        assert!((j * inv_j - SMatrix::<f64, 3, 3>::identity()).norm() < 1e-12);
    }

    #[test]
    fn test_adjoint() {
        let xi = Vector9::<f64>::from_data(ArrayStorage([[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        ]]));

        let v = Vector9::<f64>::from_data(ArrayStorage([[
            0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
        ]]));
        let transform = numerical_exp_se23(&v);
        let inv_transform = numerical_exp_se23(&(-v));

        let m1 = transform * numerical_exp_se23(&xi) * inv_transform;
        let m2 = numerical_exp_se23(&(adjoint(&transform) * xi));
        assert!((m1 - m2).norm() < 1e-6);
    }

    #[test]
    fn test_inverse_se23() {
        let v = Vector9::<f64>::from_data(ArrayStorage([[
            0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
        ]]));
        let transform = numerical_exp_se23(&v);
        let inv_transform = numerical_exp_se23(&(-v));
        assert!((inverse_se23(&transform) - inv_transform).norm() < 1e-8);
    }

    #[test]
    fn test_upsilon() {
        let dt = 0.5;
        let angular_velocity = Vector3::<f64>::new(0., 0., 0.2);
        let acceleration = Vector3::<f64>::new(0.5, 0.3, 0.1);

        #[rustfmt::skip]
        let expected = SMatrix::<f64, 5, 5>::new(
            0.99500417, -0.09983342, 0.00000000, 0.2500, 0.0625,
            0.09983342,  0.99500417, 0.00000000, 0.1500, 0.0375,
            0.00000000,  0.00000000, 1.00000000, 0.0500, 0.0125,
            0.00000000,  0.00000000, 0.00000000, 1.0000, 0.0000,
            0.00000000,  0.00000000, 0.00000000, 0.0000, 1.0000,
        );

        let upsilon = upsilon(&angular_velocity, &acceleration, dt);

        assert!((expected - upsilon).norm() < 1e-4);
    }
}
