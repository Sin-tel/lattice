use malachite_bigint::BigInt;
use num_integer::Integer;
use num_traits::ToPrimitive;
use num_traits::{Signed, Zero};
use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::ops::SubAssign;

// TODO: nearest plane
// TODO: saturate, solve_diophantine

// HNF algorithm adapted from https://github.com/lan496/hsnf
// LLL algorithm adapted from https://github.com/orisano/olll

fn swap_rows<T: Clone>(a: &mut Array2<T>, i: usize, j: usize) {
    let a_i = a.row(i).to_owned();
    let a_j = a.row(j).to_owned();
    a.row_mut(i).assign(&a_j);
    a.row_mut(j).assign(&a_i);
}

fn inner_prod(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>, w: ArrayView2<'_, f64>) -> f64 {
    let wb = w.dot(&b);
    a.dot(&wb)
}

fn gramschmidt(v: ArrayView2<'_, i64>, w: ArrayView2<'_, f64>) -> Array2<f64> {
    let v = v.mapv(|x| x as f64);
    let mut u = v.clone();

    for i in 1..v.nrows() {
        let mut ui = u.row(i).to_owned();

        for j in 0..i {
            let uj = u.row(j);

            let proj_coeff =
                inner_prod(uj.view(), v.row(i).view(), w) / inner_prod(uj.view(), uj.view(), w);

            ui -= &(proj_coeff * uj.into_owned());
        }
        u.row_mut(i).assign(&ui);
    }
    u
}

fn lll_inner(basis: ArrayView2<'_, i64>, delta: f64, w: ArrayView2<'_, f64>) -> Array2<i64> {
    let mut basis = basis.to_owned();
    let n = basis.nrows();
    let mut ortho = gramschmidt(basis.view(), w);

    let mu = |basis: &Array2<i64>, ortho: &Array2<f64>, i: usize, j: usize| -> f64 {
        let a = ortho.row(j);
        let b = basis.row(i).mapv(|x| x as f64);
        inner_prod(a, b.view(), w) / inner_prod(a, a.view(), w)
    };

    let mut k = 1;
    while k < n {
        // Size reduction step
        for j in (0..k).rev() {
            let mu_kj = mu(&basis, &ortho, k, j);
            if mu_kj.abs() > 0.5 {
                let mu_int = mu_kj.round() as i64;
                let b_j = basis.row(j).to_owned();
                basis.row_mut(k).sub_assign(&(mu_int * b_j));

                ortho = gramschmidt(basis.view(), w);
            }
        }

        // LLL condition check
        let l_condition = (delta - mu(&basis, &ortho, k, k - 1).powi(2))
            * inner_prod(ortho.row(k - 1), ortho.row(k - 1).view(), w);

        if inner_prod(ortho.row(k), ortho.row(k).view(), w) >= l_condition {
            k += 1;
        } else {
            swap_rows(&mut basis, k, k - 1);

            ortho = gramschmidt(basis.view(), w);

            k = k.saturating_sub(1).max(1);
        }
    }

    basis
}

fn get_pivot(a: ArrayView2<BigInt>, i1: usize, j: usize) -> Option<usize> {
    (i1..a.nrows())
        .filter(|&i| !a[[i, j]].is_zero())
        .min_by_key(|&i| a[[i, j]].abs())
}

fn hnf_inner(mut a: Array2<BigInt>) -> Array2<BigInt> {
    let n = a.nrows();
    let m = a.ncols();
    let mut si = 0;
    let mut sj = 0;

    while si < n && sj < m {
        // Choose a pivot
        match get_pivot(a.view(), si, sj) {
            None => {
                // No non-zero elements, move to next column
                sj += 1;
                continue;
            }
            Some(row) => {
                if row != si {
                    swap_rows(&mut a, si, row);
                }

                // Eliminate column entries below pivot
                for i in si + 1..n {
                    if !a[[i, sj]].is_zero() {
                        let k = &a[[i, sj]] / &a[[si, sj]];
                        for j in 0..m {
                            let a_si_j = a[[si, j]].clone();
                            a[[i, j]] -= &k * a_si_j;
                        }
                    }
                }

                // Check if column is now zero below pivot
                let row_done = (si + 1..n).all(|i| a[[i, sj]].is_zero());

                if row_done {
                    // Ensure pivot is positive
                    if a[[si, sj]].is_negative() {
                        for j in 0..m {
                            a[[si, j]] *= -1;
                        }
                    }

                    // Eliminate entries above pivot
                    if !a[[si, sj]].is_zero() {
                        for i in 0..si {
                            // use floor division to match python `//` semantics
                            let k = a[[i, sj]].div_floor(&a[[si, sj]]);

                            if !k.is_zero() {
                                for j in 0..m {
                                    let a_si_j = &a[[si, j]].clone();
                                    a[[i, j]] -= &k * a_si_j;
                                }
                            }
                        }
                    }

                    // Move to next row and column
                    si += 1;
                    sj += 1;
                }
            }
        }
    }

    a
}

#[pymodule]
fn rslattice<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // TODO: make 2nd and 3rd args Option
    #[pyfn(m)]
    fn lll<'py>(
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, i64>,
        delta: f64,
        w: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray2<i64>> {
        let basis = basis.as_array();
        let w = w.as_array();
        let res = lll_inner(basis, delta, w);
        res.into_pyarray(py)
    }

    #[pyfn(m)]
    fn hnf<'py>(py: Python<'py>, basis: PyReadonlyArray2<'py, i64>) -> Bound<'py, PyArray2<i64>> {
        let basis = basis.as_array();
        let basis = basis.mapv(BigInt::from);
        let res = hnf_inner(basis);

        // TODO: use result instead of expect
        let res = res.mapv(|x| x.to_i64().expect("Overflow error"));

        res.into_pyarray(py)
    }

    Ok(())
}
