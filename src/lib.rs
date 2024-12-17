use malachite_bigint::BigInt;
use num_integer::Integer;
use num_traits::ToPrimitive;
use num_traits::{Signed, Zero};
use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use std::ops::SubAssign;

// HNF algorithm adapted from https://github.com/lan496/hsnf
// LLL algorithm adapted from https://github.com/orisano/olll

fn swap_rows<T: Clone>(a: &mut Array2<T>, i: usize, j: usize) {
    let a_i = a.row(i).to_owned();
    let a_j = a.row(j).to_owned();
    a.row_mut(i).assign(&a_j);
    a.row_mut(j).assign(&a_i);
}

fn inner_prod(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>, w: ArrayView2<'_, f64>) -> f64 {
    a.dot(&w.dot(&b))
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

fn mu(basis: &Array2<i64>, ortho: &Array2<f64>, w: ArrayView2<'_, f64>, i: usize, j: usize) -> f64 {
    let a = ortho.row(j);
    let b = basis.row(i).mapv(|x| x as f64);
    inner_prod(a, b.view(), w) / inner_prod(a, a.view(), w)
}

fn lll_inner(basis: ArrayView2<'_, i64>, delta: f64, w: ArrayView2<'_, f64>) -> Array2<i64> {
    let mut basis = basis.to_owned();
    let n = basis.nrows();
    let mut ortho = gramschmidt(basis.view(), w);

    let mut k = 1;
    while k < n {
        // Size reduction step
        for j in (0..k).rev() {
            let mu_kj = mu(&basis, &ortho, w, k, j);
            if mu_kj.abs() > 0.5 {
                let mu_int = mu_kj.round_ties_even() as i64;
                let b_j = basis.row(j).to_owned();
                basis.row_mut(k).sub_assign(&(mu_int * b_j));

                ortho = gramschmidt(basis.view(), w);
            }
        }

        // LLL condition check
        let l_condition = (delta - mu(&basis, &ortho, w, k, k - 1).powi(2))
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

// Babai's nearest plane algorithm for solving approximate CVP
fn nearest_plane_inner(
    v: ArrayView1<'_, i64>,
    basis: ArrayView2<'_, i64>,
    w: ArrayView2<'_, f64>,
) -> Array1<i64> {
    let mut b = v.to_owned();
    let n = basis.shape()[0];

    let ortho = gramschmidt(basis.view(), w);

    for j in (0..n).rev() {
        let a = ortho.row(j);
        let b_f64 = b.mapv(|x| x as f64);
        let mu = inner_prod(a, b_f64.view(), w) / inner_prod(a, a.view(), w);
        let mu_int = mu.round_ties_even() as i64;
        let basis_j = basis.row(j).to_owned();
        b = b - mu_int * basis_j;
    }

    v.to_owned() - b
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
                for i in (si + 1)..n {
                    if !a[[i, sj]].is_zero() {
                        let k = &a[[i, sj]] / &a[[si, sj]];
                        for j in 0..m {
                            let a_si_j = a[[si, j]].clone();
                            a[[i, j]] -= &k * a_si_j;
                        }
                    }
                }

                // Check if column is now zero below pivot
                let row_done = ((si + 1)..n).all(|i| a[[i, sj]].is_zero());

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

// Doing these checks barely registers in the benches so it's all good
fn checked_op(a: &Array2<i64>, i: usize, j: usize, k: usize) -> Option<i64> {
    let term1 = a[[j, k]].checked_mul(a[[i, i]])?;
    let term2 = a[[j, i]].checked_mul(a[[i, k]])?;
    term1.checked_sub(term2)
}

// exact integer determinant using Bareiss algorithm
// https://stackoverflow.com/questions/66192894/precise-determinant-of-integer-nxn-matrix
fn integer_det_inner(a: ArrayView2<i64>) -> Result<i64, String> {
    let mut a = a.to_owned();
    let mut sign = 1;
    let mut prev = 1;
    let n = a.shape()[0];

    for i in 0..(n - 1) {
        if a[[i, i]] == 0 {
            let swap = ((i + 1)..n).find(|&j| a[[j, i]] != 0);
            match swap {
                Some(k) => {
                    swap_rows(&mut a, i, k);
                    sign *= -1
                }
                // Whole row is zero => det = 0
                None => return Ok(0),
            }
        }
        for j in (i + 1)..n {
            for k in (i + 1)..n {
                // naive, will overflow
                // let d = a[[j, k]] * a[[i, i]] - a[[j, i]] * a[[i, k]];

                // slower method that results in slightly less overflows

                // let d = BigInt::from(a[[j, k]]) * BigInt::from(a[[i, i]])
                //     - BigInt::from(a[[j, i]]) * BigInt::from(a[[i, k]]);
                // let d2 = Integer::div_floor(&d, &BigInt::from(prev));
                // a[[j, k]] = d2.to_i64().ok_or_else(|| "Overflow error")?;

                let d = checked_op(&a, i, j, k);
                match d {
                    Some(d) => {
                        // use floor division to match python `//` semantics
                        // note: div_floor exists in standard library but is currently unstable
                        // https://doc.rust-lang.org/std/primitive.i64.html#method.div_floor

                        assert!(d % prev == 0);
                        a[[j, k]] = Integer::div_floor(&d, &prev);
                    }
                    None => return Err("Overflow error".to_string()),
                }
            }
        }

        prev = a[[i, i]];
    }

    Ok(sign * a[[n - 1, n - 1]])
}

fn bigint_to_i64(big_array: Array2<BigInt>) -> Result<Array2<i64>, String> {
    let mut i64_array = Array2::zeros(big_array.dim());

    for ((row, col), big_val) in big_array.indexed_iter() {
        let i64_val = big_val.to_i64().ok_or_else(|| {
            format!(
                "Overflow at position [{}, {}]: {} cannot fit in i64",
                row, col, big_val
            )
        })?;
        i64_array[[row, col]] = i64_val;
    }

    Ok(i64_array)
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
    fn hnf<'py>(
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let basis = basis.as_array();
        let basis = basis.mapv(BigInt::from);
        let res = hnf_inner(basis);

        let res = bigint_to_i64(res);

        match res {
            Ok(m) => Ok(m.into_pyarray(py)),
            Err(e) => Err(PyErr::new::<PyOverflowError, _>(e)),
        }
    }

    #[pyfn(m)]
    fn nearest_plane<'py>(
        py: Python<'py>,
        v: PyReadonlyArray1<'py, i64>,
        basis: PyReadonlyArray2<'py, i64>,
        w: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<i64>> {
        let v = v.as_array();
        let basis = basis.as_array();
        let w = w.as_array();
        let res = nearest_plane_inner(v, basis, w);
        res.into_pyarray(py)
    }

    #[pyfn(m)]
    fn integer_det<'py>(_py: Python<'py>, basis: PyReadonlyArray2<'py, i64>) -> PyResult<i64> {
        let basis = basis.as_array();
        let res = integer_det_inner(basis);
        res.map_err(PyErr::new::<PyOverflowError, _>)
    }

    Ok(())
}
