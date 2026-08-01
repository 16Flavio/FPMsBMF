use boolmat::BitMatrix;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn count_ones(matrix: PyReadonlyArray2<bool>) -> usize {
    let a = matrix.as_array();
    let shapes = matrix.shape();
    let m = shapes[0];
    let n = shapes[1];

    let mut x = BitMatrix::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            if a[[i, j]] {
                x.set(i, j, true);
            }
        }
    }

    x.count_ones()
}

#[pymodule]
fn FPMsBMF(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_ones, m)?)?;
    Ok(())
}
