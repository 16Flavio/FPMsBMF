use crate::matrix::PyBitMatrix;
use bmf_core::{ao_bmf as core_ao_bmf, BoolLs, Method};
use boolmat::BitVec;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_method(name: &str) -> PyResult<Method> {
    name.parse::<Method>()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn methods() -> Vec<String> {
    Method::NAMES.iter().map(|s| s.to_string()).collect()
}

#[pyclass(name = "BoolLs", module = "FPMsBMF")]
pub struct PyBoolLs {
    inner: BoolLs,
    m: usize,
    r: usize,
}

#[pymethods]
impl PyBoolLs {
    #[new]
    fn new(w: &PyBitMatrix) -> PyResult<Self> {
        let (m, r) = (w.inner.rows(), w.inner.cols());
        if r > 26 {
            return Err(PyValueError::new_err(format!(
                "rang {r} trop grand : le tableau interne ferait 2^{r} entrees"
            )));
        }
        Ok(Self {
            inner: BoolLs::new(&w.inner),
            m,
            r,
        })
    }

    #[getter]
    fn m(&self) -> usize {
        self.m
    }

    #[getter]
    fn r(&self) -> usize {
        self.r
    }

    #[pyo3(signature = (x, method = "zeta", seed = 0))]
    fn solve(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<bool>,
        method: &str,
        seed: u64,
    ) -> PyResult<(u64, usize)> {
        let m = parse_method(method)?;
        let view = x.as_array();
        if view.len() != self.m {
            return Err(PyValueError::new_err(format!(
                "x a {} elements, W a {} lignes",
                view.len(),
                self.m
            )));
        }
        let bools: Vec<bool> = view.iter().copied().collect();
        let xv = BitVec::from_bools(&bools);

        Ok(py.allow_threads(|| self.inner.solve_seeded(&xv, m, seed)))
    }

    #[pyo3(signature = (x, method = "zeta", seed = 0))]
    fn solve_all(
        &self,
        py: Python<'_>,
        x: &PyBitMatrix,
        method: &str,
        seed: u64,
    ) -> PyResult<(PyBitMatrix, usize)> {
        let meth = parse_method(method)?;
        if x.inner.rows() != self.m {
            return Err(PyValueError::new_err(format!(
                "X a {} lignes, W en a {}",
                x.inner.rows(),
                self.m
            )));
        }
        let (n, r) = (x.inner.cols(), self.r);

        let (h, err) = py.allow_threads(|| {
            let x_t = x.inner.transpose();
            let mut h_t = boolmat::BitMatrix::zeros(n, r);
            let mut total = 0usize;
            for j in 0..n {
                let col = BitVec::from_words(x_t.row(j), self.m);
                let (h_j, c) = self.inner.solve_seeded(&col, meth, seed);
                h_t.set_row(j, &[h_j]);
                total += c;
            }
            (h_t.transpose(), total)
        });

        Ok((PyBitMatrix::wrap(h), err))
    }

    fn __repr__(&self) -> String {
        format!("<BoolLs m={} r={}>", self.m, self.r)
    }
}

#[pyclass(name = "BmfResult", module = "FPMsBMF", get_all)]
#[derive(Clone)]
pub struct PyBmfResult {
    pub w: PyBitMatrix,
    pub h: PyBitMatrix,
    pub error: usize,
    pub iterations: usize,
}

#[pymethods]
impl PyBmfResult {
    fn reconstruct(&self) -> PyBitMatrix {
        PyBitMatrix::wrap(self.w.inner.product(&self.h.inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "<BmfResult error={} iterations={} rank={}>",
            self.error,
            self.iterations,
            self.w.inner.cols()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (x, r, method = "zeta", max_iter = 50, seed = 0))]
pub fn ao_bmf(
    py: Python<'_>,
    x: &PyBitMatrix,
    r: usize,
    method: &str,
    max_iter: usize,
    seed: u64,
) -> PyResult<PyBmfResult> {
    let meth = parse_method(method)?;

    if r == 0 {
        return Err(PyValueError::new_err("r doit valoir au moins 1"));
    }
    if r > 26 {
        return Err(PyValueError::new_err(format!(
            "rang {r} trop grand : le tableau interne ferait 2^{r} entrees"
        )));
    }
    if max_iter == 0 {
        return Err(PyValueError::new_err("max_iter doit valoir au moins 1"));
    }

    let res = py.allow_threads(|| core_ao_bmf(&x.inner, r, meth, max_iter, seed));

    Ok(PyBmfResult {
        w: PyBitMatrix::wrap(res.w),
        h: PyBitMatrix::wrap(res.h),
        error: res.error,
        iterations: res.iterations,
    })
}

#[pyfunction]
#[pyo3(signature = (w, x, method = "zeta", seed = 0))]
pub fn boolls(
    py: Python<'_>,
    w: &PyBitMatrix,
    x: PyReadonlyArray1<bool>,
    method: &str,
    seed: u64,
) -> PyResult<(u64, usize)> {
    let solver = PyBoolLs::new(w)?;
    solver.solve(py, x, method, seed)
}
