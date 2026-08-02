use boolmat::BitMatrix;
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[pyclass(name = "BitMatrix", module = "FPMsBMF")]
#[derive(Clone)]
pub struct PyBitMatrix {
    pub(crate) inner: BitMatrix,
}

impl PyBitMatrix {
    pub(crate) fn wrap(inner: BitMatrix) -> Self {
        Self { inner }
    }

    fn check_dims(rows: usize, cols: usize) -> PyResult<()> {
        if rows == 0 || cols == 0 {
            return Err(PyValueError::new_err(format!(
                "dimensions invalides {rows}x{cols} : rows et cols doivent valoir au moins 1"
            )));
        }
        Ok(())
    }

    fn check_same_shape(&self, other: &Self, op: &str) -> PyResult<()> {
        if self.inner.rows() != other.inner.rows() || self.inner.cols() != other.inner.cols() {
            return Err(PyValueError::new_err(format!(
                "{op} : dimensions incompatibles {}x{} et {}x{}",
                self.inner.rows(),
                self.inner.cols(),
                other.inner.rows(),
                other.inner.cols()
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyBitMatrix {
    #[new]
    #[pyo3(signature = (rows, cols, fill = false))]
    fn new(rows: usize, cols: usize, fill: bool) -> PyResult<Self> {
        Self::check_dims(rows, cols)?;
        let inner = if fill {
            BitMatrix::ones(rows, cols)
        } else {
            BitMatrix::zeros(rows, cols)
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> PyResult<Self> {
        Self::check_dims(rows, cols)?;
        Ok(Self {
            inner: BitMatrix::zeros(rows, cols),
        })
    }

    #[staticmethod]
    fn ones(rows: usize, cols: usize) -> PyResult<Self> {
        Self::check_dims(rows, cols)?;
        Ok(Self {
            inner: BitMatrix::ones(rows, cols),
        })
    }

    #[staticmethod]
    fn from_list(data: Vec<Vec<i64>>) -> PyResult<Self> {
        if data.is_empty() {
            return Err(PyValueError::new_err("from_list : aucune ligne fournie"));
        }

        let rows = data.len();
        let cols = data[0].len();
        if cols == 0 {
            return Err(PyValueError::new_err("from_list : les lignes sont vides"));
        }

        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(PyValueError::new_err(format!(
                    "from_list : la ligne {i} a {} elements, la ligne 0 en a {cols}",
                    row.len()
                )));
            }
            for (j, &v) in row.iter().enumerate() {
                if v != 0 && v != 1 {
                    return Err(PyValueError::new_err(format!(
                        "from_list : valeur {v} en ({i}, {j}) ; seuls 0 et 1 sont admis"
                    )));
                }
            }
        }

        let mut inner = BitMatrix::zeros(rows, cols);
        for (i, row) in data.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v == 1 {
                    inner.set(i, j, true);
                }
            }
        }
        Ok(Self { inner })
    }

    fn to_list(&self) -> Vec<Vec<i64>> {
        (0..self.inner.rows())
            .map(|i| {
                (0..self.inner.cols())
                    .map(|j| self.inner.get(i, j) as i64)
                    .collect()
            })
            .collect()
    }

    #[staticmethod]
    fn from_numpy(a: PyReadonlyArray2<bool>) -> PyResult<Self> {
        let view = a.as_array();
        let (rows, cols) = view.dim();
        Self::check_dims(rows, cols)?;

        let mut inner = BitMatrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                if view[[i, j]] {
                    inner.set(i, j, true);
                }
            }
        }
        Ok(Self { inner })
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        let (rows, cols) = (self.inner.rows(), self.inner.cols());
        let mut out = Array2::from_elem((rows, cols), false);
        for i in 0..rows {
            for j in 0..cols {
                out[[i, j]] = self.inner.get(i, j);
            }
        }
        PyArray2::from_owned_array(py, out)
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.rows(), self.inner.cols())
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols()
    }

    fn count_ones(&self) -> usize {
        self.inner.count_ones()
    }

    fn hamming(&self, other: &Self) -> PyResult<usize> {
        self.check_same_shape(other, "hamming")?;
        Ok(self.inner.hamming(&other.inner))
    }

    fn count_andnot(&self, other: &Self) -> PyResult<usize> {
        self.check_same_shape(other, "count_andnot")?;
        Ok(self.inner.count_andnot(&other.inner))
    }

    fn transpose(&self) -> Self {
        Self {
            inner: self.inner.transpose(),
        }
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn row_count_ones(&self, i: usize) -> PyResult<usize> {
        if i >= self.inner.rows() {
            return Err(PyIndexError::new_err(format!(
                "ligne {i} hors limites (rows = {})",
                self.inner.rows()
            )));
        }
        Ok(self.inner.row_count_ones(i))
    }

    fn __getitem__(&self, idx: (usize, usize)) -> PyResult<bool> {
        let (i, j) = idx;
        if i >= self.inner.rows() || j >= self.inner.cols() {
            return Err(PyIndexError::new_err(format!(
                "indice ({i}, {j}) hors limites pour une matrice {}x{}",
                self.inner.rows(),
                self.inner.cols()
            )));
        }
        Ok(self.inner.get(i, j))
    }

    fn __setitem__(&mut self, idx: (usize, usize), value: bool) -> PyResult<()> {
        let (i, j) = idx;
        if i >= self.inner.rows() || j >= self.inner.cols() {
            return Err(PyIndexError::new_err(format!(
                "indice ({i}, {j}) hors limites pour une matrice {}x{}",
                self.inner.rows(),
                self.inner.cols()
            )));
        }
        self.inner.set(i, j, value);
        Ok(())
    }

    fn __or__(&self, other: &Self) -> PyResult<Self> {
        self.check_same_shape(other, "|")?;
        let mut out = self.inner.clone();
        out.or(&other.inner);
        Ok(Self { inner: out })
    }

    fn __ior__(&mut self, other: &Self) -> PyResult<()> {
        self.check_same_shape(other, "|=")?;
        self.inner.or(&other.inner);
        Ok(())
    }

    fn __and__(&self, other: &Self) -> PyResult<Self> {
        self.check_same_shape(other, "&")?;
        let mut out = self.inner.clone();
        out.and(&other.inner);
        Ok(Self { inner: out })
    }

    fn __iand__(&mut self, other: &Self) -> PyResult<()> {
        self.check_same_shape(other, "&=")?;
        self.inner.and(&other.inner);
        Ok(())
    }

    fn __xor__(&self, other: &Self) -> PyResult<Self> {
        self.check_same_shape(other, "^")?;
        let mut out = self.inner.clone();
        out.xor(&other.inner);
        Ok(Self { inner: out })
    }

    fn __ixor__(&mut self, other: &Self) -> PyResult<()> {
        self.check_same_shape(other, "^=")?;
        self.inner.xor(&other.inner);
        Ok(())
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.check_same_shape(other, "-")?;
        let mut out = self.inner.clone();
        out.andnot(&other.inner);
        Ok(Self { inner: out })
    }

    fn __invert__(&self) -> Self {
        let mut out = self.inner.clone();
        out.invert();
        Self { inner: out }
    }

    fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        if self.inner.cols() != other.inner.rows() {
            return Err(PyValueError::new_err(format!(
                "@ : dimensions incompatibles {}x{} et {}x{}",
                self.inner.rows(),
                self.inner.cols(),
                other.inner.rows(),
                other.inner.cols()
            )));
        }
        Ok(Self {
            inner: self.inner.product(&other.inner),
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __len__(&self) -> usize {
        self.inner.rows()
    }

    fn __repr__(&self) -> String {
        format!(
            "<BitMatrix {}x{}, {} bits a 1>",
            self.inner.rows(),
            self.inner.cols(),
            self.inner.count_ones()
        )
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}
