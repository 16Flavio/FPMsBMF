mod matrix;
mod solvers;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "FPMsBMF")]
fn fpms_bmf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<matrix::PyBitMatrix>()?;
    m.add_class::<solvers::PyBoolLs>()?;
    m.add_class::<solvers::PyBmfResult>()?;
    m.add_function(wrap_pyfunction!(solvers::ao_bmf, m)?)?;
    m.add_function(wrap_pyfunction!(solvers::boolls, m)?)?;
    m.add_function(wrap_pyfunction!(solvers::methods, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
