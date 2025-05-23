use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod error;
mod mstl;
mod mstl_impl;
mod mstl_params;
mod mstl_result;
mod stl;
mod stl_impl;
mod stl_params;
mod stl_result;

pub use error::Error;
pub use mstl::{Mstl, MstlParams, MstlResult};
pub use stl::{Stl, StlParams, StlResult};

#[pyclass]
pub struct STL {
    data: Vec<f32>,
    period: usize,
    params: PyStlParams,
}

#[pymethods]
impl STL {
    #[new]
    #[pyo3(signature = (data, *, period, robust=false, seasonal_length=None, trend_length=None))]
    fn new(
        data: Vec<f32>,
        period: usize,
        robust: bool,
        seasonal_length: Option<usize>,
        trend_length: Option<usize>,
    ) -> Self {
        let mut params = PyStlParams::new();

        if robust {
            params.robust(true).unwrap();
        }

        if let Some(sl) = seasonal_length {
            params.seasonal_length(sl).unwrap();
        }

        if let Some(tl) = trend_length {
            params.trend_length(tl).unwrap();
        }

        Self {
            data,
            period,
            params,
        }
    }

    fn fit(&self) -> PyResult<PyStlResult> {
        self.params.fit(self.data.clone(), self.period)
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

#[pyclass]
pub struct PyStlResult {
    inner: StlResult,
}

#[pymethods]
impl PyStlResult {
    #[getter]
    fn seasonal(&self) -> Vec<f32> {
        self.inner.seasonal().to_vec()
    }

    #[getter]
    fn trend(&self) -> Vec<f32> {
        self.inner.trend().to_vec()
    }

    #[getter]
    fn remainder(&self) -> Vec<f32> {
        self.inner.remainder().to_vec()
    }

    #[getter]
    fn weights(&self) -> Vec<f32> {
        self.inner.weights().to_vec()
    }

    fn seasonal_strength(&self) -> f32 {
        self.inner.seasonal_strength()
    }

    fn trend_strength(&self) -> f32 {
        self.inner.trend_strength()
    }

    #[getter]
    fn resid(&self) -> Vec<f32> {
        self.inner.remainder().to_vec()
    }

    #[getter]
    fn seasonal_component(&self) -> Vec<f32> {
        self.inner.seasonal().to_vec()
    }

    #[getter]
    fn trend_component(&self) -> Vec<f32> {
        self.inner.trend().to_vec()
    }
}

#[pyclass]
pub struct PyMstlResult {
    inner: MstlResult,
}

#[pymethods]
impl PyMstlResult {
    #[getter]
    fn seasonal(&self) -> Vec<Vec<f32>> {
        self.inner.seasonal().iter().map(|s| s.to_vec()).collect()
    }

    #[getter]
    fn trend(&self) -> Vec<f32> {
        self.inner.trend().to_vec()
    }

    #[getter]
    fn remainder(&self) -> Vec<f32> {
        self.inner.remainder().to_vec()
    }

    fn seasonal_strength(&self) -> Vec<f32> {
        self.inner.seasonal_strength()
    }

    fn trend_strength(&self) -> f32 {
        self.inner.trend_strength()
    }
}

#[pyclass]
pub struct PyStlParams {
    inner: StlParams,
}

#[pymethods]
impl PyStlParams {
    #[new]
    fn new() -> Self {
        Self {
            inner: StlParams::new(),
        }
    }

    fn seasonal_length(&mut self, length: usize) -> PyResult<()> {
        self.inner.seasonal_length(length);
        Ok(())
    }

    fn trend_length(&mut self, length: usize) -> PyResult<()> {
        self.inner.trend_length(length);
        Ok(())
    }

    fn robust(&mut self, robust: bool) -> PyResult<()> {
        self.inner.robust(robust);
        Ok(())
    }

    fn fit(&self, series: Vec<f32>, period: usize) -> PyResult<PyStlResult> {
        let result = self.inner.fit(&series, period)?;
        Ok(PyStlResult { inner: result })
    }
}

#[pyfunction]
fn stl_decompose(series: Vec<f32>, period: usize) -> PyResult<PyStlResult> {
    let result = Stl::fit(&series, period)?;
    Ok(PyStlResult { inner: result })
}

#[pyfunction]
fn mstl_decompose(series: Vec<f32>, periods: Vec<usize>) -> PyResult<PyMstlResult> {
    let result = Mstl::fit(&series, &periods)?;
    Ok(PyMstlResult { inner: result })
}

#[pymodule]
fn stl_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<STL>()?;
    m.add_class::<PyStlResult>()?;
    m.add_class::<PyMstlResult>()?;
    m.add_class::<PyStlParams>()?;
    m.add_function(wrap_pyfunction!(stl_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(mstl_decompose, m)?)?;
    Ok(())
}
