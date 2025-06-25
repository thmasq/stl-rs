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
    seasonal: Option<usize>,
    trend: Option<usize>,
    low_pass: Option<usize>,
    seasonal_deg: i32,
    trend_deg: i32,
    low_pass_deg: i32,
    robust: bool,
    seasonal_jump: Option<usize>,
    trend_jump: Option<usize>,
    low_pass_jump: Option<usize>,
    inner_loops: Option<usize>,
    outer_loops: Option<usize>,
}

#[pymethods]
impl STL {
    #[new]
    #[pyo3(signature = (
        endog, 
        *, 
        period=None, 
        seasonal=None, 
        trend=None, 
        low_pass=None, 
        seasonal_deg=1, 
        trend_deg=1, 
        low_pass_deg=1, 
        robust=false, 
        seasonal_jump=1, 
        trend_jump=1, 
        low_pass_jump=1
    ))]
    fn new(
        endog: Vec<f32>,
        period: Option<usize>,
        seasonal: Option<usize>,
        trend: Option<usize>,
        low_pass: Option<usize>,
        seasonal_deg: i32,
        trend_deg: i32,
        low_pass_deg: i32,
        robust: bool,
        seasonal_jump: Option<usize>,
        trend_jump: Option<usize>,
        low_pass_jump: Option<usize>,
    ) -> PyResult<Self> {
        // If period is not provided, try to infer it or raise an error
        let period = period.ok_or_else(|| {
            PyValueError::new_err("Period must be specified for ndarray input")
        })?;

        // Validate that we have at least 2 complete cycles
        if endog.len() < period * 2 {
            return Err(PyValueError::new_err(format!(
                "endog must have 2 complete cycles requires {} observations. endog only has {} observation(s)",
                period * 2,
                endog.len()
            )));
        }

        Ok(Self {
            data: endog,
            period,
            seasonal,
            trend,
            low_pass,
            seasonal_deg,
            trend_deg,
            low_pass_deg,
            robust,
            seasonal_jump,
            trend_jump,
            low_pass_jump,
            inner_loops: None,
            outer_loops: None,
        })
    }

#[pyo3(signature = (inner_iter=None, outer_iter=None))]
    fn fit(&self, inner_iter: Option<usize>, outer_iter: Option<usize>) -> PyResult<PySTLResult> {
        let mut params = StlParams::new();

        // Set seasonal length (default is period if not specified)
        let seasonal_length = self.seasonal.unwrap_or(self.period);
        params.seasonal_length(seasonal_length);

        // Set trend length with statsmodels default calculation
        if let Some(trend) = self.trend {
            params.trend_length(trend);
        } else {
            // Default trend calculation from statsmodels
            let seasonal_len = if seasonal_length % 2 == 0 { seasonal_length + 1 } else { seasonal_length };
            let trend_len = ((1.5 * self.period as f32) / (1.0 - 1.5 / seasonal_len as f32)).ceil() as usize;
            let trend_len = if trend_len % 2 == 0 { trend_len + 1 } else { trend_len };
            params.trend_length(trend_len.max(3));
        }

        // Set low pass length (default is smallest odd number > period)
        if let Some(low_pass) = self.low_pass {
            params.low_pass_length(low_pass);
        } else {
            let low_pass_len = if self.period % 2 == 0 { self.period + 1 } else { self.period };
            params.low_pass_length(low_pass_len);
        }

        // Set degrees
        params.seasonal_degree(self.seasonal_deg);
        params.trend_degree(self.trend_deg);
        params.low_pass_degree(self.low_pass_deg);

        // Set jumps
        if let Some(sj) = self.seasonal_jump {
            params.seasonal_jump(sj);
        }
        if let Some(tj) = self.trend_jump {
            params.trend_jump(tj);
        }
        if let Some(lj) = self.low_pass_jump {
            params.low_pass_jump(lj);
        }

        // Set robustness
        params.robust(self.robust);

        // Set iterations with statsmodels defaults
        let inner_loops = inner_iter.or(self.inner_loops).unwrap_or(if self.robust { 2 } else { 5 });
        let outer_loops = outer_iter.or(self.outer_loops).unwrap_or(if self.robust { 15 } else { 0 });
        
        params.inner_loops(inner_loops);
        params.outer_loops(outer_loops);

        let result = params.fit(&self.data, self.period)?;
        Ok(PySTLResult { inner: result })
    }

    #[getter]
    fn period(&self) -> usize {
        self.period
    }

    #[getter]
    fn nobs(&self) -> usize {
        self.data.len()
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

#[pyclass]
pub struct PySTLResult {
    inner: StlResult,
}

#[pymethods]
impl PySTLResult {
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
    fn resid(&self) -> Vec<f32> {
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
    fn seasonal_component(&self) -> Vec<f32> {
        self.inner.seasonal().to_vec()
    }

    #[getter]
    fn trend_component(&self) -> Vec<f32> {
        self.inner.trend().to_vec()
    }

    #[getter]
    fn nobs(&self) -> usize {
        self.inner.seasonal().len()
    }
}

// Keep the existing MSTL classes unchanged
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

    fn fit(&self, series: Vec<f32>, period: usize) -> PyResult<PySTLResult> {
        let result = self.inner.fit(&series, period)?;
        Ok(PySTLResult { inner: result })
    }
}

#[pyfunction]
fn stl_decompose(series: Vec<f32>, period: usize) -> PyResult<PySTLResult> {
    let result = Stl::fit(&series, period)?;
    Ok(PySTLResult { inner: result })
}

#[pyfunction]
fn mstl_decompose(series: Vec<f32>, periods: Vec<usize>) -> PyResult<PyMstlResult> {
    let result = Mstl::fit(&series, &periods)?;
    Ok(PyMstlResult { inner: result })
}

#[pymodule]
fn stl_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<STL>()?;
    m.add_class::<PySTLResult>()?;
    m.add_class::<PyMstlResult>()?;
    m.add_class::<PyStlParams>()?;
    m.add_function(wrap_pyfunction!(stl_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(mstl_decompose, m)?)?;
    Ok(())
}

