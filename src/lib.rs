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
    data: Vec<f64>,
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
        seasonal=7,
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
        endog: Vec<f64>,
        period: Option<usize>,
        seasonal: usize,
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
            seasonal: Some(seasonal),
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

    /// Fit with GIL release for better multi-threading performance
    #[pyo3(signature = (inner_iter=None, outer_iter=None))]
    fn fit(&self, py: Python, inner_iter: Option<usize>, outer_iter: Option<usize>) -> PyResult<PySTLResult> {
        // Clone data needed for computation
        let data = self.data.clone();
        let period = self.period;
        let seasonal_length = self.seasonal.unwrap();
        let trend = self.trend;
        let low_pass = self.low_pass;
        let seasonal_deg = self.seasonal_deg;
        let trend_deg = self.trend_deg;
        let low_pass_deg = self.low_pass_deg;
        let robust = self.robust;
        let seasonal_jump = self.seasonal_jump;
        let trend_jump = self.trend_jump;
        let low_pass_jump = self.low_pass_jump;
        let inner_loops = self.inner_loops;
        let outer_loops = self.outer_loops;

        // Release GIL during computation
        let result = py.allow_threads(|| {
            let mut params = StlParams::new();

            // Set seasonal length (use provided value, not period default)
            params.seasonal_length(seasonal_length);

            // Set trend length with statsmodels default calculation
            if let Some(trend) = trend {
                params.trend_length(trend);
            } else {
                let seasonal_len = if seasonal_length % 2 == 0 { 
                    seasonal_length + 1 
                } else { 
                    seasonal_length 
                };
                
                let trend_len = ((1.5 * period as f64) / (1.0 - 1.5 / seasonal_len as f64)).ceil() as usize;
                let trend_len = if trend_len % 2 == 0 { trend_len + 1 } else { trend_len };
                params.trend_length(trend_len.max(3));
            }

            // Set low pass length (default is smallest odd number >= period)
            if let Some(low_pass) = low_pass {
                params.low_pass_length(low_pass);
            } else {
                let low_pass_len = if period % 2 == 0 { period + 1 } else { period };
                params.low_pass_length(low_pass_len);
            }

            // Set degrees
            params.seasonal_degree(seasonal_deg);
            params.trend_degree(trend_deg);
            params.low_pass_degree(low_pass_deg);

            params.seasonal_jump(seasonal_jump.unwrap_or(1));
            params.trend_jump(trend_jump.unwrap_or(1));
            params.low_pass_jump(low_pass_jump.unwrap_or(1));

            // Set robustness
            params.robust(robust);

            let inner_loops_val = inner_iter.or(inner_loops).unwrap_or(
                if robust { 2 } else { 5 }
            );
            let outer_loops_val = outer_iter.or(outer_loops).unwrap_or(
                if robust { 15 } else { 0 }
            );
            
            params.inner_loops(inner_loops_val);
            params.outer_loops(outer_loops_val);

            params.fit(&data, period)
        })?;

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

    #[getter]
    fn seasonal(&self) -> usize {
        self.seasonal.unwrap_or(7)
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
    fn seasonal(&self) -> Vec<f64> {
        self.inner.seasonal().to_vec()
    }

    #[getter]
    fn trend(&self) -> Vec<f64> {
        self.inner.trend().to_vec()
    }

    #[getter]
    fn remainder(&self) -> Vec<f64> {
        self.inner.remainder().to_vec()
    }

    #[getter]
    fn resid(&self) -> Vec<f64> {
        self.inner.remainder().to_vec()
    }

    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights().to_vec()
    }

    fn seasonal_strength(&self) -> f64 {
        self.inner.seasonal_strength()
    }

    fn trend_strength(&self) -> f64 {
        self.inner.trend_strength()
    }

    #[getter]
    fn seasonal_component(&self) -> Vec<f64> {
        self.inner.seasonal().to_vec()
    }

    #[getter]
    fn trend_component(&self) -> Vec<f64> {
        self.inner.trend().to_vec()
    }

    #[getter]
    fn nobs(&self) -> usize {
        self.inner.seasonal().len()
    }
}

#[pyclass]
pub struct PyMstlResult {
    inner: MstlResult,
}

#[pymethods]
impl PyMstlResult {
    #[getter]
    fn seasonal(&self) -> Vec<Vec<f64>> {
        self.inner.seasonal().iter().map(|s| s.to_vec()).collect()
    }

    #[getter]
    fn trend(&self) -> Vec<f64> {
        self.inner.trend().to_vec()
    }

    #[getter]
    fn remainder(&self) -> Vec<f64> {
        self.inner.remainder().to_vec()
    }

    fn seasonal_strength(&self) -> Vec<f64> {
        self.inner.seasonal_strength()
    }

    fn trend_strength(&self) -> f64 {
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

    fn fit(&self, py: Python, series: Vec<f64>, period: usize) -> PyResult<PySTLResult> {
        let result = py.allow_threads(|| {
            self.inner.fit(&series, period)
        })?;
        Ok(PySTLResult { inner: result })
    }
}

/// Convenience function for STL decomposition with GIL release
#[pyfunction]
fn stl_decompose(py: Python, series: Vec<f64>, period: usize) -> PyResult<PySTLResult> {
    let result = py.allow_threads(|| {
        Stl::fit(&series, period)
    })?;
    Ok(PySTLResult { inner: result })
}

/// Convenience function for MSTL decomposition with GIL release
#[pyfunction]
fn mstl_decompose(py: Python, series: Vec<f64>, periods: Vec<usize>) -> PyResult<PyMstlResult> {
    let result = py.allow_threads(|| {
        Mstl::fit(&series, &periods)
    })?;
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

    m.add("__version__", "0.1.4")?;
    m.add("__author__", "Thomas Q")?;
    Ok(())
}
