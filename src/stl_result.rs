/// A STL result.
#[derive(Clone, Debug)]
pub struct StlResult {
    pub(crate) seasonal: Vec<f64>,
    pub(crate) trend: Vec<f64>,
    pub(crate) remainder: Vec<f64>,
    pub(crate) weights: Vec<f64>,
}

fn var(series: &[f64]) -> f64 {
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    series.iter().map(|v| (v - mean).powf(2.0)).sum::<f64>() / (series.len() as f64 - 1.0)
}

pub(crate) fn strength(component: &[f64], remainder: &[f64]) -> f64 {
    let sr = component
        .iter()
        .zip(remainder)
        .map(|(a, b)| a + b)
        .collect::<Vec<f64>>();
    (1.0 - var(remainder) / var(&sr)).max(0.0)
}

impl StlResult {
    /// Returns the seasonal component.
    pub fn seasonal(&self) -> &[f64] {
        &self.seasonal
    }

    /// Returns the trend component.
    pub fn trend(&self) -> &[f64] {
        &self.trend
    }

    /// Returns the remainder.
    pub fn remainder(&self) -> &[f64] {
        &self.remainder
    }

    /// Returns the weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Returns the seasonal strength.
    pub fn seasonal_strength(&self) -> f64 {
        strength(self.seasonal(), self.remainder())
    }

    /// Returns the trend strength.
    pub fn trend_strength(&self) -> f64 {
        strength(self.trend(), self.remainder())
    }

    /// Consumes the result, returning the seasonal component, trend component, remainder, and weights.
    pub fn into_parts(self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (self.seasonal, self.trend, self.remainder, self.weights)
    }
}
