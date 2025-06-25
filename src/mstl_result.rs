use super::stl_result::strength;

/// A MSTL result.
#[derive(Clone, Debug)]
pub struct MstlResult {
    pub(crate) seasonal: Vec<Vec<f64>>,
    pub(crate) trend: Vec<f64>,
    pub(crate) remainder: Vec<f64>,
}

impl MstlResult {
    /// Returns the seasonal components.
    pub fn seasonal(&self) -> &[Vec<f64>] {
        &self.seasonal[..]
    }

    /// Returns the trend component.
    pub fn trend(&self) -> &[f64] {
        &self.trend
    }

    /// Returns the remainder.
    pub fn remainder(&self) -> &[f64] {
        &self.remainder
    }

    /// Returns the seasonal strength.
    pub fn seasonal_strength(&self) -> Vec<f64> {
        self.seasonal()
            .iter()
            .map(|s| strength(s, self.remainder()))
            .collect()
    }

    /// Returns the trend strength.
    pub fn trend_strength(&self) -> f64 {
        strength(self.trend(), self.remainder())
    }

    /// Consumes the result, returning the seasonal components, trend component, and remainder.
    pub fn into_parts(self) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        (self.seasonal, self.trend, self.remainder)
    }
}
