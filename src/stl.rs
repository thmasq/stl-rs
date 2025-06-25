use super::Error;

/// Seasonal-trend decomposition using Loess (STL).
pub struct Stl;

impl Stl {
    /// Decomposes a time series.
    pub fn fit(series: &[f64], period: usize) -> Result<StlResult, Error> {
        StlParams::new().fit(series, period)
    }

    /// Creates a new set of parameters.
    pub fn params() -> StlParams {
        StlParams::new()
    }
}

// Re-export the types so they can be imported from this module
pub use super::stl_params::StlParams;
pub use super::stl_result::StlResult;

#[cfg(test)]
mod tests {
    use crate::{Error, Stl};

    fn assert_in_delta(exp: f64, act: f64) {
        assert!((exp - act).abs() < 0.001);
    }

    fn assert_elements_in_delta(exp: &[f64], act: &[f64]) {
        assert_eq!(exp.len(), act.len());
        for i in 0..exp.len() {
            assert_in_delta(exp[i], act[i]);
        }
    }

    fn generate_series() -> Vec<f64> {
        return vec![
            5.0, 9.0, 2.0, 9.0, 0.0, 6.0, 3.0, 8.0, 5.0, 8.0, 7.0, 8.0, 8.0, 0.0, 2.0, 5.0, 0.0,
            5.0, 6.0, 7.0, 3.0, 6.0, 1.0, 4.0, 4.0, 4.0, 3.0, 7.0, 5.0, 8.0,
        ];
    }

    #[test]
    fn test_works() {
        let result = Stl::fit(&generate_series(), 7).unwrap();
        assert_elements_in_delta(
            &[
                0.3721964367072997,
                0.7589122519555539,
                -1.3278225684162583,
                1.9570959767355505,
                -0.6061163711818103,
            ],
            &result.seasonal()[..5],
        );
        assert_elements_in_delta(
            &[
                4.8020778392149674,
                4.908002363781568,
                5.013926888348168,
                5.159272372540581,
                5.304617856732993,
            ],
            &result.trend()[..5],
        );
        assert_elements_in_delta(
            &[
                -0.17427427592226685,
                3.333085384262878,
                -1.6861043199319097,
                1.8836316507238688,
                -4.6985014855511835,
            ],
            &result.remainder()[..5],
        );
        assert_elements_in_delta(&[1.0, 1.0, 1.0, 1.0, 1.0], &result.weights()[..5]);
    }

    #[test]
    fn test_robust() {
        let result = Stl::params()
            .robust(true)
            .fit(&generate_series(), 7)
            .unwrap();
        assert_elements_in_delta(
            &[
                0.14695846927381206,
                0.47820120771618563,
                -1.8359143149360286,
                1.7401479245709783,
                0.8278063458596339,
            ],
            &result.seasonal()[..5],
        );
        assert_elements_in_delta(
            &[
                5.401366655998738,
                5.478292854139745,
                5.555219052280752,
                5.653092900405133,
                5.750966748529512,
            ],
            &result.trend()[..5],
        );
        assert_elements_in_delta(
            &[
                -0.5483251252725507,
                3.043505938144069,
                -1.7193047373447237,
                1.606759175023889,
                -6.578773094389146,
            ],
            &result.remainder()[..5],
        );
        assert_elements_in_delta(
            &[
                0.9937021068501506,
                0.8130745469831938,
                0.9384777923598295,
                0.9459028606712937,
                0.29526025353680324,
            ],
            &result.weights()[..5],
        );
    }

    #[test]
    fn test_into_parts() {
        let result = Stl::fit(&generate_series(), 7).unwrap();
        let (seasonal, trend, remainder, weights) = result.into_parts();
        assert_elements_in_delta(
            &[
                0.3721964367072997,
                0.7589122519555539,
                -1.3278225684162583,
                1.9570959767355505,
                -0.6061163711818103,
            ],
            &seasonal[..5],
        );
        assert_elements_in_delta(
            &[
                4.8020778392149674,
                4.908002363781568,
                5.013926888348168,
                5.159272372540581,
                5.304617856732993,
            ],
            &trend[..5],
        );
        assert_elements_in_delta(
            &[
                -0.17427427592226685,
                3.333085384262878,
                -1.6861043199319097,
                1.8836316507238688,
                -4.6985014855511835,
            ],
            &remainder[..5],
        );
        assert_elements_in_delta(&[1.0, 1.0, 1.0, 1.0, 1.0], &weights[..5]);
    }

    #[test]
    fn test_too_few_periods() {
        let result = Stl::params().fit(&generate_series(), 16);
        assert_eq!(
            result.unwrap_err(),
            Error::Series("series has less than two periods".to_string())
        );
    }

    #[test]
    fn test_bad_seasonal_degree() {
        let result = Stl::params().seasonal_degree(2).fit(&generate_series(), 7);
        assert_eq!(
            result.unwrap_err(),
            Error::Parameter("seasonal_degree must be 0 or 1".to_string())
        );
    }

    #[test]
    fn test_seasonal_strength() {
        let result = Stl::fit(&generate_series(), 7).unwrap();
        assert_in_delta(0.284111676315015, result.seasonal_strength());
    }

    #[test]
    fn test_seasonal_strength_max() {
        let series = (0..30).map(|v| (v % 7) as f64).collect::<Vec<f64>>();
        let result = Stl::fit(&series, 7).unwrap();
        assert_in_delta(1.0, result.seasonal_strength());
    }

    #[test]
    fn test_trend_strength() {
        let result = Stl::fit(&generate_series(), 7).unwrap();
        assert_in_delta(0.16384245231864702, result.trend_strength());
    }

    #[test]
    fn test_trend_strength_max() {
        let series = (0..30).map(|v| v as f64).collect::<Vec<f64>>();
        let result = Stl::fit(&series, 7).unwrap();
        assert_in_delta(1.0, result.trend_strength());
    }
}
