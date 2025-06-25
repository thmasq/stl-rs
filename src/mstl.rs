use super::Error;

/// Multiple seasonal-trend decomposition using Loess (MSTL).
pub struct Mstl;

impl Mstl {
    /// Decomposes a time series.
    pub fn fit(series: &[f64], periods: &[usize]) -> Result<MstlResult, Error> {
        MstlParams::new().fit(series, periods)
    }

    /// Creates a new set of parameters.
    pub fn params() -> MstlParams {
        MstlParams::new()
    }
}

// Re-export the types so they can be imported from this module
pub use super::mstl_params::MstlParams;
pub use super::mstl_result::MstlResult;

#[cfg(test)]
mod tests {
    use crate::{Error, Mstl, Stl};

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
        let result = Mstl::fit(&generate_series(), &[6, 10]).unwrap();
        assert_elements_in_delta(
            &[
                0.29589650916257176,
                0.7131360245365341,
                -1.9777545147806772,
                2.1624698511020926,
                -2.3451171463413205,
            ],
            &result.seasonal()[0][..5],
        );
        assert_elements_in_delta(
            &[
                1.4353756018021067,
                1.6273497148046578,
                0.06445418873689807,
                -1.8591810659363182,
                -1.7695663181726196,
            ],
            &result.seasonal()[1][..5],
        );
        assert_elements_in_delta(
            &[
                5.119031709402748,
                5.206676631219516,
                5.294321553036284,
                5.376592067002382,
                5.458862580968479,
            ],
            &result.trend()[..5],
        );
        assert_elements_in_delta(
            &[
                -1.8503038203674267,
                1.4528376294392915,
                -1.3810212269925057,
                3.3201191478318437,
                -1.3441791164545398,
            ],
            &result.remainder()[..5],
        );
    }

    #[test]
    fn test_into_parts() {
        let result = Mstl::fit(&generate_series(), &[6, 10]).unwrap();
        let (seasonal, trend, remainder) = result.into_parts();
        assert_elements_in_delta(
            &[
                0.29589650916257176,
                0.7131360245365341,
                -1.9777545147806772,
                2.1624698511020926,
                -2.3451171463413205,
            ],
            &seasonal[0][..5],
        );
        assert_elements_in_delta(
            &[
                1.4353756018021067,
                1.6273497148046578,
                0.06445418873689807,
                -1.8591810659363182,
                -1.7695663181726196,
            ],
            &seasonal[1][..5],
        );
        assert_elements_in_delta(
            &[
                5.119031709402748,
                5.206676631219516,
                5.294321553036284,
                5.376592067002382,
                5.458862580968479,
            ],
            &trend[..5],
        );
        assert_elements_in_delta(
            &[
                -1.8503038203674267,
                1.4528376294392915,
                -1.3810212269925057,
                3.3201191478318437,
                -1.3441791164545398,
            ],
            &remainder[..5],
        );
    }

    #[test]
    fn test_unsorted_periods() {
        let result = Mstl::fit(&generate_series(), &[10, 6]).unwrap();
        assert_elements_in_delta(
            &[
                1.4353756018021067,
                1.6273497148046578,
                0.06445418873689807,
                -1.8591810659363182,
                -1.7695663181726196,
            ],
            &result.seasonal()[0][..5],
        );
        assert_elements_in_delta(
            &[
                0.29589650916257176,
                0.7131360245365341,
                -1.9777545147806772,
                2.1624698511020926,
                -2.3451171463413205,
            ],
            &result.seasonal()[1][..5],
        );
        assert_elements_in_delta(
            &[
                5.119031709402748,
                5.206676631219516,
                5.294321553036284,
                5.376592067002382,
                5.458862580968479,
            ],
            &result.trend()[..5],
        );
        assert_elements_in_delta(
            &[
                -1.8503038203674267,
                1.4528376294392915,
                -1.3810212269925057,
                3.3201191478318437,
                -1.3441791164545398,
            ],
            &result.remainder()[..5],
        );
    }

    #[test]
    fn test_lambda() {
        let result = Mstl::params()
            .lambda(0.5)
            .fit(&generate_series(), &[6, 10])
            .unwrap();
        assert_elements_in_delta(
            &[
                0.44430562369096716,
                0.11285293282005056,
                -0.7162125659784292,
                1.2348450515667595,
                -1.8345949154421768,
            ],
            &result.seasonal()[0][..5],
        );
        assert_elements_in_delta(
            &[
                1.0681318097548325,
                0.8873143999108631,
                0.08834843785509017,
                -1.4177721339186748,
                -1.1964788702205362,
            ],
            &result.seasonal()[1][..5],
        );
        assert_elements_in_delta(
            &[
                2.0540321512407833,
                2.1118216113077026,
                2.169611071374622,
                2.2221608300805222,
                2.274710588786423,
            ],
            &result.trend()[..5],
        );
        assert_elements_in_delta(
            &[
                -1.0943336296870034,
                0.8880110559613836,
                -0.7133198185050929,
                1.9607662522713927,
                -1.24363680312371,
            ],
            &result.remainder()[..5],
        );
    }

    #[test]
    fn test_lambda_zero() {
        let series: Vec<f64> = generate_series().iter().map(|&v| v + 1.0).collect();
        let result = Mstl::params().lambda(0.0).fit(&series, &[6, 10]).unwrap();
        assert_elements_in_delta(
            &[
                0.19159465027753064,
                0.03310720411000314,
                -0.27095605068498,
                0.4771776113104161,
                -0.7357826033253875,
            ],
            &result.seasonal()[0][..5],
        );
        assert_elements_in_delta(
            &[
                0.4372710942583092,
                0.3305976763779958,
                -0.012745197414685421,
                -0.5616181209681718,
                -0.4666170642482943,
            ],
            &result.seasonal()[1][..5],
        );
        assert_elements_in_delta(
            &[
                1.5842742819222766,
                1.6073425369662586,
                1.6304107920102409,
                1.6514868588310692,
                1.6725629256518972,
            ],
            &result.trend()[..5],
        );
        assert_elements_in_delta(
            &[
                -0.4213805572300615,
                0.3315376755397885,
                -0.24809725524246562,
                0.7355387438207321,
                -0.47016325807821535,
            ],
            &result.remainder()[..5],
        );
    }

    #[test]
    fn test_lambda_out_of_range() {
        let result = Mstl::params().lambda(2.0).fit(&generate_series(), &[6, 10]);
        assert_eq!(
            result.unwrap_err(),
            Error::Parameter("lambda must be between 0 and 1".to_string())
        );
    }

    #[test]
    fn test_empty_periods() {
        let periods: Vec<usize> = Vec::new();
        let result = Mstl::fit(&generate_series(), &periods);
        assert_eq!(
            result.unwrap_err(),
            Error::Parameter("periods must not be empty".to_string())
        );
    }

    #[test]
    fn test_period_one() {
        let result = Mstl::fit(&generate_series(), &[1]);
        assert_eq!(
            result.unwrap_err(),
            Error::Parameter("periods must be at least 2".to_string())
        );
    }

    #[test]
    fn test_too_few_periods() {
        let result = Mstl::fit(&generate_series(), &[16]);
        assert_eq!(
            result.unwrap_err(),
            Error::Series("series has less than two periods".to_string())
        );
    }

    #[test]
    fn test_seasonal_strength() {
        let mut stl_params = Stl::params();
        stl_params.seasonal_length(7);
        let result = Mstl::params()
            .stl_params(stl_params)
            .fit(&generate_series(), &[7])
            .unwrap();
        assert_in_delta(0.284111676315015, result.seasonal_strength()[0]);
    }

    #[test]
    fn test_seasonal_strength_max() {
        let series = (0..30).map(|v| (v % 7) as f64).collect::<Vec<f64>>();
        let mut stl_params = Stl::params();
        stl_params.seasonal_length(7);
        let result = Mstl::params()
            .stl_params(stl_params)
            .fit(&series, &[7])
            .unwrap();
        assert_in_delta(1.0, result.seasonal_strength()[0]);
    }

    #[test]
    fn test_trend_strength() {
        let mut stl_params = Stl::params();
        stl_params.seasonal_length(7);
        let result = Mstl::params()
            .stl_params(stl_params)
            .fit(&generate_series(), &[7])
            .unwrap();
        assert_in_delta(0.16384245231864702, result.trend_strength());
    }

    #[test]
    fn test_trend_strength_max() {
        let series = (0..30).map(|v| v as f64).collect::<Vec<f64>>();
        let mut stl_params = Stl::params();
        stl_params.seasonal_length(7);
        let result = Mstl::params()
            .stl_params(stl_params)
            .fit(&series, &[7])
            .unwrap();
        assert_in_delta(1.0, result.trend_strength());
    }
}
