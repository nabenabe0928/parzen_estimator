import numpy as np
import pytest
import unittest

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from parzen_estimator.parzen_estimator import (
    CategoricalParzenEstimator,
    NumericalParzenEstimator,
    _get_min_bandwidth_factor,
    build_categorical_parzen_estimator,
    build_numerical_parzen_estimator,
)
from parzen_estimator.constants import NumericalHPType


config2type = {
    "UniformFloatHyperparameter": float,
    "UniformIntegerHyperparameter": int,
    "OrdinalHyperparameter": float,
}


def test_get_min_bandwidth() -> None:
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameters(
        [
            CSH.UniformFloatHyperparameter("x0", 1, 5, log=True, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformFloatHyperparameter("x1", 1, 5, log=True),
            CSH.UniformFloatHyperparameter("x2", 1, 5, log=True, q=0.5, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformFloatHyperparameter("x3", 1, 5, log=True, q=0.5),
            CSH.UniformFloatHyperparameter("x4", 1, 5, q=0.5, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformFloatHyperparameter("x5", 1, 5, q=0.5),
            CSH.UniformFloatHyperparameter("x6", 1, 5, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformFloatHyperparameter("x7", 1, 5),
            CSH.UniformIntegerHyperparameter("x8", 1, 5, log=True, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformIntegerHyperparameter("x9", 1, 5, log=True),
            CSH.UniformIntegerHyperparameter("x10", 1, 5, log=True, q=2, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformIntegerHyperparameter("x11", 1, 5, log=True, q=2),
            CSH.UniformIntegerHyperparameter("x12", 1, 5, q=2, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformIntegerHyperparameter("x13", 1, 5, q=2),
            CSH.UniformIntegerHyperparameter("x14", 1, 5, meta={"min_bandwidth_factor": 0.1}),
            CSH.UniformIntegerHyperparameter("x15", 1, 5),
            CSH.OrdinalHyperparameter("x16", sequence=[1, 2, 3, 4, 5], meta={"lower": 1, "upper": 5}),
            CSH.OrdinalHyperparameter(
                "x17", sequence=[1, 2, 3, 4, 5], meta={"lower": 1, "upper": 5, "min_bandwidth_factor": 0.1}
            ),
        ]
    )

    for d, ans in enumerate(
        [0.1, 0.01, 0.1, 0.01, 0.1, 1 / 9, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 1 / 3, 0.1, 1 / 5, 1 / 5, 0.1]
    ):
        config = config_space.get_hyperparameter(f"x{d}")
        is_ordinal = config.__class__.__name__.startswith("Ordinal")
        assert _get_min_bandwidth_factor(config, is_ordinal, default_min_bandwidth_factor=1e-2) == ans


class TestNumericalParzenEstimator(unittest.TestCase):
    def test_init(self) -> None:
        kwargs_dict = [
            dict(samples=np.array([5]), lb=-3.0, ub=3.0),
            dict(samples=np.array([2]), lb=-3.0, ub=3.0, dtype=None),
        ]
        for kwargs in kwargs_dict:
            with pytest.raises(ValueError):
                NumericalParzenEstimator(**kwargs)

    def test_init_without_prior(self) -> None:
        lb, ub = -50, 50
        samples = np.array([-2, -1, 0, 1, 2])
        pe = NumericalParzenEstimator(samples=samples, lb=lb, ub=ub, q=1, prior=False)
        assert np.allclose(pe._means, samples)
        assert pe._stds.size == samples.size
        assert np.isclose(pe._weight, 1.0 / samples.size)
        pe = NumericalParzenEstimator(samples=samples, lb=lb, ub=ub, q=1, prior=True)
        assert np.allclose(pe._means, list(samples) + [0])
        assert pe._stds.size == samples.size + 1
        assert np.isclose(pe._weight, 1.0 / (samples.size + 1))

    def test_cdf_discrete(self) -> None:
        lb, ub = -50, 50
        samples = np.array([-2, -1, 0, 1, 2])
        pe = NumericalParzenEstimator(samples=np.array([0]), lb=lb, ub=ub, q=1)
        integral_lb, integral_ub = pe.cdf(lb), pe.cdf(ub)
        assert np.allclose(integral_ub - integral_lb, 1.0)
        cdf_vals = pe.cdf(samples) - integral_lb
        ans = [
            [0.02275013, 0.15865525, 0.5, 0.84134475, 0.97724987],
            [0.47916481, 0.48958188, 0.5, 0.51041812, 0.52083519],
        ]
        assert np.allclose(ans, cdf_vals)

    def test_basis_loglikelihood_discrete(self) -> None:
        lb, ub = -50, 50
        samples = np.array([-2, -1, 0, 1, 2])
        pe = NumericalParzenEstimator(samples=np.array([0]), lb=lb, ub=ub, q=1)
        bll_vals = pe.basis_loglikelihood(samples)
        ans = [
            [-2.803501047738798, -1.4199324821566262, -0.9599163336956227, -1.419932482156626, -2.803501047738798],
            [-4.564396550490211, -4.5642465517402, -4.564196552156876, -4.564246551740179, -4.564396550490211],
        ]
        assert np.allclose(ans, bll_vals)

        # Calculate the integral
        bll = pe.basis_loglikelihood(np.linspace(lb, ub, 100000))
        assert 0.99 <= np.exp(bll).mean() * (ub - lb) <= 1.01

    def test_sample(self) -> None:
        rng = np.random.RandomState()
        samples = np.array([0, 1, 2, 3] * 10)
        pe = NumericalParzenEstimator(samples=samples, lb=-3.0, ub=3.0)
        ss = pe.sample(rng, 10)
        assert ss.shape == (10,)

        samples = np.array([-1, 0, 1] * 10)
        pe = NumericalParzenEstimator(samples=samples, lb=-1.25, ub=1.25, q=0.5, hard_lb=-1.0)
        ss = pe.sample(rng, 100)
        choices = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        assert np.allclose(np.unique(ss), choices)

    def test_basis_loglikelihood(self) -> None:
        lb, ub = -3, 3
        samples = np.array([0, 1, 2, 3] * 10)
        pe = NumericalParzenEstimator(samples=samples, lb=lb, ub=ub)
        assert pe.basis_loglikelihood(np.arange(4)).shape == (41, 4)

        samples = np.array(
            [
                0.79069728,
                0.31428171,
                0.57129296,
                0.44143764,
                0.40593304,
                0.4402409,
                0.95061797,
                0.10177345,
                0.46096913,
                0.47954407,
                0.44844498,
                0.40064817,
                0.42593935,
                0.40812766,
                0.45422322,
                0.41891571,
                0.45336276,
                0.10557477,
                0.41424797,
                0.49825133,
                0.44741231,
                0.5678886,
                0.46689275,
                0.48488553,
            ]
        )
        mus = np.array([0.4372727, 0.44549973])
        lb, ub = 0, 1
        pe = NumericalParzenEstimator(samples=mus, lb=lb, ub=ub)
        assert np.allclose([pe._weight], [0.33333333])
        assert np.allclose(pe._means, [0.43727276, 0.44549973, 0.5])
        assert np.allclose(pe._stds, [0.019897270747110334, 0.019897270747110334, 1.0])
        ans = [
            [
                -154.7543487548828,
                -16.10599708557129,
                -19.68598747253418,
                2.9763262271881104,
                1.7578061819076538,
                2.987107515335083,
                -329.816162109375,
                -139.15806579589844,
                2.2890665531158447,
                0.7415248155593872,
                2.8405940532684326,
                1.3041807413101196,
                2.8360159397125244,
                1.9254502058029175,
                2.6353659629821777,
                2.5726494789123535,
                2.671271324157715,
                -135.9549560546875,
                2.3287010192871094,
                -1.6978763341903687,
                2.868389129638672,
                -18.54818344116211,
                1.890196442604065,
                0.13517069816589355,
            ],
            [
                -147.49549865722656,
                -18.74729347229004,
                -16.986461639404297,
                2.9773948192596436,
                1.021071434020996,
                2.9633071422576904,
                -319.23406982421875,
                -146.2153778076172,
                2.696009397506714,
                1.5344642400741577,
                2.987278699874878,
                0.4576236605644226,
                2.5150222778320312,
                1.2343206405639648,
                2.902125358581543,
                2.105701208114624,
                2.9201500415802,
                -142.93328857421875,
                1.7647546529769897,
                -0.5161906480789185,
                2.993614435195923,
                -15.919400215148926,
                2.4202351570129395,
                1.039108395576477,
            ],
            [
                -0.0012746538268402219,
                0.023732159286737442,
                0.03843645751476288,
                0.03926302492618561,
                0.0365535058081150,
                0.03919222578406334,
                -0.06055047735571861,
                -0.03831439092755318,
                0.04021609574556351,
                0.0407685786485672,
                0.03964884206652641,
                0.036042407155036926,
                0.03823531046509743,
                0.036757536232471466,
                0.03993004187941551,
                0.03769046813249588,
                0.03989028558135033,
                -0.036807831376791,
                0.037301093339920044,
                0.04097627103328705,
                0.039595067501068115,
                0.03867337107658386,
                0.04042975604534149,
                0.04086357727646828,
            ],
        ]
        bll = pe.basis_loglikelihood(samples).astype(np.float32)
        err = np.abs(bll - ans) / np.abs(ans)
        assert err.sum() < 1e-4

        # Calculate the integral
        lb, ub = -3, 3
        pe = NumericalParzenEstimator(samples=np.random.random(10) * (ub - lb) + lb, lb=lb, ub=ub)
        bll = pe.basis_loglikelihood(np.linspace(lb, ub, 100000))
        assert 0.99 <= np.exp(bll).mean() * (ub - lb) <= 1.01

    def test_pdf(self) -> None:
        lb, ub = -3, 3
        samples = np.array([0, 1, 2, 3] * 10)
        pe = NumericalParzenEstimator(samples=samples, lb=lb, ub=ub)
        x = np.linspace(-3, 3, 1000)
        bll = pe.basis_loglikelihood(x)
        # Calculate integral
        assert 0.99 < pe.pdf(x).mean() * (ub - lb) < 1.01
        assert np.allclose(np.exp(bll).mean(axis=0), pe.pdf(x))

        lb, ub = -3, 3
        samples = np.array([0, 1, 2, 3] * 10)
        pe = NumericalParzenEstimator(samples=samples, lb=lb, ub=ub, q=1)
        x = np.arange(-3, 4)
        bll = pe.basis_loglikelihood(x)
        # Calculate integral
        assert 0.99 < pe.pdf(x).sum() < 1.01
        assert np.allclose(np.exp(bll).mean(axis=0), pe.pdf(x))

    def test_sample_by_indices(self) -> None:
        lb, ub = -10, 10
        samples = np.arange(-10, 11)
        pe = NumericalParzenEstimator(samples=samples, lb=lb, ub=ub, q=1)
        pe._stds = np.full_like(pe._stds, 1e-12)
        rng = np.random.RandomState()
        indices = np.arange(pe.size)
        assert np.allclose(pe.sample_by_indices(rng, indices), pe._means)

        lb, ub = -1, 1
        samples = np.linspace(-1, 1, 100)
        pe = NumericalParzenEstimator(samples=samples, lb=lb, ub=ub)
        pe._stds = np.full_like(pe._stds, 1e-12)
        rng = np.random.RandomState()
        indices = np.arange(pe.size)
        assert np.allclose(pe.sample_by_indices(rng, indices), pe._means)

    def test_uniform_to_valid_range(self) -> None:
        lb, ub = -3, 3
        samples = np.arange(-3, 4)
        ans = np.arange(-3, 4)
        for q in [1, None]:
            pe = NumericalParzenEstimator(samples=samples, lb=lb, ub=ub, q=1)
            x = np.linspace(0, 1, pe.domain_size + 1)
            assert np.allclose(pe.uniform_to_valid_range(x), ans)


class TestCategoricalParzenEstimator(unittest.TestCase):
    def test_init(self) -> None:
        samples_set = [np.array([5]), np.array([1.0]), np.array(["hoge"])]
        for samples in samples_set:
            with pytest.raises(ValueError):
                CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7)

    def test_init_without_prior(self) -> None:
        samples = np.array([0, 1])
        pe = CategoricalParzenEstimator(samples=samples, n_choices=3, top=0.9, prior=False)
        assert np.allclose(pe._probs, np.array([0.475, 0.475, 0.05]))
        pe = CategoricalParzenEstimator(samples=samples, n_choices=3, top=0.9, prior=True)
        assert np.allclose(pe._probs, np.array([0.42777778, 0.42777778, 0.14444444]))

    def test_sample(self) -> None:
        rng = np.random.RandomState()
        n_samples = 10000

        samples_set = [
            np.array([0]),  # [0.475, 0.175, 0.175, 0.175]
            np.array([0, 3]),  # [0.35, 0.15, 0.15, 0.35]
            np.array([0, 1, 2, 3] * 10),  # [0.25, 0.25, 0.25, 0.25]
        ]
        bounds = [
            [[0.45, 0.50], [0.15, 0.20], [0.15, 0.20], [0.15, 0.20]],
            [[0.33, 0.37], [0.13, 0.17], [0.13, 0.17], [0.33, 0.37]],
            [[0.23, 0.27], [0.23, 0.27], [0.23, 0.27], [0.23, 0.27]],
        ]
        ans_probs = [[0.475, 0.175, 0.175, 0.175], [0.35, 0.15, 0.15, 0.35], [0.25, 0.25, 0.25, 0.25]]
        for samples, bound, ans_prob in zip(samples_set, bounds, ans_probs):
            pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7)
            vals, counts = np.unique(pe.sample(rng, n_samples), return_counts=True)

            assert np.allclose(ans_prob, pe._probs)

            for i in range(4):
                assert bound[i][0] <= counts[vals == i] / n_samples <= bound[i][1]
        else:
            ans = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]] * 10 + [
                [0.25] * 4
            ]
            assert np.allclose(np.log(ans), pe._basis_loglikelihoods)
            ans = [[0.1, 0.1, 0.1, 0.7] * 10 + [0.25], [0.1, 0.7, 0.1, 0.1] * 10 + [0.25]]
            assert np.allclose(np.log(ans).T, pe.basis_loglikelihood([3, 1]))

    def test_basis_loglikelihood(self) -> None:
        samples = np.array([0, 1, 2, 3] * 3)
        pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7)
        bll = pe.basis_loglikelihood(np.arange(4))
        assert bll.shape == (13, 4)

        ans = [
            [-0.3566749393939972, -2.3025851249694824, -2.3025851249694824, -2.3025851249694824],
            [-2.3025851249694824, -0.3566749393939972, -2.3025851249694824, -2.3025851249694824],
            [-2.3025851249694824, -2.3025851249694824, -0.3566749393939972, -2.3025851249694824],
            [-2.3025851249694824, -2.3025851249694824, -2.3025851249694824, -0.3566749393939972],
            [-0.3566749393939972, -2.3025851249694824, -2.3025851249694824, -2.3025851249694824],
            [-2.3025851249694824, -0.3566749393939972, -2.3025851249694824, -2.3025851249694824],
            [-2.3025851249694824, -2.3025851249694824, -0.3566749393939972, -2.3025851249694824],
            [-2.3025851249694824, -2.3025851249694824, -2.3025851249694824, -0.3566749393939972],
            [-0.3566749393939972, -2.3025851249694824, -2.3025851249694824, -2.3025851249694824],
            [-2.3025851249694824, -0.3566749393939972, -2.3025851249694824, -2.3025851249694824],
            [-2.3025851249694824, -2.3025851249694824, -0.3566749393939972, -2.3025851249694824],
            [-2.3025851249694824, -2.3025851249694824, -2.3025851249694824, -0.3566749393939972],
            [-1.3862943649291992, -1.3862943649291992, -1.3862943649291992, -1.3862943649291992],
        ]

        assert np.allclose(ans, bll)

        # Calculate the integral
        bll = pe.basis_loglikelihood(np.arange(4))
        assert 0.99 <= np.exp(bll).mean() * 4 <= 1.01

    def test_uniform_to_valid_range(self) -> None:
        samples = np.array([0, 1, 2, 3] * 3)
        ans = np.arange(4)
        pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7)
        x = np.linspace(0, 1, pe.domain_size)
        assert np.allclose(pe.uniform_to_valid_range(x), ans)

    def test_pdf(self) -> None:
        samples = np.array([0, 1, 2, 3] * 10)
        pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=0.7)
        assert pe.size == 41
        x = np.arange(4)
        # Calculate integral
        assert 0.99 < pe.pdf(x).sum() < 1.01
        assert np.allclose(pe.pdf(x), [0.25] * 4)

        samples = np.array([0])
        top = 0.7
        pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=top)
        x = np.arange(4)
        # Calculate integral
        assert 0.99 < pe.pdf(x).sum() < 1.01
        high = (top + 0.25) / 2
        assert np.allclose(pe.pdf(x), [high] + [(1 - high) / 3] * 3)

    def test_sample_by_indices(self) -> None:
        samples = np.array([0, 1, 2, 3])
        ans = samples.copy()
        pe = CategoricalParzenEstimator(samples=samples, n_choices=4, top=1.0)
        rng = np.random.RandomState()
        indices = np.arange(pe.size - 1)
        assert np.allclose(pe.sample_by_indices(rng, indices), ans)


class TestBuildParzenEstimators(unittest.TestCase):
    def build_npe(self, vals: np.ndarray, config: NumericalHPType) -> NumericalParzenEstimator:
        config_type = config.__class__.__name__
        return build_numerical_parzen_estimator(
            vals=vals,
            config=config,
            dtype=config2type[config_type],
            is_ordinal=(config.__class__.__name__ == "OrdinalHyperparameter"),
        )

    def _check(self, lb_diff: float, ub_diff: float) -> None:
        self.assertAlmostEqual(lb_diff, 0)
        self.assertAlmostEqual(ub_diff, 0)

    def test_build_categorical_parzen_estimator(self) -> None:
        C = CSH.CategoricalHyperparameter("c1", choices=["a", "b", "c"])
        vals = [1]

        with pytest.raises(ValueError):
            build_categorical_parzen_estimator(config=C, vals=vals)

        s_vals = ["a", "b", "c"]
        try:
            build_categorical_parzen_estimator(config=C, vals=s_vals)
        except Exception as e:
            raise ValueError(f"test_build_numerical_parzen_estimator failed with {e}.")

    def test_build_numerical_parzen_estimator(self) -> None:
        lb, ub, q = 1, 100, 0.5
        x = CSH.UniformFloatHyperparameter("x", lower=lb, upper=ub)
        vals = np.arange(1, 20)
        pe = self.build_npe(vals, x)
        assert pe._q is None
        self._check(pe._lb - lb, pe._ub - ub)

        x = CSH.UniformFloatHyperparameter("x", lower=lb, upper=ub, log=True)
        pe = self.build_npe(vals, x)

        assert pe._q is None
        self._check(pe._lb - np.log(lb), pe._ub - np.log(ub))

        x = CSH.UniformFloatHyperparameter("x", lower=lb, upper=ub, q=q)
        pe = self.build_npe(vals, x)
        assert pe._q == q
        self._check(pe._lb - (lb - 0.5 * q), pe._ub - (ub + 0.5 * q))

        x = CSH.UniformFloatHyperparameter("x", lower=lb, upper=ub, q=q, log=True)
        pe = self.build_npe(vals, x)
        assert pe._q is None
        self._check(pe._lb - np.log(lb), pe._ub - np.log(ub))

        x = CSH.UniformIntegerHyperparameter("x", lower=lb, upper=ub)
        pe = self.build_npe(vals, x)
        assert pe._q == 1
        self._check(pe._lb - (lb - 0.5), pe._ub - (ub + 0.5))

        x = CSH.UniformIntegerHyperparameter("x", lower=lb, upper=ub, log=True)
        pe = self.build_npe(vals, x)
        assert pe._q is None
        self._check(pe._lb - np.log(lb), pe._ub - np.log(ub))

        vals = np.arange(2, 20)
        x = CSH.UniformIntegerHyperparameter("x", lower=2, upper=30, q=2)
        with pytest.raises(ValueError):
            pe = self.build_npe(vals, x)


if __name__ == "__main__":
    unittest.main()
