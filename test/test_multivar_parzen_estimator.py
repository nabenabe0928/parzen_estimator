import pytest
import unittest
from typing import Dict, Union

import numpy as np

from parzen_estimator import CategoricalParzenEstimator, MultiVariateParzenEstimator, NumericalParzenEstimator
from parzen_estimator.constants import NULL_VALUE


def default_multivar_pe(top: float = 0.8) -> MultiVariateParzenEstimator:
    lb, ub, n_choices, size = -3, 3, 4, 10
    samples = np.random.random(size=size) * (ub - lb) + lb
    choices = np.random.randint(n_choices, size=size)
    pes: Dict[str, Union[NumericalParzenEstimator, CategoricalParzenEstimator]] = {
        "n1": NumericalParzenEstimator(samples, lb, ub),
        "n2": NumericalParzenEstimator(samples.astype(np.int32), lb, ub, q=1),
        "c1": CategoricalParzenEstimator(choices, n_choices, top=top),
    }
    mvpe = MultiVariateParzenEstimator(pes)
    return mvpe


class TestMultiVariateParzenEstimator(unittest.TestCase):
    def test_init(self) -> None:
        mvpe = default_multivar_pe()
        assert mvpe.dim == 3
        assert mvpe.size == 10 + 1

    def test_error_in_init(self) -> None:
        lb, ub, n_choices, size = -3, 3, 4, 10
        samples = np.random.random(size=size + 1) * (ub - lb) + lb
        choices = np.random.randint(n_choices, size=size)
        pes: Dict[str, Union[NumericalParzenEstimator, CategoricalParzenEstimator]] = {
            "n1": NumericalParzenEstimator(samples, lb, ub),
            "n2": NumericalParzenEstimator(samples.astype(np.int32), lb, ub, q=1),
            "c1": CategoricalParzenEstimator(choices, n_choices),
        }
        with pytest.raises(ValueError):
            MultiVariateParzenEstimator(pes)

    def test_dimension_wise_pdf(self) -> None:
        mvpe = default_multivar_pe()
        X = []
        assert isinstance(mvpe._parzen_estimators["n1"], NumericalParzenEstimator)
        assert isinstance(mvpe._parzen_estimators["n2"], NumericalParzenEstimator)
        assert isinstance(mvpe._parzen_estimators["c1"], CategoricalParzenEstimator)
        lb = mvpe._parzen_estimators["n1"]._lb
        ub = mvpe._parzen_estimators["n1"]._ub
        X.append(np.linspace(lb, ub, 100))
        lb = mvpe._parzen_estimators["n2"]._lb
        ub = mvpe._parzen_estimators["n2"]._ub
        X.append(np.linspace(lb, ub, 100).astype(np.int32)[::-1])
        n_choices = mvpe._parzen_estimators["c1"]._n_choices
        X.append(np.random.randint(n_choices, size=100))

        for pe, pdf_val, x in zip(mvpe._parzen_estimators.values(), mvpe.dimension_wise_pdf(X), X):
            assert np.allclose(pe.pdf(x), pdf_val)

    def test_pdf_and_log_pdf(self) -> None:
        mvpe = default_multivar_pe()
        size = 100
        X = []
        assert isinstance(mvpe._parzen_estimators["n1"], NumericalParzenEstimator)
        assert isinstance(mvpe._parzen_estimators["n2"], NumericalParzenEstimator)
        assert isinstance(mvpe._parzen_estimators["c1"], CategoricalParzenEstimator)
        lb = mvpe._parzen_estimators["n1"]._lb
        ub = mvpe._parzen_estimators["n1"]._ub
        X.append(np.linspace(lb, ub, size))
        lb = mvpe._parzen_estimators["n2"]._lb
        ub = mvpe._parzen_estimators["n2"]._ub
        X.append(np.linspace(lb, ub, size).astype(np.int32)[::-1])
        n_choices = mvpe._parzen_estimators["c1"]._n_choices
        X.append(np.random.randint(n_choices, size=size))

        pdf_vals = mvpe.pdf(X)
        log_pdf_vals = mvpe.log_pdf(X)
        assert pdf_vals.size == size
        assert np.allclose(pdf_vals, np.exp(log_pdf_vals))

    def test_pdf_by_integral(self) -> None:
        lb, ub, n_choices, size = -3, 3, 4, 10
        samples = np.random.random(size=size) * (ub - lb) + lb
        choices = np.random.randint(n_choices, size=size)
        pes: Dict[str, Union[NumericalParzenEstimator, CategoricalParzenEstimator]] = {
            "n": NumericalParzenEstimator(samples.astype(np.int32), lb, ub, q=1),
            "c": CategoricalParzenEstimator(choices, n_choices, top=0.7),
        }
        mvpe = MultiVariateParzenEstimator(pes)

        n_configs = (ub - lb + 1) * n_choices
        X = [np.zeros(n_configs), np.zeros(n_configs, dtype=np.int32)]
        index = 0
        for v in range(lb, ub + 1):
            for c in range(n_choices):
                X[0][index] = v
                X[1][index] = c
                index += 1

        pdf_val = mvpe.pdf(X)
        integral = pdf_val.sum()
        self.assertAlmostEqual(integral, 1.0)

    def test_sample_by_independent(self) -> None:
        mvpe = default_multivar_pe(top=1.0)
        size = mvpe.size
        assert isinstance(mvpe._parzen_estimators["n1"], NumericalParzenEstimator)
        assert isinstance(mvpe._parzen_estimators["n2"], NumericalParzenEstimator)
        assert isinstance(mvpe._parzen_estimators["c1"], CategoricalParzenEstimator)
        mvpe._parzen_estimators["n1"]._stds[:] = 1e-12
        mvpe._parzen_estimators["n2"]._stds[:] = 1e-12
        configs = [
            [
                getattr(pe, "_means")[i] if hasattr(pe, "_means") else getattr(pe, "_samples")[i]
                for pe in mvpe._parzen_estimators.values()
            ]
            for i in range(size)
        ]

        n_samples = 30
        at_least_one_is_independent = False
        samples = mvpe.sample(n_samples=n_samples, rng=np.random.RandomState(), dim_independent=True)
        for i in range(n_samples):
            sampled_config = [samples[d][i] for d in range(3)]
            dependent = False
            for config in configs:
                if np.allclose(sampled_config, config):
                    dependent = True
                    break
                elif config[-1] == NULL_VALUE and np.allclose(sampled_config[:2], config[:2]):
                    dependent = True
                    break
            print(not dependent)
            at_least_one_is_independent |= not dependent

        assert at_least_one_is_independent

    def test_sample(self) -> None:
        mvpe = default_multivar_pe(top=1.0)
        size = mvpe.size
        assert isinstance(mvpe._parzen_estimators["n1"], NumericalParzenEstimator)
        assert isinstance(mvpe._parzen_estimators["n2"], NumericalParzenEstimator)
        assert isinstance(mvpe._parzen_estimators["c1"], CategoricalParzenEstimator)
        mvpe._parzen_estimators["n1"]._stds[:] = 1e-12
        mvpe._parzen_estimators["n2"]._stds[:] = 1e-12
        configs = [
            [
                getattr(pe, "_means")[i] if hasattr(pe, "_means") else getattr(pe, "_samples")[i]
                for pe in mvpe._parzen_estimators.values()
            ]
            for i in range(size)
        ]

        n_samples = 30
        samples = mvpe.sample(n_samples=n_samples, rng=np.random.RandomState())
        for i in range(n_samples):
            sampled_config = [samples[d][i] for d in range(3)]
            ok = False
            for config in configs:
                if np.allclose(sampled_config, config):
                    ok = True
                    break
                elif config[-1] == NULL_VALUE and np.allclose(sampled_config[:2], config[:2]):
                    ok = True
                    break

            assert ok

    def test_uniform_sample(self) -> None:
        mvpe = default_multivar_pe()
        rng = np.random.RandomState()
        size = 1 << 8
        samples = mvpe.uniform_sample(n_samples=size, rng=rng)
        assert len(samples) == mvpe.dim
        assert all(s.size == size for s in samples)

        assert isinstance(mvpe._parzen_estimators["n1"], NumericalParzenEstimator)
        assert isinstance(mvpe._parzen_estimators["n2"], NumericalParzenEstimator)
        assert isinstance(mvpe._parzen_estimators["c1"], CategoricalParzenEstimator)
        lb = mvpe._parzen_estimators["n1"]._lb
        ub = mvpe._parzen_estimators["n1"]._ub
        assert np.all(lb <= samples[0]) and np.all(samples[0] <= ub)
        lb = mvpe._parzen_estimators["n2"]._lb
        ub = mvpe._parzen_estimators["n2"]._ub
        assert all(s in np.arange(-3, 4) for s in samples[1])
        n_choices = mvpe._parzen_estimators["c1"]._n_choices
        assert all(s in np.arange(n_choices) for s in samples[2])

    def test_error_in_uniform_sample(self) -> None:
        with pytest.raises(ValueError):
            mvpe = default_multivar_pe()
            rng = np.random.RandomState()
            mvpe.uniform_sample(n_samples=100, rng=rng, sampling_method="dummy")


if __name__ == "__main__":
    unittest.main()
