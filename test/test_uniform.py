import pytest
import unittest

import numpy as np

from parzen_estimator.uniform import CategoricalUniform, NumericalUniform


def test_numerical_uniform():
    u = NumericalUniform(lb=-3, ub=2)
    print(u)
    samples = u.uniform_to_valid_range(np.random.random(100))
    assert np.all((u.lb <= samples) & (samples <= u.ub))
    samples = u.sample(rng=np.random.RandomState(), n_samples=100)
    assert np.all((u.lb <= samples) & (samples <= u.ub))
    assert np.allclose(u(samples), 0.2)

    with pytest.raises(NotImplementedError):
        u.basis_loglikelihood(samples)

    with pytest.raises(NotImplementedError):
        u.size


def test_discrete_uniform():
    u = NumericalUniform(lb=-1, ub=3, q=2)
    print(u)
    samples = u.uniform_to_valid_range(np.random.random(100))
    assert np.all((samples == -1) | (samples == 1) | (samples == 3))
    samples = u.sample(rng=np.random.RandomState(), n_samples=100)
    assert np.all((samples == -1) | (samples == 1) | (samples == 3))
    assert np.allclose(u(samples), 1 / 3)

    with pytest.raises(NotImplementedError):
        u.basis_loglikelihood(samples)

    with pytest.raises(NotImplementedError):
        u.size


def test_categorical_uniform():
    u = CategoricalUniform(n_choices=5)
    print(u)
    samples = u.uniform_to_valid_range(np.random.random(100))
    assert np.all((0 <= samples) & (samples <= u.n_choices))
    assert samples.dtype in (np.int32, np.int64)
    samples = u.sample(rng=np.random.RandomState(), n_samples=100)
    assert samples.dtype in (np.int32, np.int64)
    assert np.all((0 <= samples) & (samples < u.n_choices))
    assert np.allclose(u(samples), 0.2)

    with pytest.raises(NotImplementedError):
        u.basis_loglikelihood(samples)

    with pytest.raises(NotImplementedError):
        u.size


if __name__ == "__main__":
    unittest.main()
