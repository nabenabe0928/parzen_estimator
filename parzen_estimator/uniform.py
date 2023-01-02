from typing import Optional, Type, Union

import numpy as np

from parzen_estimator.constants import (
    NumericType,
)
from parzen_estimator.parzen_estimator import AbstractParzenEstimator
from parzen_estimator.utils import validate_and_update_dtype, validate_and_update_q


class NumericalUniform(AbstractParzenEstimator):
    def __init__(
        self,
        lb: NumericType,
        ub: NumericType,
        *,
        q: Optional[NumericType] = None,
        dtype: Type[Union[np.number, int, float]] = np.float64,
    ):
        self._lb = lb
        self._ub = ub
        self._dtype = validate_and_update_dtype(dtype=dtype)
        self._q = validate_and_update_q(dtype=self._dtype, q=q)

    def __repr__(self) -> str:
        ret = f"{self.__class__.__name__}(lb={self.lb}, ub={self.ub}, q={self.q})"
        return ret

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        if self.q is not None:
            val_range = int(self.domain_size / self.q + 0.5)
            samples = self.lb + self.q * rng.randint(val_range, size=n_samples)
        else:
            samples = rng.random(n_samples) * self.domain_size + self.lb

        return samples.astype(self._dtype)

    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        scaled_x = self.lb + x * (self.ub - self.lb)
        scaled_x = scaled_x if self.q is None else np.round((scaled_x - self.lb) / self.q) * self.q + self.lb
        return scaled_x.astype(self._dtype)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.size, 1.0 / self.domain_size)

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} does not have basis.")

    @property
    def domain_size(self) -> NumericType:
        return self.ub - self.lb if self.q is None else int(np.round((self.ub - self.lb) / self.q)) + 1

    @property
    def lb(self) -> NumericType:
        return self._lb

    @property
    def ub(self) -> NumericType:
        return self._ub

    @property
    def q(self) -> Optional[NumericType]:
        return self._q

    @property
    def size(self) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} does not have size.")


class CategoricalUniform(AbstractParzenEstimator):
    def __init__(self, n_choices: int):
        self._n_choices = n_choices
        self._dtype = np.int32

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_choices={self.n_choices})"

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        return rng.randint(self.n_choices, size=n_samples)

    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        scaled_x = x * (self.n_choices - 1)
        return scaled_x.astype(self._dtype)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.size, 1.0 / self.domain_size)

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} does not have basis.")

    @property
    def domain_size(self) -> NumericType:
        return self.n_choices

    @property
    def n_choices(self) -> int:
        return self._n_choices

    @property
    def size(self) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} does not have size.")
