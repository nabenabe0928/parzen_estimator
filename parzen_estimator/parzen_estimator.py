from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Type, Union

import ConfigSpace as CS

import numpy as np

from parzen_estimator.constants import (
    CategoricalHPType,
    EPS,
    NULL_VALUE,
    NumericType,
    NumericalHPType,
    SQR2,
    SQR2PI,
    config2type,
    uniform_weight,
)
from parzen_estimator.utils import erf, exp, log


def _get_min_bandwidth_factor(
    config: CS.hyperparameters, is_ordinal: bool, default_min_bandwidth_factor: float
) -> float:

    if config.meta and "min_bandwidth_factor" in config.meta:
        return config.meta["min_bandwidth_factor"]
    if is_ordinal:
        return 1.0 / len(config.sequence)

    dtype = config2type[config.__class__.__name__]
    lb, ub, log, q = config.lower, config.upper, config.log, config.q

    if not log and (q is not None or dtype is int):
        q = q if q is not None else 1
        n_grids = int((ub - lb) / q) + 1
        return 1.0 / n_grids

    return default_min_bandwidth_factor


def calculate_norm_consts(
    lb: NumericType, ub: NumericType, means: np.ndarray, stds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        lb (NumericType):
            The lower bound of a parameter.
        ub (NumericType):
            The upper bound of a parameter.
        means (np.ndarray):
            The mean value for each kernel basis. The shape is (n_samples, ).
        stds (np.ndarray):
            The bandwidth value for each kernel basis. The shape is (n_samples, ).

    Returns:
        norm_consts (np.ndarray):
            The normalization constants of each kernel due to the truncation.
        logpdf_consts (np.ndarray):
            The constants for loglikelihood computation.
    """
    zl = (lb - means) / (SQR2 * stds)
    zu = (ub - means) / (SQR2 * stds)
    norm_consts = 2.0 / (erf(zu) - erf(zl))
    logpdf_consts = log(norm_consts / (SQR2PI * stds))
    return norm_consts, logpdf_consts


class AbstractParzenEstimator(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        """
        Convert the uniform samples [0, 1] into valid range.

        Args:
            x (np.ndarray):
                uniform samples in [0, 1].
                The shape is (n_samples, ).

        Returns:
            converted_x (np.ndarray):
                Converted values.
                The shape is (n_samples, ).
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probability density values for each data point.

        Args:
            x (np.ndarray): The sampled values to compute density values
                            The shape is (n_samples, )

        Returns:
            pdf_vals (np.ndarray):
                The density values given sampled values
                The shape is (n_samples, )
        """
        raise NotImplementedError

    @abstractmethod
    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the kernel value for each basis in the parzen estimator.

        Args:
            x (np.ndarray): The sampled values to compute each kernel value
                            The shape is (n_samples, )

        Returns:
            basis_loglikelihoods (np.ndarray):
                The kernel values for each basis given sampled values
                The shape is (B, n_samples)
                where B is the number of basis and n_samples = xs.size

        NOTE:
            When the parzen estimator is computed by:
                p(x) = sum[i = 0 to B] weights[i] * basis[i](x)
            where basis[i] is the i-th kernel function.
            Then this function returns the following:
                [log(basis[0](xs)), ..., log(basis[B - 1](xs))]
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def domain_size(self) -> NumericType:
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError


class NumericalParzenEstimator(AbstractParzenEstimator):
    def __init__(
        self,
        samples: np.ndarray,
        lb: NumericType,
        ub: NumericType,
        q: Optional[NumericType] = None,
        hard_lb: Optional[NumericType] = None,
        dtype: Type[Union[np.number, int, float]] = np.float64,
        min_bandwidth_factor: float = 1e-2,
    ):

        self._lb, self._ub, self._q = lb, ub, q
        self._hard_lb = hard_lb if hard_lb else lb
        self._size = samples.size + 1
        dtype_choices = (np.int32, np.int64, np.float32, np.float64)
        self._dtype: Type[np.number]
        if dtype is int:
            self._dtype = np.int32
        elif dtype is float:
            self._dtype = np.float64
        elif dtype in dtype_choices:
            self._dtype = dtype  # type: ignore
        else:
            raise ValueError(f"dtype for {self.__class__.__name__} must be {dtype_choices}, but got {dtype}")
        if np.any(samples < lb) or np.any(samples > ub):
            raise ValueError(f"All the samples must be in [{lb}, {ub}].")
        if q is not None:
            cands = np.unique(samples)
            converted_cands = np.round((cands - self._hard_lb) / q) * q + self._hard_lb
            if not np.allclose(cands, converted_cands):
                raise ValueError(
                    "All the samples for q != None must be discritized appropriately."
                    f" Expected each value to be in {converted_cands}, but got {cands}"
                )

        self._calculate(samples=samples, min_bandwidth_factor=min_bandwidth_factor)

    def __repr__(self) -> str:
        ret = f"{self.__class__.__name__}(\n\tlb={self._lb}, ub={self._ub}, q={self._q},\n"
        for i, (m, s) in enumerate(zip(self._means, self._stds)):
            ret += f"\t({i + 1}) weight: {self._weight}, basis: GaussKernel(mean={m}, std={s}),\n"
        return ret + ")"

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        if self._q is None:
            mahalanobis = ((x - self._means[:, np.newaxis]) / self._stds[:, np.newaxis]) ** 2
            return self._logpdf_consts[:, np.newaxis] - 0.5 * mahalanobis
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self._q, self._ub))
            integral_l = self.cdf(np.maximum(x - 0.5 * self._q, self._lb))
            return log(integral_u - integral_l + EPS)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if self._q is None:
            norm_consts = self._norm_consts / (SQR2PI * self._stds)  # noqa: F841
            mahalanobis = ((x[:, np.newaxis] - self._means) / self._stds) ** 2  # noqa: F841
            return self._weight * np.sum(norm_consts * exp(-0.5 * mahalanobis), axis=-1)
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self._q, self._ub))
            integral_l = self.cdf(np.maximum(x - 0.5 * self._q, self._lb))
            return self._weight * np.sum(integral_u - integral_l, axis=0)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the cumulative density function values.

        Args:
            x (np.ndarray): Samples to compute the cdf

        Returns:
            cdf (np.ndarray):
                The cumulative density function value for each sample
                cdf[i] = integral[from -inf to x[i]] pdf(x') dx'
        """
        z = (x - self._means[:, np.newaxis]) / (SQR2 * self._stds[:, np.newaxis])
        norm_consts = self._norm_consts[:, np.newaxis]
        return norm_consts * 0.5 * (1.0 + erf(z))

    def _sample(self, rng: np.random.RandomState, idx: int) -> NumericType:
        while True:
            val = rng.normal(loc=self._means[idx], scale=self._stds[idx])
            if self._lb <= val <= self._ub:
                return val if self._q is None else np.round((val - self._hard_lb) / self._q) * self._q + self._hard_lb

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        weights = np.full(self.size, self._weight)
        samples = [self._sample(rng, active) for active in rng.choice(self.size, p=weights, size=n_samples)]
        return np.array(samples, dtype=self._dtype)

    def sample_by_indices(self, rng: np.random.RandomState, indices: np.ndarray) -> np.ndarray:
        return np.array([self._sample(rng, idx) for idx in indices])

    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        scaled_x = self._lb + x * (self._ub - self._lb)
        scaled_x = (
            scaled_x if self._q is None else np.round((scaled_x - self._hard_lb) / self._q) * self._q + self._hard_lb
        )
        return scaled_x.astype(self._dtype)

    def _calculate(self, samples: np.ndarray, min_bandwidth_factor: float) -> None:
        """
        Calculate parameters of KDE based on Scott's rule

        Args:
            samples (np.ndarray): Samples to use for the construction of
                                  the parzen estimator

        NOTE:
            The bandwidth is computed using the following reference:
                * Scott, D.W. (1992) Multivariate Density Estimation:
                  Theory, Practice, and Visualization.
                * Berwin, A.T. (1993) Bandwidth Selection in Kernel
                  Density Estimation: A Review. (page 12)
                * Nils, B.H, (2013) Bandwidth selection for kernel
                  density estimation: a review of fully automatic selector
                * Wolfgang, H (2005) Nonparametric and Semiparametric Models
        """
        domain_range = self._ub - self._lb
        means = np.append(samples, 0.5 * (self._lb + self._ub))  # Add prior at the end
        self._weight = uniform_weight(means.size)
        std = means.std(ddof=1)

        IQR = np.subtract.reduce(np.percentile(means, [75, 25]))
        bandwidth = 1.059 * min(IQR / 1.34, std) * means.size ** (-0.2)
        # 99% of samples will be confined in mean \pm 0.025 * domain_range (2.5 sigma)
        min_bandwidth = min_bandwidth_factor * domain_range
        clipped_bandwidth = np.ones_like(means) * np.clip(bandwidth, min_bandwidth, 0.5 * domain_range)
        clipped_bandwidth[-1] = domain_range  # The bandwidth for the prior

        self._means, self._stds = means, clipped_bandwidth
        self._norm_consts, self._logpdf_consts = calculate_norm_consts(
            lb=self._lb, ub=self._ub, means=self._means, stds=self._stds
        )

    @property
    def domain_size(self) -> NumericType:
        return self._ub - self._lb

    @property
    def size(self) -> int:
        return self._size


class CategoricalParzenEstimator(AbstractParzenEstimator):
    def __init__(self, samples: np.ndarray, n_choices: int, top: float):

        if samples.dtype not in [np.int32, np.int64]:
            raise ValueError(
                f"samples for {self.__class__.__name__} must be np.ndarray[np.int32/64], " f"but got {samples.dtype}."
            )
        if np.any(samples < 0) or np.any(samples >= n_choices):
            raise ValueError("All the samples must be in [0, n_choices).")

        self._dtype = np.int32
        self._size = samples.size + 1
        self._samples = np.append(samples, NULL_VALUE)
        self._n_choices = n_choices
        # AitchisonAitkenKernel: p = top or (1 - top) / (c - 1)
        # UniformKernel: p = 1 / c
        self._top, self._bottom, self._uniform = top, (1 - top) / (n_choices - 1), 1.0 / n_choices
        self._weight = uniform_weight(samples.size + 1)
        indices, counts = np.unique(samples, return_counts=True)
        self._probs = np.full(n_choices, self._uniform)  # uniform prior, so the initial value is 1 / c.

        slicer = np.arange(n_choices)
        for idx, count in zip(indices, counts):
            self._probs[slicer != idx] += count * self._bottom
            self._probs[slicer == idx] += count * self._top

        self._probs *= self._weight
        likelihood_choices = np.array(
            [[self._top if i == j else self._bottom for j in range(n_choices)] for i in range(n_choices)]
        )

        # shape = (n_basis, n_choices)
        bls = np.maximum(1e-12, np.vstack([likelihood_choices[samples], np.full(n_choices, self._uniform)]))
        self._basis_loglikelihoods = np.log(bls)
        self._cum_basis_likelihoods = np.cumsum(bls, axis=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_choices={self._n_choices}, top={self._top}, probs={self._probs})"

    def uniform_to_valid_range(self, x: np.ndarray) -> np.ndarray:
        scaled_x = x * (self._n_choices - 1)
        return scaled_x.astype(self._dtype)

    def basis_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        return self._basis_loglikelihoods[:, x]

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._probs[x]

    def sample(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        return rng.choice(self._n_choices, p=self._probs, size=n_samples)

    def sample_by_indices(self, rng: np.random.RandomState, indices: np.ndarray) -> np.ndarray:
        n_samples = indices.size
        # equiv to ==> [rng.choice(n_choices, p=basis_likelihoods[idx], size=1)[0] for idx in indices]
        return (self._cum_basis_likelihoods[indices] > rng.random(n_samples)[:, np.newaxis]).argmax(axis=-1)

    @property
    def domain_size(self) -> NumericType:
        return self._n_choices

    @property
    def size(self) -> int:
        return self._size


ParzenEstimatorType = Union[NumericalParzenEstimator, CategoricalParzenEstimator]


def build_numerical_parzen_estimator(
    config: NumericalHPType,
    dtype: Type[Union[float, int]],
    vals: np.ndarray,
    is_ordinal: bool,
    default_min_bandwidth_factor: float = 1e-2,
) -> NumericalParzenEstimator:
    """
    Build a numerical parzen estimator

    Args:
        config (NumericalHPType): Hyperparameter information from the ConfigSpace
        dtype (Type[np.number]): The data type of the hyperparameter
        vals (np.ndarray): The observed hyperparameter values
        is_ordinal (bool): Whether the configuration is ordinal

    Returns:
        pe (NumericalParzenEstimator): Parzen estimator given a set of observations
    """
    min_bandwidth_factor = _get_min_bandwidth_factor(config, is_ordinal, default_min_bandwidth_factor)

    if is_ordinal:
        info = config.meta
        q, log, lb, ub = info.get("q", None), info.get("log", False), info["lower"], info["upper"]
    else:
        q, log, lb, ub = config.q, config.log, config.lower, config.upper

    hard_lb: Optional[NumericType] = None
    if dtype is int or q is not None:
        if log:
            q = None
        elif q is None:
            q = 1
        if q is not None:
            hard_lb = lb
            lb -= 0.5 * q
            ub += 0.5 * q

    if log:
        dtype = float
        lb, ub = np.log(lb), np.log(ub)
        vals = np.log(vals)

    pe = NumericalParzenEstimator(
        samples=vals, lb=lb, ub=ub, q=q, hard_lb=hard_lb, dtype=dtype, min_bandwidth_factor=min_bandwidth_factor
    )

    return pe


def build_categorical_parzen_estimator(
    config: CategoricalHPType, vals: np.ndarray, top: float = 1.0
) -> CategoricalParzenEstimator:
    """
    Build a categorical parzen estimator

    Args:
        config (CategoricalHPType): Hyperparameter information from the ConfigSpace
        vals (np.ndarray): The observed hyperparameter values (i.e. symbols, but not indices)
        top (float): The hyperparameter to define the probability of the category.

    Returns:
        pe (CategoricalParzenEstimator): Parzen estimators given a set of observations
    """
    choices = config.choices
    n_choices = len(choices)

    try:
        choice_indices = np.array([choices.index(val) for val in vals], dtype=np.int32)
    except ValueError:
        raise ValueError(
            "vals to build categorical parzen estimator must be "
            f"the list of symbols {choices}, but got the list of indices."
        )

    pe = CategoricalParzenEstimator(samples=choice_indices, n_choices=n_choices, top=top)

    return pe
