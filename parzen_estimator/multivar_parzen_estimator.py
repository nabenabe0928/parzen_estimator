from typing import Dict, List, Union

import numpy as np

from parzen_estimator import CategoricalParzenEstimator, NumericalParzenEstimator
from parzen_estimator.constants import uniform_weight
from parzen_estimator.loglikelihoods import compute_config_loglikelihoods
from parzen_estimator.utils import exp

from scipy.stats.qmc import LatinHypercube as LHS
from scipy.stats.qmc import Sobol


SAMPLING_CHOICES = {"sobol": Sobol, "lhs": LHS}
ParzenEstimatorType = Union[CategoricalParzenEstimator, NumericalParzenEstimator]


class MultiVariateParzenEstimator:
    def __init__(self, parzen_estimators: Dict[str, ParzenEstimatorType]):
        """
        MultiVariateParzenEstimator.

        Attributes:
            parzen_estimators (Dict[str, ParzenEstimatorType]):
                Parzen estimators for each hyperparameter.
            dim (int):
                The dimensions of search space.
            size (int):
                The number of observations used for the parzen estimators.
            weight (np.ndarray):
                The weight value for each basis.
        """
        self._parzen_estimators = parzen_estimators
        self._dim = len(parzen_estimators)
        self._size = list(parzen_estimators.values())[0].size
        self._weight = uniform_weight(self._size)

        if any(pe.size != self._size for pe in parzen_estimators.values()):
            raise ValueError("All parzen estimators must be identical.")

    def __repr__(self) -> str:
        return "\n".join(
            [f"({idx + 1}): {hp_name}\n{pe}" for idx, (hp_name, pe) in enumerate(self._parzen_estimators.items())]
        )

    def dimension_wise_pdf(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Compute the probability density value in each dimension given data points X.

        Args:
            X (List[np.ndarray]):
                Data points with the shape of (dim, n_samples)

        Returns:
            pdf_values (np.ndarray):
                The density values in each dimension for each data point.
                The shape is (dim, n_samples).
        """
        n_samples = X[0].size
        pdfs = np.zeros((self._dim, n_samples))

        for d, (hp_name, pe) in enumerate(self._parzen_estimators.items()):
            pdfs[d] += pe.pdf(X[d])

        return pdfs

    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the probability density value given data points X.

        Args:
            X (np.ndarray):
                Data points with the shape of (dim, n_samples)

        Returns:
            log_pdf_values (np.ndarray):
                The log density values for each data point.
                The shape is (n_samples, ).
        """
        n_samples = X[0].size
        blls = np.zeros((self._dim, self._size, n_samples))

        for d, (hp_name, pe) in enumerate(self._parzen_estimators.items()):
            blls[d] += pe.basis_loglikelihood(X[d])

        config_ll = compute_config_loglikelihoods(blls, self._weight)
        return config_ll

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the probability density value given data points X.

        Args:
            X (np.ndarray):
                Data points with the shape of (dim, n_samples)

        Returns:
            pdf_values (np.ndarray):
                The density values for each data point.
                The shape is (n_samples, ).
        """
        return exp(self.log_pdf(X))

    def sample(self, n_samples: int, rng: np.random.RandomState, dim_independent: bool = False) -> List[np.ndarray]:
        samples = []
        if dim_independent:
            samples = [pe.sample(rng, n_samples) for d, pe in enumerate(self._parzen_estimators.values())]
        else:
            indices = rng.randint(self._size, size=n_samples)
            samples = [pe.sample_by_indices(rng, indices) for d, pe in enumerate(self._parzen_estimators.values())]

        return samples

    def uniform_sample(
        self,
        n_samples: int,
        rng: np.random.RandomState,
        sampling_method: str = "sobol",  # Literal["sobol", "lhs"]
    ) -> List[np.ndarray]:
        """
        Sample points using latin hypercube sampling.

        Args:
            parzen_estimator (List[ParzenEstimatorType]):
                The list that contains the information of each dimension.
            n_samples (int):
                The number of samples.

        Returns:
            samples (List[np.ndarray]):
                Random samplings converted accordingly.
                The shape is (dim, n_samples).
        """
        if sampling_method not in SAMPLING_CHOICES:
            raise ValueError(f"sampling_method must be in {SAMPLING_CHOICES}, but got {sampling_method}")

        sampler = SAMPLING_CHOICES[sampling_method](d=self._dim, seed=rng)
        # We need to do it to maintain dtype for each dimension
        samples = [sample for sample in sampler.random(n=n_samples).T]

        for d, pe in enumerate(self._parzen_estimators.values()):
            samples[d] = pe.uniform_to_valid_range(samples[d])

        return samples

    @property
    def size(self) -> int:
        return self._size

    @property
    def dim(self) -> int:
        return self._dim
