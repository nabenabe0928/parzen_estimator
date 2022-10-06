from typing import Optional

import numpy as np

from torch import as_tensor
from torch import erf as torch_erf
from torch import exp as torch_exp
from torch import log as torch_log
from torch import logsumexp as torch_logsumexp


# torch implementation is quicker than that of numpy!
def log(x: np.ndarray) -> np.ndarray:
    return torch_log(as_tensor(x)).cpu().detach().numpy()


def logsumexp(x: np.ndarray, axis: Optional[int], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Output log sum exp(x).

    NOTE:
        log sum w * exp(x) = log sum exp(log(w)) * exp(x)
                           = log sum exp(x + log(w))
    """
    if weights is not None:
        x += log(weights)[:, np.newaxis]

    return torch_logsumexp(as_tensor(x), axis=axis).cpu().detach().numpy()


def exp(x: np.ndarray) -> np.ndarray:
    return torch_exp(as_tensor(x)).cpu().detach().numpy()


def erf(x: np.ndarray) -> np.ndarray:
    return torch_erf(as_tensor(x)).cpu().detach().numpy()
