[![Build Status](https://github.com/nabenabe0928/parzen_estimator/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/parzen_estimator)
[![codecov](https://codecov.io/gh/nabenabe0928/parzen_estimator/branch/main/graph/badge.svg?token=64WK4ZWGA2)](https://codecov.io/gh/nabenabe0928/parzen_estimator)

# Introduction
This package is mainly used for the implementation of tree-structured parzen estimator (TPE).
TPE is an hyperparameter optimization (HPO) method invented in [`Algorithms for Hyper-Parameter Optimization`](https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf).

**NOTE**: The parzen estimators are built based on the [BOHB](http://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf) implementation.

# Setup
This package requires python 3.7 or later version and you can install 

```
pip install parzen-estimator
```

# Running example

Please see [examples](examples/visualize_kde.ipynb).
