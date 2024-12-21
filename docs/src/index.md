# MPSTime.jl
A Julia package for time-series machine learning (ML) using Matrix-Product States (MPS) built on the [ITensors.jl](https://github.com/ITensor/ITensors.jl) framework [ITensor, ITensor-r0.3](@cite).

![](./assets/logo.svg)

## Overview

__MPSTime__ is a Julia package for learning the joint probability distribution of time series directly from data using [matrix product state (MPS)](https://en.wikipedia.org/wiki/Matrix_product_state) methods inspired by quantum many-body physics. 
It provides a unified formalism for:
- Time-series classification.
- Univariate time-series imputation across fixed-length time series.

!!! info "Info"
    MPSTime is currently under active development. Many features are in an experimental stage and may undergo significant changes, refinements, or removals.

## Installation
This is not yet a registered Julia package, but it will be soon (TM)! In the meantime, you can install it directly from github:

```Julia
julia> ]
pkg> add https://github.com/jmoo2880/MPSTime.jl.git
```

## Usage
See the [Tutorial](@ref) section and other sidebars for a basic usage examples. We're continually adding more features and documentation as we go.

## Citation
If you use MPSTime in your research, please cite the MPSTime Paper:
```
Coming soon (TM).
```