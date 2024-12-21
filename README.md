<p align="center">
  <picture>
    <source srcset="docs/src/assets/logo-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="docs/src/assets/logo.svg" alt="mpstime logo" height="200"/>
  </picture>
</p>


<h1 align="center"><em>MPSTime.jl</em>: Matrix-Product States for Time-Series Analysis</h1>


<div style="text-align: center;">
  <a href="https://jmoo2880.github.io/MPSTime.jl/dev/">
    <img src="https://img.shields.io/badge/docs-latest-2ea44f?style=flat-square" alt="docs - latest">
  </a>
  <a href="https://jmoo2880.github.io/MPSTime.jl/dev/">
    <img src="https://img.shields.io/badge/version-0.1.0--DEV-blue?style=flat-square" alt="version - 0.1.0-DEV">
  </a>
</div>


__MPSTime__ is a Julia package for learning the joint probability distribution of time series directly from data using matrix-product state (MPS) methods inspired by quantum many-body physics. 
It provides a unified formalism for classifying unseen data, as well as imputing gaps in time-series data, which regularly occur in real-world datasets due to sensor failure, routine maintenance, or other problems.

## Installation
This is not yet a registered Julia package, but it will be soon (TM)! In the meantime, you can install it directly from github:

```Julia
julia> ]
pkg> add https://github.com/jmoo2880/MPSTime.jl.git
```

## Usage and Documentation
We provide [tutorials](https://jmoo2880.github.io/MPSTime.jl/dev/tutorial/) and basic usage examples in the documentation:
- [LATEST DOCS](https://jmoo2880.github.io/MPSTime.jl/) -- the latest documentation.

## Citation
This software is citable via Coming Soon (TM). Please cite as:
> arXiv preprint

## Contributing to MPSTime
We welcome contributions from the community! 
MPSTime is under active development and resarch, making it an exciting opprtunity for collaboration.
Whether you are interested in adding new features, improving existing documentation, fixing bugs, or sharing ideas, your input is valuable.
Feel free to submit a [PR](https://github.com/jmoo2880/MPSTime.jl/pulls) or open an [issue](https://github.com/jmoo2880/MPSTime.jl/issues) for discussion.
