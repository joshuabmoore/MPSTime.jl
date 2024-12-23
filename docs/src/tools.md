# Tools

## Entanglement Entropy

### Overview

### Bipartite Entanglement Entropy (BEE)
Given a [`trainedMPS`](@ref), we can compute the bipartite entanglement entropy (BEE) using:
```Julia
bees = bipartite_spectrum(mps);
``` 
The return value is a vector corresponding to the BEE of each class mps.


### Single-Site Entanglement Entropy (SEE)
Given a [`trainedMPS`](@ref), we can compute the single-site entanglement entropy (SEE) using:
```Julia
sees = MPSTime.single_site_spectrum(mps);
``` 

## Docstrings
```@docs
MPSTime.von_neumann_entropy
MPSTime.bipartite_spectrum
```

## Internal Methods
```@docs
MPSTime.single_site_spectrum
```
