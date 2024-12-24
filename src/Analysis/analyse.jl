"""
```Julia
von_neumann_entropy(mps::MPS; logfn::Function=log) -> Vector{Float64}
```
Compute the [von Neumann entanglement entropy](https://en.wikipedia.org/wiki/Entropy_of_entanglement) for each site in a Matrix Product State (MPS).

The von Neumann entropy quantifies the entanglement at each bond of the MPS by computing the entropy of the singular value spectrum obtained from a singular value decomposition (SVD). The entropy is computed as:

``S = -\\sum_{i} p_i \\log(p_i)``

where ``p_i`` are the squared singular values normalized to sum to 1.

# Arguments
- `mps::MPS`: The Matrix Product State (MPS) whose entanglement entropy is to be computed.
- `logfn::Function`: (Optional) The logarithm function to use (`log`, `log2`, or `log10`). Defaults to the natural logarithm (`log`).

# Returns
A vector of `Float64` values where the i-th element represents the von Neumann entropy at site i of the MPS.
"""
function von_neumann_entropy(mps::MPS, logfn::Function=log)
    # adapted from http://itensor.org/docs.cgi?page=formulas/entanglement_mps
    if !(logfn in (log, log2, log10))
        throw(ArgumentError("logfn must be one of: log, log2, or log10"))
    end
    N = length(mps)
    entropy = zeros(Float64, N)
    for i in eachindex(entropy)
        orthogonalize!(mps, i) # place orthogonality center on site i 
        S = 0.0
        if i == 1 || i == N
            _, S, _ = svd(mps[i], (siteind(mps, i))) # make the cut at bond i
        else
            _, S, _ = svd(mps[i], (linkind(mps, i-1), siteind(mps, i)))
        end
        SvN = 0.0
        for n in 1:ITensors.dim(S, 1)
            p = S[n, n]^2
            if (p > 1E-12) # avoid log 0
                SvN += -p * logfn(p) 
            end
        end
        entropy[i] = SvN
    end
    return entropy
end

"""
```Julia
bipartite_spectrum(mps::TrainedMPS; logfn::Function=log) -> Vector{Vector{Float64}}
```
Compute the bipartite entanglement entropy (BEE) of a trained MPS across each bond.
Given a single unlabeled MPS the BEE is defined as:

``\\sum_i \\alpha_i^2 \\log(\\alpha_i^2)``
\nwhere ``\\alpha_i`` are the eigenvalues obtained from the shmidt decomposition. 
"""
function bipartite_spectrum(mps::TrainedMPS; logfn::Function=log)
    if !(logfn in (log, log2, log10))
        throw(ArgumentError("logfn must be one of: log, log2, or log10"))
    end
    mpss, _ = expand_label_index(mps.mps);  # expand the label index 
    bees = Vector{Vector{Float64}}(undef, length(mpss))
    for i in eachindex(bees)
        bees[i] = von_neumann_entropy(mpss[i], logfn);
    end
    return bees
end

function rho_correct(rho::Matrix, eigentol::Float64=sqrt(eps()))
    
    eigvals, eigvecs = eigen(rho) # do an eigendecomp on the rdm
    neg_eigs = findall(<(0), eigvals) # find negative eigenvalues
    if isempty(neg_eigs)
        return rho
    end
    # check eigenvalues within tolerance
    oot = findall(x -> x < -eigentol, eigvals) # out of tolerance
    if isempty(oot)
        # clamp negative eigenvalues to the range [tol, ∞]
        eigs_clamped = clamp.(eigvals, eigentol, Inf)
    else
        throw(DomainError("RDM contains large negative eigenvalues outside of the tolerance $eigentol: λ = $(eigvals[oot]...)")) 
    end
    # reconstruct the rdm with the clamped eigenvalues
    rho_corrected = eigvecs * LinearAlgebra.Diagonal(eigs_clamped) * (eigvecs)'
    # check trace
    if !isapprox(tr(rho_corrected), 1.0)
       throw(DomainError("Tr(ρ_corrected) > 1.0!"))
    end
    return rho_corrected
end

"""
```Julia
one_site_rdm(mps::MPS, site::Int) -> Matrix
```
Compute the single-site reduced density matrix (RDM) of the 
MPS at a given site. 
If the RDM is not positive semidefinite, clamp the negative eigenvalues
(if within the tolerance) and reconstruct the rdm.
"""
function one_site_rdm(mps::MPS, site::Int)
    s = siteinds(mps)
    orthogonalize!(mps, site)
    psi_dag = dag(mps) # conjugate transpose of MPS
    rho = matrix(prime(mps[site], s[site]) * psi_dag[site]) # compute the reduced density matrix
    rho_corrected = rho_correct(rho) # clamp negative eigenvalues to pos range
    return rho_corrected
end

function single_site_entropy(mps::MPS)
    N = length(mps)
    entropy = zeros(Float64, N)
    for i in 1:N
        rho = one_site_rdm(mps, i)
        rho_log_rho = rho * log(rho)
        entropy[i] = -tr(rho_log_rho)
    end
    return entropy
end

"""
```Julia
single_site_spectrum(mps::TrainedMPS) -> Vector{Vector{Float64}}
```
Compute the single-site entanglement entropy (SEE) spectrum of a trained MPS.

The single-site entanglement entropy (SEE) quantifies the entanglement at each site of the MPS. It is computed as:

``\\text{SEE} = -\\text{tr}(\\rho \\log(\\rho))``

where ``\\rho`` is the single-site reduced density matrix (RDM).

# Arguments
- `mps::TrainedMPS`: A trained Matrix Product State (MPS).

# Returns
A vector of vectors, where the outer vector corresponds to each label in the expanded MPS, and the inner vectors contain the SEE values for the respective sites.
"""
function single_site_spectrum(mps::TrainedMPS)
    # expand the label index 
    mpss, _ = expand_label_index(mps.mps);
    sees = Vector{Vector{Float64}}(undef, length(mpss))
    for i in eachindex(sees)
        sees[i] = single_site_entropy(mpss[i]);
    end
    return sees
end
