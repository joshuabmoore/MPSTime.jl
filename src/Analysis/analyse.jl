
function von_neumann_entropy(mps::MPS)
    # make a deepcopy of the mps so we don't alter it
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
        normalize!(S)
        SvN = 0.0
        for n in 1:ITensors.dim(S, 1)
            p = S[n, n]^2
            if (p > 1E-12) # avoid log 0
                SvN += -p * log(p) 
            end
        end
        entropy[i] = SvN
    end
    return entropy
end

"""
```Julia
Compute the bipartite entanglement entropy (BEE) of a trained MPS across each bond.
Given a single unlabeled MPS the BEE is defined as:

∑ α^2 log(α^2)
where α are the eigenvalues obtained from the shmidt decomposition. 
    
```
Compute the bipartite entanglement entropy (BEE) of a trained MPS.
"""
function bipartite_spectrum(mps::TrainedMPS)
    # expand the label index 
    mpss, _ = expand_label_index(mps.mps);
    bees = Vector{Vector{Float64}}(undef, length(mpss))
    for i in eachindex(bees)
        bees[i] = von_neumann_entropy(mpss[i]);
    end
    return bees
end

function rho_correct(rho::Matrix, eigtol::Float64=eps(), abstol::Float64=5*eps())

    eigvals, eigvecs = eigen(rho) # do an eigendecomp on the rdm
    neg_eigs = findall(<(0), eigvals) # find negative eigenvalues
    if isempty(neg_eigs)
        return rho
    end
    # check eigenvalues within tolerance
    oot = findall(x -> x < -eigtol, eigvals) # out of tolerance
    if isempty(oot)
        # clamp negative eigenvalues to the range [tol, ∞]
        @info "Clamping the following negative eigenvalues to the range [$(tol), ∞]: λ = $(eigvals[neg_eigs])"
        eigs_clamped = clamp.(eigvals, tol, Inf)
    else
        @error "RDM contains large negative eigenvalues outside of the tolerance $tol: λ = $(oot...)"
    end
    # reconstruct the rdm with the clamped eigenvalues
    rho_corrected = eigvecs * Diagonal(eigs_clamped) * (vecs)'
    # assess reconstruction error
    delta_norm = norm((rho - rho_corrected), 2)
    if delta_norm > abstol
    # verify trace preservation
        @error "RDM reconstruction error larger than tolerance: $abstol ($delta_norm)"
    end
    # check trace
    if !isapprox(tr(rho_corrected), 1.0)
        @error "Tr(ρ_corrected) > 1.0!"
    end
    return rho_corrected
end

function one_site_rdm(mps::MPS, site::Int)

    s = siteinds(mps)
    orthogonalize!(mps, site)
    psi_dag = dag(mps) # conjugate transpose of MPS
    rho = matrix(prime(mps[site], s[site]) * psi_dag[site]) # compute the reduced density matrix
    rho_corrected = rho_correct(rho) # clamp negative eigenvalues
    return rho_corrected
end

"""
```Julia
Compute the single-site entanglement entropy (SEE) of a trained MPS. 
Given a single unlabeled MPS the SEE is defined as:

SEE = -tr(ρ ⋅ log ρ)

Where ρ is the single-site reduced density matrix (RDM).
```
Compute the single-site entanglement entropy (SEE) of a trained MPS.
"""
function single_site_spectrum(mps::MPS)
    N = length(mps)
    entropy = zeros(Float64, N)
    for i in 1:N
        rho = one_site_rdm(mps, i)
        rho_log_rho = rho * log(rho)
        entropy[i] = -tr(rho_log_rho)
    end
    return entropy
end
