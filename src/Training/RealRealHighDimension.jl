function generate_startingMPS(chi_init::Integer, site_indices::Vector{Index{T}}, num_classes::Integer, opts::Options;
    init_rng=nothing, label_tag::String="f(x)", verbosity::Real=opts.verbosity, dtype::DataType=opts.dtype) where {T <: Integer}
    """Generate the starting weight MPS, W using values sampled from a 
    Gaussian (normal) distribution. Accepts a chi_init parameter which
    specifies the initial (uniform) bond dimension of the MPS."""
    verbosity = verbosity

    if init_rng !== nothing
        # use seed if specified
        Random.seed!(init_rng)
        verbosity >= 0 && println("Generating initial weight MPS with bond dimension χ_init = $chi_init
        using random state $init_rng.")
    else
        verbosity >= 0 && println("Generating initial weight MPS with bond dimension χ_init = $chi_init.")
    end

    W = random_mps(dtype, site_indices, linkdims=chi_init)

    label_idx = Index(num_classes, label_tag)

    # get the site of interest and copy over the indices at the last site where we attach the label 
    old_site_idxs = inds(W[end])
    # new_site_idxs =  (old_site_idxs..., label_idx)
    new_site_idxs =  label_idx, old_site_idxs

    new_site = random_itensor(dtype,new_site_idxs)

    # add the new site back into the MPS
    W[end] = new_site

    # normalise the MPS
    normalize!(W)

    # canonicalise - bring MPS into canonical form by making all tensors 1,...,j-1 left orthogonal
    # here we assume we start at the right most index
    last_site = length(site_indices)
    orthogonalize!(W, last_site)

    return W

end



function construct_caches(W::MPS, training_pstates::TimeSeriesIterable; going_left=true, dtype::DataType=ComplexF64)
    """Function to pre-compute tensor contractions between the MPS and the product states. """

    # get the num of training samples to pre-allocate a caching matrix
    N_train, N = size(training_pstates)[2:3] 

    # pre-allocate left and right environment matrices 
    LE = PCache(undef, N_train, N) 
    RE = PCache(undef, N_train, N)

    if going_left
        # backward direction - initialise the LE with the first site
        w_1 = matrix(W[1])
        for i = 1:N_train
            LE[i,1] = transpose(training_pstates[:,i,1]' * w_1)
        end

        for j = 2:N-1
            t = tensor(W[j])
            for i = 1:N_train
                LE[i,j] = [ (training_pstates[:,i,j]' * sl) * LE[i,j-1] for sl in eachslice(t, dims=3)]
            end
        end
    
    else
        # going right
        # initialise RE cache with the terminal site and work backwards
        w_end = matrix(W[N])
        for i = 1:N_train
            RE[i,N] = transpose(training_pstates[:,i,N]' * w_end)
        end

        for j = (N-1):-1:2
            t = tensor(W[j])
      


            for i = 1:N_train
                RE[i,j] =  [(training_pstates[:,i,j]' * sl) * RE[i,j+1] for sl in eachslice(t, dims=3) ] 
            end
        end
        
    end

    @assert !isa(eltype(eltype(RE)), dtype) || !isa(eltype(eltype(LE)), dtype)  "Caches are not the correct datatype!"

    return LE, RE

end



function update_caches!(left_site_new::ITensor, right_site_new::ITensor, 
    LE::PCache, RE::PCache, lid::Int, rid::Int, training_pstates::TimeSeriesIterable; going_left::Bool=true)
    """Given a newly updated bond tensor, update the caches."""
    num_train, num_sites = size(training_pstates)[2:3]

    # @show inds(right_site_new)
    # @show inds(left_site_new)

    if going_left
        for i = 1:num_train
            if rid == num_sites
                RE[i,num_sites] = (training_pstates[:,i,rid]' * matrix(right_site_new))'
            else
                RE[i,rid] = [(training_pstates[:,i,rid]' * sl) * RE[i,rid+1] for sl in eachslice(tensor(right_site_new), dims=3) ]
            end
        end
    else
        # going right
        for i = 1:num_train
            if lid == 1
                m = matrix(left_site_new)
                LE[i,1] = (training_pstates[:,i,lid]' * m)'
            else
                LE[i,lid] = [(training_pstates[:,i,lid]' * sl) * LE[i,lid-1] for sl in eachslice(tensor(left_site_new), dims=3) ]
            end
        end
    end

end


function decomposeBT(BT::ITensor, lid::Int, rid::Int; 
    chi_max=nothing, cutoff=nothing, going_left=true, dtype::DataType=ComplexF64)
    """Decompose an updated bond tensor back into two tensors using SVD"""



    if going_left
        left_site_index = find_index(BT, "n=$lid")
        label_idx = find_index(BT, "f(x)")
        # need to make sure the label index is transferred to the next site to be updated
        if lid == 1
            U, S, V = svd(BT, (label_idx, left_site_index); maxdim=chi_max, cutoff=cutoff)
        else
            bond_index = find_index(BT, "Link,l=$(lid-1)")
            U, S, V = svd(BT, (bond_index, label_idx, left_site_index); maxdim=chi_max, cutoff=cutoff)
        end
        # absorb singular values into the next site to update to preserve canonicalisation
        left_site_new = U * S
        right_site_new = V
        # fix tag names 
        replacetags!(left_site_new, "Link,v", "Link,l=$lid")
        replacetags!(right_site_new, "Link,v", "Link,l=$lid")
    else
        # going right, label index automatically moves to the next site
        right_site_index = find_index(BT, "n=$rid")
        label_idx = find_index(BT, "f(x)")
        bond_index = find_index(BT, "Link,l=$(lid+1)")


        if isnothing(bond_index)
            V, S, U = svd(BT, (label_idx, right_site_index); maxdim=chi_max, cutoff=cutoff)
        else
            V, S, U = svd(BT, (bond_index, label_idx, right_site_index); maxdim=chi_max, cutoff=cutoff)
        end
        # absorb into next site to be updated 
        left_site_new = U
        right_site_new = V * S
        # fix tag names 
        replacetags!(left_site_new, "Link,v", "Link,l=$lid")
        replacetags!(right_site_new, "Link,v", "Link,l=$lid")
        # @show inds(left_site_new)
        # @show inds(right_site_new)

    end


    return left_site_new, right_site_new

end

function decomposeBT(
        BT::BondTensor, 
        lid::Int, 
        rid::Int; 
        chi_max=nothing, 
        cutoff=nothing, 
        going_left=true, 
        dtype::DataType=ComplexF64
    )
    """Decompose an updated bond tensor back into two tensors using SVD"""



    if going_left
        left_site_index = find_index(BT, "n=$lid")
        label_idx = find_index(BT, "f(x)")
        # need to make sure the label index is transferred to the next site to be updated
        if lid == 1
            U, S, V = svd(BT, (label_idx, left_site_index); maxdim=chi_max, cutoff=cutoff)
        else
            bond_index = find_index(BT, "Link,l=$(lid-1)")
            U, S, V = svd(BT, (bond_index, label_idx, left_site_index); maxdim=chi_max, cutoff=cutoff)
        end
        # absorb singular values into the next site to update to preserve canonicalisation
        left_site_new = U * S
        right_site_new = V
        # fix tag names 
        replacetags!(left_site_new, "Link,v", "Link,l=$lid")
        replacetags!(right_site_new, "Link,v", "Link,l=$lid")
    else
        # going right, label index automatically moves to the next site
        right_site_index = find_index(BT, "n=$rid")
        label_idx = find_index(BT, "f(x)")
        bond_index = find_index(BT, "Link,l=$(lid+1)")


        if isnothing(bond_index)
            V, S, U = svd(BT, (label_idx, right_site_index); maxdim=chi_max, cutoff=cutoff)
        else
            V, S, U = svd(BT, (bond_index, label_idx, right_site_index); maxdim=chi_max, cutoff=cutoff)
        end
        # absorb into next site to be updated 
        left_site_new = U
        right_site_new = V * S
        # fix tag names 
        replacetags!(left_site_new, "Link,v", "Link,l=$lid")
        replacetags!(right_site_new, "Link,v", "Link,l=$lid")
        # @show inds(left_site_new)
        # @show inds(right_site_new)

    end


    return left_site_new, right_site_new

end


function flatten_bt(s1, s2, label_idx, dtype=Float64; going_left=true)

    # infer the location of the site indices
    nc_inds = filter(i-> !hastags(i, "f(x)"), noncommoninds(s1,s2))
    len = prod(ITensors.dim.(nc_inds))
    bt_f = Matrix{dtype}(undef, len, ITensors.dim(label_idx))
    for (ci,c) in enumerate(eachindval(label_idx))
        if going_left
            bt = tensor( s1 * (s2 * onehot(c)) )
        else
            bt = tensor((s1 * onehot(c)) * s2)

        end
        
        bt_f[:,ci] = bt[:] # flatten
    end
    return bt_f, (nc_inds..., label_idx)
end

function unflatten_bt(bt_new, bt_inds)

    return itensor(reshape(bt_new, ITensors.dim(bt_inds)), bt_inds)
end


function realise(B::ITensor, C_index::Index{Int64}; dtype::DataType=ComplexF64)
    """Converts a Complex {s} dimension r itensor into a eal 2x{s} dimension itensor. Increases the rank from rank{s} to 1+ rank{s} by adding a 2-dimensional index "C_index" to the start"""
    ib = inds(B)
    inds_c = C_index,ib
    B_m = Array{dtype}(B, ib)

    out = Array{real(dtype)}(undef, 2,size(B)...)
    
    ls = eachslice(out; dims=1)
    
    ls[1] = real(B_m)
    ls[2] = imag(B_m)

    return ITensor(real(dtype), out, inds_c)
end


function complexify(B::ITensor, C_index::Index{Int64}; dtype::DataType=ComplexF64)
    """Converts a real 2x{s} dimension itensor into a Complex {s} dimension itensor. Reduces the rank from rank{s}+1 to rank{s} by removing the first index"""
    ib = inds(B)
    C_index, c_inds... = ib
    B_ra = NDTensors.array(B, ib) # should return a view


    re_part = selectdim(B_ra, 1,1);
    im_part = selectdim(B_ra, 1,2);

    return ITensor(dtype, complex.(re_part,im_part), c_inds)
end





function loss_grad_enforce_real(tsep::TrainSeparate, BT::ITensor, LE::PCache, RE::PCache,
    ETSs::EncodedTimeSeriesSet, lid::Int, rid::Int, C_index::Union{Index{Int64},Nothing}; dtype::DataType=ComplexF64, loss_grad::Function=loss_grad_KLD)
    """Function for computing the loss function and the gradient over all samples using a left and right cache. 
        Takes a real itensor and will convert it to complex before calling loss_grad if dtype is complex. Returns a real gradient. """
    

    if isnothing(C_index) # the itensor is real
        loss, grad = loss_grad(tsep, BT, LE, RE, ETSs, lid, rid)
    else
        # pass in a complex itensor
        BT_c = complexify(BT, C_index; dtype=dtype)

        loss, grad = loss_grad(tsep, BT_c, LE, RE, ETSs, lid, rid)

        grad = realise(grad, C_index; dtype=dtype)
    end


    return loss, grad

end

function loss_grad!(tsep::TrainSeparate, F,G,B_flat::AbstractArray, b_inds::Tuple{Vararg{Index{Int64}}}, LE::PCache, RE::PCache,
    ETSs::EncodedTimeSeriesSet, lid::Int, rid::Int, C_index::Union{Index{Int64},Nothing}; dtype::DataType=ComplexF64, loss_grad::Function=loss_grad_KLD)

    """Calculates the loss and gradient in a way compatible with Optim. Takes a flat, real array and converts it into an itensor before it passes it loss_grad """
    BT = itensor(real(dtype), B_flat, b_inds) # convert the bond tensor from a flat array to an itensor

    loss, grad = loss_grad_enforce_real(tsep, BT, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)

    if !isnothing(G)
        G .= NDTensors.array(grad,b_inds)
    end

    if !isnothing(F)
        return loss
    end

end

function custGD(tsep::TrainSeparate, BT_init::BondTensor, LE::PCache, RE::PCache, lid::Int, rid::Int, ETSs::EncodedTimeSeriesSet;
    iters=10, verbosity::Real=1, dtype::DataType=ComplexF64, loss_grad::Function=loss_grad_KLD, track_cost::Bool=false, eta::Real=0.01)
    BT = copy(BT_init)

    for i in 1:iters
        # get the gradient
        loss, grad = loss_grad(tsep, BT, LE, RE, ETSs, lid, rid)
        #zygote_gradient_per_batch(bt_old, LE, RE, pss, lid, rid)
        # update the bond tensor
        @. BT -= eta * grad
        if verbosity >=1 && track_cost
            # get the new loss
            println("Loss at step $i: $loss")
        end

    end

    return BT
end

function TSGO(tsep::TrainSeparate, BT_init::BondTensor, LE::PCache, RE::PCache, lid::Int, rid::Int, ETSs::EncodedTimeSeriesSet;
    iters=10, verbosity::Real=1, dtype::DataType=ComplexF64, loss_grad::Function=loss_grad_KLD, track_cost::Bool=false, eta::Real=0.01)
    BT = copy(BT_init) # perhaps not necessary?
    for i in 1:iters
        # get the gradient
        loss, grad = loss_grad(tsep, BT, LE, RE, ETSs, lid, rid)
        # update the bond tensor   
        
        # @show isassigned(Array(grad, inds(grad)))
        # grad /= norm(grad)
        # BT .-= eta .* grad

        # BT .-= eta .* (grad / norm(grad))

        # just sidestep itensor completely for this one?
        #@fastmath map!((x,y)-> x - eta * y / norm(grad), tensor(BT).storage.data, tensor(BT).storage.data,tensor(grad).storage.data )
        @. BT -= eta * $/(grad, $norm(grad)) #TODO investigate the absolutely bizarre behaviour that happens here with bigfloats if the arithmetic order is changed
        if verbosity >=1 && track_cost
            # get the new loss
            println("Loss at step $i: $loss")
        end

    end
    return BT
end

function apply_update(tsep::TrainSeparate, BT_init::BondTensor, LE::PCache, RE::PCache, lid::Int, rid::Int,
    ETSs::EncodedTimeSeriesSet; iters=10, verbosity::Real=1, dtype::DataType=ComplexF64, loss_grad::Function=loss_grad_KLD, bbopt::BBOpt=BBOpt("Optim"),
    track_cost::Bool=false, eta::Real=0.01, rescale::Tuple{Bool,Bool} = (false, true))
    """Apply update to bond tensor using the method specified by BBOpt. Will normalise B before and/or after it computes the update B+dB depending on the value of rescale [before::Bool,after::Bool]"""

    iscomplex = !(dtype <: Real)

    if rescale[1]
        normalize!(BT_init)
    end

    if bbopt.name == "CustomGD"
        if uppercase(bbopt.fl) == "GD"
            BT_new = custGD(tsep, BT_init, LE, RE, lid, rid, ETSs; iters=iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad, track_cost=track_cost, eta=eta)

        elseif uppercase(bbopt.fl) == "TSGO"
            BT_new = TSGO(tsep, BT_init, LE, RE, lid, rid, ETSs; iters=iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad, track_cost=track_cost, eta=eta)

        end
    else
        # break down the bond tensor to feed into optimkit or optim
        # if iscomplex
        #     C_index = Index(2, "C")
        #     bt_re = realise(BT_init, C_index; dtype=dtype)
        # else
        #     C_index = nothing
        #     bt_re = BT_init
        # end

        # if bbopt.name == "Optim" 
        #      # flatten bond tensor into a vector and get the indices
        #     bt_inds = inds(bt_re)
        #     bt_flat = NDTensors.array(bt_re, bt_inds) # should return a view

        #     # create anonymous function to feed into optim, function of bond tensor only
        #     fgcustom! = (F,G,B) -> loss_grad!(tsep, F, G, B, bt_inds, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)
        #     # set the optimisation manfiold
        #     # apply optim using specified gradient descent algorithm and corresp. paramters 
        #     # set the manifold to either flat, sphere or Stiefel 
        #     if bbopt.fl == "CGD"
        #         method = Optim.ConjugateGradient(eta=eta)
        #     else
        #         method = Optim.GradientDescent(alphaguess=eta)
        #     end
        #     #method = Optim.LBFGS()
        #     res = Optim.optimize(Optim.only_fg!(fgcustom!), bt_flat; method=method, iterations = iters, 
        #     show_trace = (verbosity >=1),  g_abstol=1e-20)
        #     result_flattened = Optim.minimizer(res)

        #     BT_new = itensor(real(dtype), result_flattened, bt_inds)


        # elseif bbopt.name == "OptimKit"

        #     lg = BT -> loss_grad_enforce_real(tsep, BT, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)
        #     if bbopt.fl == "CGD"
        #         alg = OptimKit.ConjugateGradient(; verbosity=verbosity, maxiter=iters)
        #     else
        #         alg = OptimKit.GradientDescent(; verbosity=verbosity, maxiter=iters)
        #     end
        #     BT_new, fx, _ = OptimKit.optimize(lg, bt_re, alg)


        # else
            # error("Unknown Black Box Optimiser $bbopt, options are [CustomGD, Optim, OptimKit]")
        error("Unknown Black Box Optimiser $bbopt, options are [CustomGD, TSGO]")

        # end

        if iscomplex # convert back to a complex itensor
            BT_new = complexify(BT_new, C_index; dtype=dtype)
        end
    end

    if rescale[2]
        normalize!(BT_new)
    end

    if track_cost
        loss, grad = loss_grad(tsep, BT_new, LE, RE, ETSs, lid, rid)
        println("Loss at site $lid*$rid: $loss")
    end

    return BT_new

end


# ensure the presence of the DIR value type 
# This is the intended entrypoint for calls to fitMPS, so input sanitisation can be done here
# If you call a method further down it's assumed you know what you're doing
#TODO fix the opts so it isnt such a disaster


"""
```Julia
fitMPS(X_train::AbstractMatrix, 
       y_train::AbstractVector=zeros(Int, size(X_train, 1)), 
       X_test::AbstractMatrix=zeros(0,0), 
       y_test::AbstractVector=zeros(Int, 0), 
       opts::AbstractMPSOptions=MPSOptions(),
       custom_encoding::Union{Encoding, Nothing}=nothing) -> (MPS::TrainedMPS, training_info::Dict, encoded_test_states::EncodedTimeSeriesSet)
```

Train an MPS on the data `X_train` using the hyperparameters `opts`, see [`MPSOptions`](@ref). The number of classes are determined by the entries of `y_train`.

Returns a trained MPS, a dictionary containing training info, and the encoded test states. `X_test` and `y_test` are used only to print performance evaluations, and may be empty. The `custom_encoding` argument allows the use of user defined custom encodings, see [`function_basis`](@ref). This requires that `encoding=:Custom` is specified in [`MPSOptions`](@ref)

NOTE: the return value `encoded_test_states` will be sorted by class, so predictions shouldn't be compared directly to `y_test`. 

See also: [`Encoding`](@ref)

# Example
See ??fitMPS to for a more verbose example

```
julia> opts = MPSOptions(; d=5, chi_max=30, encoding=:Legendre, eta=0.05);
julia> print_opts(opts) # Prints options as a table
       ...
julia> W, info, test_states = fitMPS( X_train, y_train, X_test, y_test, opts);
Generating initial weight MPS with bond dimension χ_init = 4
        using random state 1234.
Initialising train states.
Using 1 iterations per update.
Training KL Div. 28.213032851945012 | Training acc. 0.31343283582089554.
Using optimiser CustomGD with the "TSGO" algorithm
Starting backward sweeep: [1/5]
        ...

Starting forward sweep: [5/5]
Finished sweep 5. Time for sweep: 0.76s
Training KL Div. -12.577920427063361 | Training acc. 1.0.

MPS normalised!

Training KL Div. -12.57792042706337 | Training acc. 1.0.
Test KL Div. -9.815236609211746 | Testing acc. 0.9504373177842566.

Test conf: [497 16; 35 481].

julia> 

```

# Extended help
```
julia> Using JLD2 # load some data
julia> dloc = "test/Data/italypower/datasets/ItalyPowerDemandOrig.jld2"
julia> f = jldopen(dloc, "r") 
           X_train = read(f, "X_train")
           y_train = read(f, "y_train")
           X_test = read(f, "X_test")
           y_test = read(f, "y_test")
       close(f);
julia> opts = MPSOptions(; d=5, chi_max=30, encoding=:Legendre, eta=0.05);
julia> print_opts(opts) # Prints options as a table
       ...
julia> W, info, test_states = fitMPS( X_train, y_train, X_test, y_test, opts);
Generating initial weight MPS with bond dimension χ_init = 4
        using random state 1234.
Initialising train states.
Using 1 iterations per update.
Training KL Div. 28.213032851945012 | Training acc. 0.31343283582089554.
Using optimiser CustomGD with the "TSGO" algorithm
Starting backward sweeep: [1/5]
        ...

Starting forward sweep: [5/5]
Finished sweep 5. Time for sweep: 0.76s
Training KL Div. -12.577920427063361 | Training acc. 1.0.

MPS normalised!

Training KL Div. -12.57792042706337 | Training acc. 1.0.
Test KL Div. -9.815236609211746 | Testing acc. 0.9504373177842566.

Test conf: [497 16; 35 481].

julia> get_training_summary(W, test_states; print_stats=true);
         Overlap Matrix
┌──────┬───────────┬───────────┐
│      │   |ψ0⟩    │   |ψ1⟩    │
├──────┼───────────┼───────────┤
│ ⟨ψ0| │ 5.074e-01 │ 1.463e-02 │
├──────┼───────────┼───────────┤
│ ⟨ψ1| │ 1.463e-02 │ 4.926e-01 │
└──────┴───────────┴───────────┘
          Confusion Matrix
┌──────────┬───────────┬───────────┐
│          │ Pred. |0⟩ │ Pred. |1⟩ │
├──────────┼───────────┼───────────┤
│ True |0⟩ │       497 │        16 │
├──────────┼───────────┼───────────┤
│ True |1⟩ │        35 │       481 │
└──────────┴───────────┴───────────┘
┌───────────────────┬───────────┬──────────┬──────────┬─────────────┬──────────┬───────────┐
│ test_balanced_acc │ train_acc │ test_acc │ f1_score │ specificity │   recall │ precision │
│           Float64 │   Float64 │  Float64 │  Float64 │     Float64 │  Float64 │   Float64 │
├───────────────────┼───────────┼──────────┼──────────┼─────────────┼──────────┼───────────┤
│          0.950491 │       1.0 │ 0.950437 │ 0.950425 │    0.950491 │ 0.950491 │  0.951009 │
└───────────────────┴───────────┴──────────┴──────────┴─────────────┴──────────┴───────────┘

julia> sweep_summary(info)
┌────────────────┬──────────┬───────────────┬───────────────┬───────────────┬───────────────┬───────────────┬────────────┬──────────┐
│                │ Initial  │ After Sweep 1 │ After Sweep 2 │ After Sweep 3 │ After Sweep 4 │ After Sweep 5 │ After Norm │   Mean   │
├────────────────┼──────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼────────────┼──────────┤
│ Train Accuracy │ 0.313433 │      1.0      │      1.0      │      1.0      │      1.0      │      1.0      │    1.0     │   1.0    │
├────────────────┼──────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼────────────┼──────────┤
│  Test Accuracy │ 0.409135 │   0.947522    │   0.951409    │   0.948494    │   0.948494    │   0.950437    │  0.950437  │ 0.949271 │
├────────────────┼──────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼────────────┼──────────┤
│  Train KL Div. │  28.213  │   -11.7855    │    -12.391    │   -12.4831    │   -12.5466    │   -12.5779    │  -12.5779  │ -12.3568 │
├────────────────┼──────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼────────────┼──────────┤
│   Test KL Div. │ 27.7435  │   -9.12893    │   -9.73479    │   -9.79248    │    -9.8158    │   -9.81524    │  -9.81524  │ -9.65745 │
├────────────────┼──────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼────────────┼──────────┤
│     Time taken │   0.0    │   0.658366    │    0.75551    │   0.719035    │   0.718444    │    1.16256    │    NaN     │ 0.802783 │
└────────────────┴──────────┴───────────────┴───────────────┴───────────────┴───────────────┴───────────────┴────────────┴──────────┘

```
"""
function fitMPS(X_train::AbstractMatrix, y_train::AbstractVector, X_test::AbstractMatrix, y_test::AbstractVector, opts::AbstractMPSOptions=MPSOptions(), custom_encoding::Union{Encoding, Nothing}=nothing;  kwargs...)    
    # handle how the encoding is specified
    if opts isa Options
        @warn("Calling fitMPS with the Options struct is deprecated and can lead to serialisation issues! Use the MPSOptions struct instead.")
    end

    if !isnothing(custom_encoding)
        if opts isa Options
            throw(ArgumentError("Cannot use a custom encoding if using the Options struct, use MPSOptions instead"))

        elseif !(opts.encoding in [:custom, :Custom])
            throw(ArgumentError("To use a custom encoding, you must set \'encoding = :Custom\' in MPSOptions"))
        else
            opts = safe_options(opts) # make sure options isnt abstract
            opts = _set_options(opts; encoding=custom_encoding)
        end

    else
        opts = safe_options(opts) # make sure options is abstract

    end


    

    return fitMPS(DataIsRescaled{false}(), X_train, y_train, X_test, y_test, opts; kwargs...) 
end


# empty test data
fitMPS(X_train::AbstractMatrix, y_train::AbstractVector, opts::AbstractMPSOptions=MPSOptions(), custom_encoding::Union{Encoding, Nothing}=nothing; kwargs...) = fitMPS(X_train, y_train, zeros(0,0), zeros(Int, 0), opts, custom_encoding;  kwargs...)    

# completely unsupervised
fitMPS(X_train::AbstractMatrix, opts::AbstractMPSOptions=MPSOptions(), custom_encoding::Union{Encoding, Nothing}=nothing; kwargs...) = fitMPS(X_train, zeros(Int, size(X_train, 1)), zeros(0,0), zeros(Int, 0), opts, custom_encoding;  kwargs...)    


function fitMPS(DIS::DataIsRescaled, X_train::AbstractMatrix, y_train::AbstractVector, X_test::AbstractMatrix, y_test::AbstractVector, opts::AbstractMPSOptions; kwargs...)
    # first, create the site indices for the MPS and product states 
    

    opts = safe_options(opts) # make sure options isnt abstract


    if DIS isa DataIsRescaled{false}
        num_mps_sites = size(X_train, 2)
    else
        num_mps_sites = size(X_train, 1)
    end
    sites = siteinds(opts.d, num_mps_sites)

    # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
    num_classes = length(unique(y_train))
    W = generate_startingMPS(opts.chi_init, sites, num_classes, opts; init_rng=opts.init_rng)

    return fitMPS(DIS, W, X_train, y_train, X_test, y_test, opts; kwargs...)
    
end

function fitMPS(::DataIsRescaled{false}, W::MPS, X_train::AbstractMatrix, y_train::AbstractVector, X_test::AbstractMatrix, y_test::AbstractVector, opts::AbstractMPSOptions; kwargs...)
    @assert eltype(W[1]) == opts.dtype  "The MPS elements are of type $(eltype(W[1])) but the datatype is opts.dtype=$(opts.dtype)"
    opts = safe_options(opts) # make sure options is abstract
    
    X_train_scaled, X_test_scaled, norms, oob_rescales = transform_data(permutedims(X_train), permutedims(X_test); opts=opts)
    
    return fitMPS(DataIsRescaled{true}(), W, X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test, opts; kwargs...)

end

function fitMPS(::DataIsRescaled{true}, W::MPS, X_train::Matrix, X_train_scaled::Matrix, y_train::Vector, X_test::Matrix, X_test_scaled::Matrix, y_test::Vector, opts::AbstractMPSOptions; test_run=false, return_sample_encoding::Bool=false)
    opts = safe_options(opts) # make sure options is abstract
    # first, get the site indices for the product states from the MPS
    sites = get_siteinds(W)
    num_mps_sites = length(sites)
    @assert num_mps_sites == size(X_train_scaled, 1) && (size(X_test_scaled, 1) in [num_mps_sites, 0]) "The number of sites supported by the MPS, training, and testing data do not match! "


    @assert size(X_train_scaled, 2) == size(y_train, 1) "Size of training dataset and number of training labels are different!"
    @assert size(X_test_scaled, 2) == size(y_test, 1) "Size of testing dataset and number of testing labels are different!"

    # generate product states using rescaled data
    if opts.encoding.iscomplex
        if opts.dtype <: Real
            error("Using a complex valued encoding but the MPS is real. If using a complex-valued custom encoding, set 'dtype <: Complex' in MPSOptions")
        end

    elseif !(opts.dtype <: Real)
        @warn "Using a complex valued MPS but the encoding is real"
    end


    # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
    classes = unique(y_train)
    test_classes = unique(y_test)
    if !isempty(setdiff(test_classes, classes))
        throw(ArgumentError("Test set has classes not present in the training set, this is currently unsupported."))
    end

    num_classes = length(classes)
    _, label_idx = find_label(W)

    @assert num_classes == ITensors.dim(label_idx) "Number of Classes in the training data doesn't match the dimension of the label index!"
    @assert eltype(classes) <: Integer "Classes must be integers" #TODO fix PState so this is unnecessary
    sort!(classes)
    class_keys = Dict(zip(classes, 1:num_classes))

    
    s = EncodeSeparate{opts.encode_classes_separately}()
    training_states, enc_args_tr = encode_dataset(s, X_train, X_train_scaled, y_train, "train", sites; opts=opts, class_keys=class_keys)
    testing_states, enc_args_test = encode_dataset(s, X_test, X_test_scaled, y_test, "test", sites; opts=opts, class_keys=class_keys, training_encoding_args=enc_args_tr)
    

    return fitMPS(W, training_states, testing_states, opts; test_run=test_run)
end

function fitMPS(training_states_meta::EncodedTimeSeriesSet, testing_states_meta::EncodedTimeSeriesSet, opts::AbstractMPSOptions; test_run=false) # optimise bond tensor)
    # first, create the site indices for the MPS and product states 
    opts = safe_options(opts) # make sure options is abstract


    training_states = training_states_meta.timeseries

    @assert opts.d == length(training_states[1].data) "Dimension of site indices must match feature map dimension"
    sites = siteinds(opts.d, size(training_states, 2))

    # generate the starting MPS with unfirom bond dimension chi_init and random values (with seed if provided)
    num_classes = length(unique([ps.label for ps in training_states]))
    W = generate_startingMPS(opts.chi_init, sites, num_classes, opts; init_rng=opts.init_rng)

    fitMPS(W, training_states_meta, testing_states_meta, opts, test_run=test_run)

end



function fitMPS(W::MPS, training_states_meta::EncodedTimeSeriesSet, testing_states_meta::EncodedTimeSeriesSet, opts::AbstractMPSOptions=Options(); test_run=false) # optimise bond tensor)
     opts = safe_options(opts) # make sure options is abstract


    verbosity = opts.verbosity
    nsweeps = opts.nsweeps

    if test_run
        verbosity > -1 && println("Encoding completed! Returning initial states without training.")
        return W, [], training_states, testing_states, []
    end

    blas_name = GenericLinearAlgebra.LinearAlgebra.BLAS.get_config() |> string
    if !occursin("mkl", blas_name)
        @warn "Not using MKL BLAS, which may lead to worse performance.\nTo fix this, Import MPSTime into Julia first or use the MKL package"
        @show blas_name
    end

    # @unpack_Options opts # unpacks the attributes of opts into the local namespace
    tsep = TrainSeparate{opts.train_classes_separately}() # value type to determine training style

    

    training_states = training_states_meta.timeseries
    testing_states = testing_states_meta.timeseries
    sites = siteinds(W)

    if opts.encode_classes_separately && !opts.train_classes_separately
        @warn "Classes are encoded separately, but not trained separately"
    elseif opts.train_classes_separately && !opts.encode_classes_separately
        @warn "Classes are trained separately, but not encoded separately"
    end

    has_test = !isempty(testing_states_meta)

    verbosity > -1 && println("Using $(opts.update_iters) iterations per update.")
    # construct initial caches
    LE, RE = construct_caches(W, training_states; going_left=true, dtype=opts.dtype)


    # create structures to store training information

    if has_test
        training_information = Dict(
            "train_loss" => Float64[],
            "train_acc" => Float64[],
            "test_loss" => Float64[],
            "test_acc" => Float64[],
            "time_taken" => Float64[], # sweep duration
            "train_KL_div" => Float64[],
            "test_KL_div" => Float64[],
            "test_conf" => Matrix{Float64}[]
        )
    else
        training_information = Dict(
        "train_loss" => Float64[],
        "train_acc" => Float64[],
        "test_loss" => Float64[],
        "time_taken" => Float64[], # sweep duration
        "train_KL_div" => Float64[]
    )
    end

    if opts.log_level > 0

        # compute initial training and validation acc/loss
        init_train_loss, init_train_acc = MSE_loss_acc(W, training_states)
        train_KL_div = KL_div(W, training_states)
        
        push!(training_information["train_loss"], init_train_loss)
        push!(training_information["train_acc"], init_train_acc)
        push!(training_information["time_taken"], 0.)
        push!(training_information["train_KL_div"], train_KL_div)


        if has_test 
            init_test_loss, init_test_acc, conf = MSE_loss_acc_conf(W, testing_states)
            init_KL_div = KL_div(W, testing_states)

            push!(training_information["test_loss"], init_test_loss)
            push!(training_information["test_acc"], init_test_acc)
            push!(training_information["test_KL_div"], init_KL_div)
            push!(training_information["test_conf"], conf)
        end
    

        #print loss and acc
        if verbosity > -1
            println("Training KL Div. $train_KL_div | Training acc. $init_train_acc.")# | Training MSE: $init_train_loss." )

            if has_test 
                println("Test KL Div. $init_KL_div | Testing acc. $init_test_acc.")#  | Testing MSE: $init_test_loss." )
                println("")
                println("Test conf: $conf.")
            end

        end
    end


    # initialising loss algorithms
    if typeof(opts.loss_grad) <: AbstractArray
        @assert length(opts.loss_grad) == nsweeps "loss_grad(...)::(loss,grad) must be a loss function or an array of loss functions with length nsweeps"
        loss_grads = opts.loss_grad
    elseif typeof(opts.loss_grad) <: Function
        loss_grads = [opts.loss_grad for _ in 1:nsweeps]
    else
        error("loss_grad(...)::(loss,grad) must be a loss function or an array of loss functions with length nsweeps")
    end

    if opts.train_classes_separately && !(eltype(loss_grads) <: KLDLoss)
        @warn "Classes will be trained separately, but the cost function _may_ depend on measurements of multiple classes. Switch to a KLD style cost function or ensure your custom cost function depends only on one class at a time."
    end

    if typeof(opts.bbopt) <: AbstractArray
        @assert length(opts.bbopt) == nsweeps "bbopt must be an optimiser or an array of optimisers to use with length nsweeps"
        bbopts = opts.bbopt
    elseif typeof(opts.bbopt) <: BBOpt
        bbopts = [opts.bbopt for _ in 1:nsweeps]
    else
        error("bbopt must be an optimiser or an array of optimisers to use with length nsweeps")
    end

    # start the sweep
    update_iters = opts.update_iters
    dtype = opts.dtype
    track_cost = opts.track_cost
    eta = opts.eta
    chi_max = opts.chi_max
    rescale = opts.rescale
    cutoff=opts.cutoff

    pos, label_idx = find_label(W)
    # bts = Vector{BondTensor{dtype}}(undef, ITensors.dim(label_idx))
    for itS = 1:opts.nsweeps        
        start = time()
        verbosity > -1 && println("Using optimiser $(bbopts[itS].name) with the \"$(bbopts[itS].fl)\" algorithm")
        verbosity > -1 && println("Starting backward sweeep: [$itS/$nsweeps]")
        sinds = siteinds(W)
        sinds[end] = inds(W[end])[2]
        for j = (length(sites)-1):-1:1
            # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
            bt, bt_inds = flatten_bt(W[j], W[(j+1)], label_idx, dtype; going_left=true) 
            bt_new = apply_update(
                tsep, 
                bt, 
                LE, 
                RE, 
                j, 
                (j+1), 
                training_states_meta; 
                iters=update_iters, 
                verbosity=verbosity,                                
                dtype=dtype, 
                loss_grad=loss_grads[itS], 
                bbopt=bbopts[itS],
                track_cost=track_cost, 
                eta=eta, 
                rescale = rescale
            ) # optimise bond tensor
            bt_it = unflatten_bt(bt_new, bt_inds)

            # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
            lsn, rsn = decomposeBT(bt_it, j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=true, dtype=dtype)
                
            # update the caches to reflect the new tensors
            update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
            # place the updated sites back into the MPS
            W[j] = lsn
            W[(j+1)] = rsn
        end
    
        # add time taken for backward sweep.
        verbosity > -1 && println("Backward sweep finished.")
        
        # finished a full backward sweep, reset the caches and start again
        # this can be simplified dramatically, only need to reset the LE
        LE, RE = construct_caches(W, training_states; going_left=false)
        
        verbosity > -1 && println("Starting forward sweep: [$itS/$nsweeps]")

        

        for j = 1:(length(sites)-1)
            bt, bt_inds = flatten_bt(W[j], W[(j+1)], label_idx, dtype; going_left=false)
   
            bt_new = apply_update(
                tsep, 
                bt, 
                LE, 
                RE, 
                j, 
                (j+1), 
                training_states_meta; 
                iters=update_iters, 
                verbosity=verbosity, 
                dtype=dtype, 
                loss_grad=loss_grads[itS], 
                bbopt=bbopts[itS],
                track_cost=track_cost, 
                eta=eta, 
                rescale=rescale
            ) # optimise bond tensor

            bt_it = unflatten_bt(bt_new, bt_inds)
            lsn, rsn = decomposeBT(bt_it, j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=false, dtype=dtype)
            update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
            W[j] = lsn
            W[(j+1)] = rsn
        end

        LE, RE = construct_caches(W, training_states; going_left=true)
        
        finish = time()

        time_elapsed = finish - start
        
        # add time taken for full sweep 
        verbosity > -1 && println("Finished sweep $itS. Time for sweep: $(round(time_elapsed,digits=2))s")

        if opts.log_level > 0

            # compute the loss and acc on both training and validation sets
            train_loss, train_acc = MSE_loss_acc(W, training_states)
            train_KL_div = KL_div(W, training_states)


            push!(training_information["train_loss"], train_loss)
            push!(training_information["train_acc"], train_acc)
            push!(training_information["time_taken"], time_elapsed)
            push!(training_information["train_KL_div"], train_KL_div)


            if has_test 
                test_loss, test_acc, conf = MSE_loss_acc_conf(W, testing_states)
                test_KL_div = KL_div(W, testing_states)
        
                push!(training_information["test_loss"], test_loss)
                push!(training_information["test_acc"], test_acc)
                push!(training_information["test_KL_div"], test_KL_div)
                push!(training_information["test_conf"], conf)
            end
        

            if verbosity > -1
                println("Training KL Div. $train_KL_div | Training acc. $train_acc.")#  | Training MSE: $train_loss." )

                if has_test 
                    println("Test KL Div. $test_KL_div | Testing acc. $test_acc.")#  | Testing MSE: $test_loss." )
                    println("")
                    println("Test conf: $conf.")
                end

            end
        end

        if opts.exit_early && train_acc == 1.
            break
        end
       
    end
    normalize!(W)
    verbosity > -1 && println("\nMPS normalised!\n")
    if opts.log_level > 0

        # compute the loss and acc on both training and validation sets
        train_loss, train_acc = MSE_loss_acc(W, training_states)
        train_KL_div = KL_div(W, training_states)


        push!(training_information["train_loss"], train_loss)
        push!(training_information["train_acc"], train_acc)
        push!(training_information["time_taken"], NaN)
        push!(training_information["train_KL_div"], train_KL_div)


        if has_test 
            test_loss, test_acc, conf = MSE_loss_acc_conf(W, testing_states)
            test_KL_div = KL_div(W, testing_states)

            push!(training_information["test_loss"], test_loss)
            push!(training_information["test_acc"], test_acc)
            push!(training_information["test_KL_div"], test_KL_div)
            push!(training_information["test_conf"], conf)
        end
    

        if verbosity > -1
            println("Training KL Div. $train_KL_div | Training acc. $train_acc.")#  | Training MSE: $train_loss." )

            if has_test 
                println("Test KL Div. $test_KL_div | Testing acc. $test_acc.")#  | Testing MSE: $test_loss." )
                println("")
                println("Test conf: $conf.")
            end
        end
    end

   
    return TrainedMPS(W, MPSOptions(opts), training_states_meta), training_information, testing_states_meta

end
