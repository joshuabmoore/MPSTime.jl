function is_omp_threading()
    return "OMP_NUM_THREADS" in keys(ENV) && ENV["OMP_NUM_THREADS"] == "1"
end


function evaluate(
    X::AbstractMatrix, 
    y::AbstractVector, 
    tuning_parameters::NamedTuple,
    tuning_optimiser=SAMIN(); # bounded simulated annealing from Optim
    objective::TuningLoss=ImputationLoss(), 
    verbosity::Integer=1,
    opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1),
    tuning_opts0::AbstractMPSOptions=opts0,
    nfolds::Integer=30,
    n_cvfolds::Integer=5,
    rng::Union{Integer, AbstractRNG}=1,
    tuning_rng::Union{Integer, AbstractRNG}=1,
    foldmethod::Union{Function, Vector}=make_stratified_folds, 
    tuning_foldmethod::Union{Function, Vector}=make_stratified_cvfolds, 
    pms::Union{Nothing, AbstractVector}=collect(0.05:0.15:0.95),
    tuning_pms::Union{Nothing, AbstractVector}=[0.05, 0.95],
    tuning_abstol::Float64=1e-3,
    tuning_maxiters::Integer=500,
    distribute_folds::Bool=false,   
    )
    

    abs_rng = rng isa Integer ? Xoshiro(rng) : rng

    folds::Vector = foldmethod isa Function ? foldmethod(X,y, nfolds; rng=abs_rng) : foldmethod

    if distribute_folds
        mapfunc = pmap
        if nprocs() == 1
            println("No workers")
        end
        threading = pmap(i -> is_omp_threading(), 1:nworkers())

        if ~all(threading)
            @warn "Using both threading and multiprocessing at the same time is not advised, set OMP_NUM_THREADS=1 when adding a new process to disable this messaage"
        end

    else
        mapfunc = map

    end

    results = Vector(undef, nfolds)
    tstart = time()
    for (fold, (train_inds, test_inds)) in enumerate(folds[1:nfolds])

        X_train, y_train, X_test, y_test = X[train_inds,:], y[train_inds], X[test_inds,:], y[test_inds]

        best_params = tune(
            X_train, 
            y_train, 
            tuning_parameters,
            tuning_optimiser; 
            objective=objective, 
            opts0=tuning_opts0,
            nfolds=n_cvfolds, 
            pms=tuning_pms,
            abstol=tuning_abstol, 
            maxiters=tuning_maxiters,
            verbosity=verbosity,
            rng=tuning_rng,
            foldmethod=tuning_foldmethod
        )
        opts = _set_options(opts0; best_params...)
        verbosity >= 1 && print("fold $fold: t=$(rtime(tstart)): training MPS with $(best_params)... ")
        mps, _... = fitMPS(X_train, y_train, opts);
        println(" done")
        p = verbosity, tstart, fold, nfolds
        results[fold] = Dict("train_inds"=>train_inds, 
                             "test_inds"=>test_inds, 
                             "optimiser"=>string(tuning_optimiser),
                             "pms"=>pms,
                             "opts"=>opts, 
                             "Loss"=>eval_loss(objective, mps, X_test, y_test, pms; p=p)
        )

    end
    return results
end