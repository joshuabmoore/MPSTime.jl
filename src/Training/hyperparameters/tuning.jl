
abstract type TuningLoss end
struct ClassificationLoss <: TuningLoss end
struct ImputationLoss <: TuningLoss end 


function make_objective(
    objective::TuningLoss, 
    opts0::AbstractMPSOptions, 
    fields::Vector{Symbol}, 
    types::Vector{Type},
    X_train::AbstractMatrix, 
    y_train::AbstractVector, 
    X_val::AbstractMatrix, 
    y_val::AbstractVector, 
    pms::Union{Nothing, AbstractVector}
    )

    fieldnames = Tuple(fields)
    cache = Dict{Tuple{types...}, Float64}()


    function tr_objective(optslist::AbstractVector, p)
        verbosity, tstart, fold, nfolds = p
        optslist_safe = Vector{Union{AbstractFloat, Integer}}(undef, length(optslist))
        for (i, field) in enumerate(optslist)
            t = types[i]
            if t <: Integer
                rounded = round(Int,field)
                optslist_safe[i] = rounded
                if verbosity >= 2 && ~isapprox(field, rounded)
                    println("fold $fold: Integer parameter $(fieldnames[i])=$field rounded to $(rounded)!")
                end
            else
                optslist_safe[i] = convert(t, field)
            end
        end
        
        key = tuple(optslist_safe...)
        if haskey(cache, key )
            verbosity >= 1 && println("fold $fold: Cache hit!")
            loss = cache[key]
        else
            hparams = NamedTuple{fieldnames}(Tuple(optslist_safe))
            opts = _set_options(opts0; hparams...)

            
            
            verbosity >= 1 && print("fold $fold: t=$(rtime(tstart)): training MPS with ($hparams)... ")

            mps, _... = fitMPS(X_train, y_train, opts);
            println(" done")

            loss = eval_loss(objective, mps, X_val, y_val, pms; p=p)
            cache[key] = loss
            verbosity >= 1 && println("fold $fold: t=$(rtime(tstart)): Loss $loss")
        end
        return loss
    end

    return tr_objective, cache
end

function tune_fold(
    fold::Integer, 
    parameter_info::Tuple,
    tuning_settings::Tuple,
    X_train::AbstractMatrix, 
    y_train::AbstractVector, 
    X_val::AbstractMatrix, 
    y_val::AbstractVector, 
    tstart::Real
    )
    x0, opts0, lb, ub, is_disc, fields, types = parameter_info
    objective, method, nfolds, pms, abstol, maxiters, verbosity = tuning_settings 

    t = round(time() - tstart; digits=2)
    println("t=$t: Evaluating fold ($fold/$nfolds)")

    tr_objective, cache = make_objective(objective, opts0, fields, types, X_train, y_train, X_val, y_val, pms)
    # tr_objective(x,_p) = sum(x.^2)
    p = (verbosity, tstart, fold, nfolds)


    obj = OptimizationFunction(tr_objective, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(obj, x0, p; int=is_disc, lb=lb, ub=ub)
    return solve(prob, method; abstol=abstol, maxiters=maxiters)

end

"""
    tune(
        X::AbstractMatrix, 
        y::AbstractVector, 
        parameters::NamedTuple,
        method=SAMIN(); # bounded simulated annealing from Optim
        opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1),
        objective::TuningLoss=ImputationLoss(), 
        nfolds::Integer=5,
        rng::Union{Integer, AbstractRNG}=1,
        foldmethod::Union{Function, Vector}=make_stratified_folds, 
        pms::Union{Nothing, AbstractVector}=collect(0.05:0.15:0.95),
        verbosity::Integer=1,
        abstol::Float64=1e-3,
        maxiters::Integer=500
    )

Train a hyperparameter tuned MPS model on X,y with 
# Arguments
...
- `method::TuningMethod`: either GridSearch or an algorithm from Optim (including Pswarm, simulatedAnnealing, grad desc methods if we want to be fancy )
- `objective::TuningLoss`: I'll setup a predetermined list including classification losses ('inaccuracy') or imputation metrics (MAE)
- `distributed::Bool`: I can use all the work I put in to optimise imputation with the distributed library here


"""
function tune(
        X::AbstractMatrix, 
        y::AbstractVector, 
        parameters::NamedTuple,
        method=SAMIN(); # bounded simulated annealing from Optim
        opts0::AbstractMPSOptions=MPSOptions(; verbosity=-5, log_level=-1),
        objective::TuningLoss=ImputationLoss(), 
        nfolds::Integer=5,
        rng::Union{Integer, AbstractRNG}=1,
        foldmethod::Union{Function, Vector}=make_stratified_folds, 
        pms::Union{Nothing, AbstractVector}=collect(0.05:0.15:0.95),
        verbosity::Integer=1,
        abstol::Float64=1e-3,
        maxiters::Integer=500,
        distribute_folds::Bool=false,
        disable_threading::Bool=distribute_folds

    )
    # basic checks    
    if !(length(unique(parameters)) == length(parameters))
       throw(ArgumentError("The 'parameters' argument contains duplicates!")) 
    end

    abs_rng = rng isa Integer ? Xoshiro(rng) : rng

    
    is_disc = Vector{Bool}(undef, length(parameters))
    lb = Vector{Float64}(undef, length(parameters))
    ub = Vector{Float64}(undef, length(parameters))
    x0 = Vector{Float64}(undef, length(parameters))
    types = Vector{Type}(undef, length(parameters))

    


    # setup tuned hyperparameters
    for (i, (key, val)) in enumerate(pairs(parameters))
        startx = getproperty(opts0, key)
        param_type = typeof(startx)
        if !( param_type <: Number)
            throw(ArgumentError("Cannot tune '$key', only numeric types can be hyperoptimised."))
        end
        is_disc[i] = param_type <: Integer

        if !isempty(val)
            lb[i], ub[i] = convert(param_type, val[1]), convert(param_type, val[2])
        else
            lb[i], ub[i] = one(param_type), typemax(param_type)
        end

        if startx < lb[i] || startx > ub[i]
            startx = lb[i]
        end
        x0[i] = startx
        types[i] = param_type

    end

    # not super necessary, but its nice to have the result be independent of the order of the paramters vector
    fields = [keys(parameters)...]
    perm = sortperm(fields)

    for vec in [fields, types, x0, is_disc, lb, ub]
        permute!(vec, perm)
    end


    parameter_info = x0, opts0, lb, ub, is_disc, fields, types
    tuning_settings = objective, method, nfolds, pms, abstol, maxiters, verbosity

    println("Generating Folds")
    folds = foldmethod(X,y, nfolds; rng=abs_rng)
    sols = Vector(undef, nfolds)
    tstart = time()

    mapfunc = distribute_folds ? pmap : map
    if disable_threading 
        GenericLinearAlgebra.LinearAlgebra.BLAS.set_num_threads(1)
        ITensors.Strided.disable_threads()
    end
    sols = mapfunc( (fold) -> 
        tune_fold(
            fold, 
            parameter_info,
            tuning_settings,
            X[folds[fold][1],:], 
            y[folds[fold][1]], 
            X[folds[fold][2],:], 
            y[folds[fold][2]], 
            tstart
        ),
        1:nfolds
    )

    return sols

end

function eval_loss(::ClassificationLoss, mps::TrainedMPS, X_val::AbstractMatrix, y_val::AbstractVector, pms; p=nothing)
    return 1. - mean(classify(mps, X_val) .== y_val) # misclassification rate
end

function eval_loss(::ImputationLoss, mps::TrainedMPS, X_val::AbstractMatrix, y_val::AbstractVector, pms::Union{Nothing, AbstractVector} = collect(0.05:0.15:0.95); p=nothing)
    
    if ~isnothing(p)
        verbosity, tstart, fold, nfolds = p
        logging = verbosity >= 2
    else
        logging = false
    end
    imp = init_imputation_problem(mps, X_val, verbosity=-5);
    numval = size(X_val, 1)
    instance_scores = Vector{Float64}(undef, numval) # score for each instance across all % missing
    for inst in eachindex(instance_scores)
        logging && print("fold $fold: Evaluating instance $inst/$numval...")
        t = time()
        instance_pm_scores_mps = Vector{Float64}(undef, length(pms)) # score for each % missing for a given instance
        for (ipm, pm) in enumerate(pms)
            impute_sites = mar(X_val[inst, :], pm)[2]
            stats = MPS_impute(imp, 0, inst, impute_sites, :median; NN_baseline=false, plot_fits=false)[4]
            instance_pm_scores_mps[ipm] = stats[1][:MAE]
        end
        logging && println("done ($(rtime(t))s)")
        instance_scores[inst] = mean(instance_pm_scores_mps)
    end

    return mean(instance_scores) # mean score across all instances in validation set
end