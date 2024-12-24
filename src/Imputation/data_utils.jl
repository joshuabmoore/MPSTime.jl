
remove_values(X::AbstractVector{Float64}, idxs::Vector{Int64}) = (Xc = deepcopy(X); Xc[idxs] .= NaN; Xc)

function block_missing(X::AbstractVector{Float64}, fraction_missing::Float64, 
        block_length::Int)
    
    # checks
    if percentage_missing > 1
        throw(ArgumentError("fraction missing cannot be greater than 1."))
    end
    num_samples = length(X)
    num_blocks = (fraction_missing * num_samples) / block_length
    start_idxs = collect(1:(num_samples-block_length)+1)
    for b in num_blocks

    end

end




