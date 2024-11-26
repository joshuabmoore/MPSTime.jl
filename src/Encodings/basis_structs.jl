# timeseries encoding shell
abstract type Encoding end

struct Basis <: Encoding # probably should not be called directly
    name::String
    init::Union{Function}
    encode::Function
    iscomplex::Bool
    istimedependent::Bool
    range::Tuple{Real, Real}
end


function Base.show(io::IO, E::Basis)
    print(io,"Basis($(E.name))")
end

# Splitting up into a time dependent histogram
struct SplitBasis <: Encoding
    name::String
    init::Union{Function}
    splitmethod::Function
    basis::Basis
    encode::Function
    iscomplex::Bool
    istimedependent::Bool
    range::Tuple{Real, Real}
    SplitBasis(s::String, init::Union{Function,Nothing}, spm::Function, basis::Basis, enc::Function, isc::Bool, istd::Bool, isb, range::Tuple{Real, Real}) = begin
        # spname = replace(s, Regex(" "*basis.name*"\$")=>"") # strip the basis name from the end
        # if !(titlecase(spname) in ["Hist Split", "Histogram Split", "Hist Split Balanced", "Histogram Split Balanced", "Uniform Split Balanced", "Uniform Split"])
        #     error("""Unkown split type "$spname", options are ["Hist Split", "Histogram Split", "Hist Split Balanced", "Histogram Split Balanced", "Uniform Split Balanced", "Uniform Split"]""")
        # end

        if basis.iscomplex != isc
            error("The SplitBasis and its auxilliary basis must agree on whether they are complex!")
        end

        if basis.range != range #TODO This is probably not actually necessary, likely could be handled in encode_TS?
            error("The SplitBasis and its auxilliary basis must agree on the normalised timeseries range!")
        end
        new(s, init, spm, basis, enc, isc, istd, isb, range)
    end
end

function Base.show(io::IO, E::SplitBasis)
    print(io,"SplitBasis($(E.name))")
end


##############################################

function stoudenmire()
    sl = "Stoudenmire" 
    enc = angle_encode
    iscomplex=true
    istimedependent=false
    range = (0,1)
    init = nothing

    return Basis(sl, init, enc, iscomplex, istimedependent, range)
end


function fourier(; project=false)
    sl = project ? "Projected Fourier" : "Fourier"
    enc = fourier_encode
    iscomplex=true
    istimedependent=project
    range = (-1,1)
    init = project ? project_fourier : no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, range)
end

function legendre(; norm=false, project=false)
    sl = norm ? "Legendre_Norm" : "Legendre"
    sl = project ? "Projected "* sl : sl
    enc = norm ? legendre_encode : legendre_encode_no_norm
    iscomplex = false
    istimedependent=project
    range = (-1,1)
    init = project ? project_legendre : no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, range)
end

legendre_no_norm(; project=false) = legendre(; norm=false, project) 

function sahand_legendre(istimedependent::Bool=true)
    sl = "Sahand-Legendre" * (istimedependent ? " Time Dependent" : " Time Independent")
    enc = sahand_legendre_encode
    iscomplex = false
    istimedependent=istimedependent
    range = (-1,1)
    init = istimedependent ?  init_sahand_legendre_time_dependent : init_sahand_legendre

    return Basis(sl, init, enc, iscomplex, istimedependent, range)
end

function sahand()
    sl = "Sahand"
    enc = sahand_encode
    iscomplex=true
    istimedependent=false
    range = (0,1)
    init = no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, range)
end

function uniform()
    sl = "Uniform"
    enc = uniform_encode
    iscomplex = false
    istimedependent=false
    range = (0,1)
    init = no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, range)
end

# the error function Basis, raises an error (used as a placeholder only)
function erf()
    f = _ -> error("Tried to use a basis that isn't implemented")
    iscomplex = false # POSIX compliant error function
    istimedependent = false
    range = (-1,1)
    return Basis("Pun Intended", no_init, basis, iscomplex, istimedependent, range)
end


# constructs a time-dependent encoding from a function
# For a time independent basis, the input function must have the signature :
# b(x::Float64, d:Integer, init_args...) and return a d-dimensional Numerical Vector
# A vector [x_1, x_2, x_3, ..., x_N] will be encoded as [b(x_1), b(x_2), b(x_3),..., b(x_N)]
# To use a time dependent basis, set is_time_dependent to true. The input function must have the signature 
# b(x::Float64, d:Integer, ti::Int, init_args...) and return a d-dimensional Numerical Vector
# A vector [x_1, x_2, x_3, ..., x_N] will be encoded as  [b_1(x_1), b_2(x_2), b_3(x_3),..., b_N(x_N)]
function construct_basis(basis::Function, is_complex::Bool, range::Tuple{<:Real,<:Real}, init::Function=no_init; is_time_dependent::Bool=false, name::String="Custom" )
    return Basis(name, init, basis, is_complex, is_time_dependent, range)
end

function hist_split(basis::Basis)
    isc = basis.iscomplex
    range = basis.range

    name = "Hist. Split $(basis.name)" 

    init = hist_split_init
    splitmethod = hist_split
    istd = true
    enc = project_onto_hist_bins

    return SplitBasis(name, init, splitmethod, basis, enc, isc, istd, range)

end

function uniform_split(basis::Basis)
    isc = basis.iscomplex
    range = basis.range

    name = "Unif. Split $(basis.name)" 

    init = unif_split_init
    splitmethod = unif_split
    istd = basis.istimedependent # if the aux. basis _is_ time dependent we have to treat the entire split as such
    enc = project_onto_unif_bins

    return SplitBasis(name, init, splitmethod, basis, enc, isc, istd, range)

end

hist_split() = hist_split(uniform())
uniform_split() = uniform_split(uniform())