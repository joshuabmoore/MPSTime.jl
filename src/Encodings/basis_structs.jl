# timeseries encoding shell
abstract type Encoding end

struct Basis <: Encoding 
    name::String
    init::Function
    encode::Function
    iscomplex::Bool
    istimedependent::Bool
    isdatadriven::Bool
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
    aux_enc::Encoding
    encode::Function
    iscomplex::Bool
    istimedependent::Bool
    isdatadriven::Bool
    range::Tuple{Real, Real}
    SplitBasis(s::String, init::Function, spm::Function, aux_enc::Encoding, encode_func::Function, isc::Bool, istd::Bool, isdd::Bool, range::Tuple{<:Real, <:Real}) = begin

        if aux_enc.iscomplex != isc
            error("The SplitBasis and its auxilliary basis must agree on whether they are complex!")
        end

        if aux_enc.range != range #TODO This is probably not actually necessary, likely could be handled in encode_TS?
            error("The SplitBasis and its auxilliary basis must agree on the normalised timeseries range!")
        end
        if aux_enc.isdatadriven || aux_enc.istimedependent
            error("Splitting up a data-driven encoding is not yet supported, sorry")
        end

        isdd |= aux_enc.isdatadriven

        new(s, init, spm, aux_enc, encode_func, isc, istd, isdd, range)
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
    isdatadriven = false
    range = (0,1)
    init = no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end


function fourier(; project=false)
    sl = project ? "Projected Fourier" : "Fourier"
    enc = fourier_encode
    iscomplex=true
    istimedependent=project
    isdatadriven = project
    range = (-1,1)
    init = project ? project_fourier : no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end

function legendre(; norm=false, project=false)
    sl = norm ? "Legendre_Norm" : "Legendre"
    sl = project ? "Projected "* sl : sl
    enc = norm ? legendre_encode : legendre_encode_no_norm
    iscomplex = false
    istimedependent=project
    isdatadriven = project
    range = (-1,1)
    init = project ? project_legendre : no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end

legendre_no_norm(; project=false) = legendre(; norm=false, project) 

function sahand_legendre(istimedependent::Bool=true)
    sl = "Sahand-Legendre" * (istimedependent ? " Time Dependent" : " Time Independent")
    enc = sahand_legendre_encode
    iscomplex = false
    istimedependent=istimedependent
    isdatadriven=true
    range = (-1,1)
    init = istimedependent ?  init_sahand_legendre_time_dependent : init_sahand_legendre

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end

function sahand()
    sl = "Sahand"
    enc = sahand_encode
    iscomplex=true
    istimedependent=false
    isdatadriven=false
    range = (0,1)
    init = no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end

function uniform()
    sl = "Uniform"
    enc = uniform_encode
    iscomplex = false
    istimedependent=false
    isdatadriven=false
    range = (0,1)
    init = no_init

    return Basis(sl, init, enc, iscomplex, istimedependent, isdatadriven, range)
end

# the error function Basis, raises an error (used as a placeholder only)
function erf()
    f = _ -> error("Tried to use a basis that isn't implemented")
    iscomplex = false # POSIX compliant error function
    istimedependent = false
    isdatadriven = false
    range = (-1,1)
    return Basis("Pun Intended", no_init, f, iscomplex, istimedependent, isdatadriven, range)
end


# constructs a time-dependent encoding from a function
# For a time independent basis, the input function must have the signature :
# b(x::Float64, d::Integer, init_args...) and return a d-dimensional Numerical Vector
# A vector [x_1, x_2, x_3, ..., x_N] will be encoded as [b(x_1), b(x_2), b(x_3),..., b(x_N)]
# To use a time dependent basis, set is_time_dependent to true. The input function must have the signature 
# b(x::Float64, d::Integer, ti::Int, init_args...) and return a d-dimensional Numerical Vector
# A vector [x_1, x_2, x_3, ..., x_N] will be encoded as  [b_1(x_1), b_2(x_2), b_3(x_3),..., b_N(x_N)]
function function_basis(
    basis::Function, 
    is_complex::Bool, 
    range::Tuple{<:Real,<:Real}, 
    is_time_dependent::Bool=false, 
    is_data_driven::Bool=false,
    init::Function=no_init; 
    name::String="Custom" )
    return Basis(name, init, basis, is_complex, is_time_dependent, is_data_driven, range)
end


function histogram_split(aux_enc::Encoding)
    isc = aux_enc.iscomplex
    range = aux_enc.range

    name = "Hist Split $(aux_enc.name)" 

    init = split_init
    splitmethod = hist_split
    istd = true
    isdatadriven = true
    encode_func = project_onto_bins

    return SplitBasis(name, init, splitmethod, aux_enc, encode_func, isc, istd, isdatadriven, range)
end

function uniform_split(aux_enc::Encoding)
    isc = aux_enc.iscomplex
    range = aux_enc.range

    name = "Unif Split $(aux_enc.name)" 

    init = split_init
    splitmethod = unif_split
    istd = aux_enc.istimedependent # if the aux.enc _is_ time dependent we have to treat the entire split as such
    isdatadriven = aux_enc.isdatadriven
    encode_func = project_onto_bins

    return SplitBasis(name, init, splitmethod, aux_enc, encode_func, isc, istd, isdatadriven, range)

end

histogram_split(s::Symbol) = symbolic_encoding(histogram_split(model_encoding(s)))
uniform_split(s::Symbol) = symbolic_encoding(uniform_split(model_encoding(s)))


histogram_split() = histogram_split(uniform())
uniform_split() = uniform_split(uniform())