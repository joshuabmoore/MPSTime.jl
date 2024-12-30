using MPSTime
using Test
using TestItems


@testset "Bases" begin 
    include("basis_tests.jl")
end

@testset "Analysis" begin
    include("analysis_tests.jl")
end

@testset "Imputation Data Utils" begin
    include("data_utils_tests.jl")
end

# @testset "QuantumInspiredML.jl" begin
#     # Write your tests here.
# end
