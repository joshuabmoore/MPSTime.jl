using MPSTime
using Test
using TestItems
using Aqua

@testset "Aqua.jl Quality Assurance" begin
    Aqua.test_ambiguities(MPSTime) # test for method ambiguities
    Aqua.test_unbound_args(MPSTime) # test that all methods have bounded type parameters
    Aqua.test_undefined_exports(MPSTime) # test that all exported names exist
    Aqua.test_stale_deps(MPSTime) # test that the package loads all deps listed in the root Project.toml
    Aqua.test_piracies(MPSTime) # test that the package does not commit type piracies
end

@testset "Bases" begin 
    include("basis_tests.jl")
end

@testset "Analysis" begin
    include("analysis_tests.jl")
end

@testset "Imputation Data Utils" begin
    include("simulation_tests.jl")
end
