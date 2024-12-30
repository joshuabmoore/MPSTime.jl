
# generate some random data for the tests
test_data = rand(5, 10_000);

@testset "mcar" begin 
    # test the missing completely at random mechanism
    pm = 0.3 # 30% missing
    outX, outInd = mcar(test_data[1, :], pm)
    @test isa(outX, Vector)
    @test !isnothing(outInd)
    @test isa(outInd, Vector{Int})
    # check percentage missing expectation
    ntrials = 10_000
    nmissing = zeros(Int64, ntrials)
    for trial in 1:ntrials
        outX, outInd = mcar(test_data[1, :], pm)
        nmissing[trial] = length(outInd)
    end
    # check for more than 1% difference
    @test isapprox(mean(nmissing), (pm * size(test_data, 2)); rtol=0.01)
    # check for reproducibility/randomness
    outX1, outInd1 = mcar(test_data[1, :], pm; state=42)
    outX2, outInd2 = mcar(test_data[1, :], pm; state=42)
    @test isequal(outX1, outX2)
    @test isequal(outInd1, outInd2)
    @test !isequal(outX1, outX)
    @test !isequal(outInd1, outInd)

    # invalid fraction missing
    @test_throws ArgumentError mcar(test_data[1, :], 1.1)
    @test_throws ArgumentError mcar(test_data[1, :], -0.1)
end

@testset "mar" begin
    pm = 0.3 # 30% missing
    outX, outInd = mar(test_data[1, :], pm)
    @test !isnothing(outX)
    @test isa(outX, Vector)
    @test !isnothing(outInd)
    @test isa(outInd, Vector{Int})
    
    # check percentage missing
    @test isequal(length(outInd), (pm * size(test_data, 2)))
    
    # check for reproducibility/randomness
    outX1, outInd1 = mar(test_data[1, :], pm; state=42)
    outX2, outInd2 = mar(test_data[1, :], pm; state=42)
    @test isequal(outX1, outX2)
    @test isequal(outInd1, outInd2)
    @test !isequal(outX1, outX)
    @test !isequal(outInd1, outInd)

    # invalid fraction missing
    @test_throws ArgumentError mar(test_data[1, :], 1.1)
    @test_throws ArgumentError mar(test_data[1, :], -0.1)
    
end

@testset "mnar" begin 
    pm = 0.3 # 30% missing
    outX_lowest, outInd_lowest = mnar(test_data[1, :], pm) # default is lowest
    outX_highest, outInd_highest = mnar(test_data[1, :], pm, MPSTime.HighestMNAR()) # default is lowest
    @test !isnothing(outX_lowest)
    @test !isnothing(outX_highest)
    @test isa(outX_lowest, Vector)
    @test isa(outX_highest, Vector)
    @test !isnothing(outInd_lowest)
    @test !isnothing(outInd_highest)
    @test isa(outInd_lowest, Vector{Int})
    @test isa(outInd_highest, Vector{Int})

    # check percentage missing
    @test isequal(length(outInd_lowest), (pm * size(test_data, 2)))
    @test isequal(length(outInd_highest), (pm * size(test_data, 2)))

    # invalid fraction missing
    @test_throws ArgumentError mnar(test_data[1, :], 1.1)
    @test_throws ArgumentError mnar(test_data[1, :], -0.1)

end