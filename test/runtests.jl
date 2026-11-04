using ProductArrays, Test
using Aqua, JET, StaticArrays

# bytes allocated by f(args...) after a warm-up call, behind a function barrier;
# the minimum of 3 calls, since a stray allocation can land in one
function allocs(f::F, args::Vararg{Any, N}) where {F, N}
    f(args...)
    return minimum(_ -> @allocated(f(args...)), 1:3)
end

@testset "ProductArrays" begin
    @testset "Aqua" begin
        Aqua.test_all(ProductArrays)
    end

    @testset "ProductArray" begin
        x, y = range(0.0, 1.0; length = 5), [0.0, 0.5, 2.0]
        f(a, b) = a + 10b
        A = @inferred ProductArray(f, x, y)
        @test A isa AbstractMatrix{Float64}
        @test axes(A) == (Base.OneTo(5), Base.OneTo(3)) && size(A) == (5, 3)
        @test A == [f(a, b) for a in x, b in y]
        @test A[2, 3] == f(x[2], y[3]) == A[CartesianIndex(2, 3)] && A[7] == A[2, 2]
        @test_throws BoundsError A[6, 1]
        @test_throws ArgumentError ProductArray((a, b) -> error(), x, y)
        # the tuple form is not supported
        @test_throws MethodError ProductArray(f, (x, y))
        # a type constructs the elements: the points, column-major
        P = @inferred ProductArray(SVector, x, y)
        @test eltype(P) === SVector{2, Float64}
        @test P[2, 3] === SVector(x[2], y[3])
        @test vec(P)[1:2] == [SVector(x[1], y[1]), SVector(x[2], y[1])]
        @test ProductArray(Float32, x) == Float32.(x)
        # by value into a kernel: isbits with a function or a type, and ranges
        @test isbits(ProductArray(+, x, x)) && isbits(ProductArray(SVector, x, x))
        @test allocs(getindex, A, 2, 3) == 0 && allocs(getindex, P, 2, 3) == 0
        @test allocs(ProductArray, SVector, x, y) == 0
        @test allocs(collect, P) == allocs(a -> Array{eltype(a)}(undef, size(a)), P)
        @test_opt getindex(P, 2, 3)
        @test_opt ProductArray(SVector, x, y)
        # tuple code loses inference past 10 elements
        xs = ntuple(Returns(1:2), Val(13))
        B = @inferred ProductArray(tuple, xs...)
        @test @inferred(B[ntuple(Returns(2), Val(13))...]) === ntuple(Returns(2), Val(13))
    end
end
