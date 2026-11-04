"""
    ProductArray(fn, x, y, …) -> AbstractArray{T, N}

The lazy, read-only array of `fn(x[i], y[j], …)` over the Cartesian product of
the vectors `x, y, …`, one per axis: element `(i, j, …)` is computed when read,
and nothing is stored but `fn` and the vectors. Its axes are the vectors' axes;
its eltype `T` is the return type of `fn` on the vectors' eltypes.

A type in place of `fn` constructs the elements: `ProductArray(SVector, x, y)`
holds the points themselves, `SVector(x[i], y[j])`, in column-major order. The
array is `isbits` whenever `fn` and the vectors are (a function without
captured state, a type, a range), so that it passes to a GPU kernel by value.
"""
struct ProductArray{T, N, F, A<:NTuple{N, AbstractVector}} <: AbstractArray{T, N}
    fn::F
    args::A
end

function ProductArray(fn::F, args::Vararg{AbstractVector, N}) where {F, N}
    T = Base.promote_op(fn, map(eltype, args)...)
    T === Union{} && _throw_no_eltype(fn, args)
    return ProductArray{T, N, F, typeof(args)}(fn, args)
end
# a type as a field is a DataType reference, not isbits: a singleton calls it
ProductArray(::Type{S}, args::Vararg{AbstractVector, N}) where {S, N} =
    ProductArray(_Construct{S}(), args...)

@noinline _throw_no_eltype(fn, args) = throw(ArgumentError(
    "ProductArray: $fn cannot return for arguments of types $(map(eltype, args))"))

struct _Construct{S} end
(::_Construct{S})(args...) where {S} = S(args...)

Base.axes(A::ProductArray) = map(a -> axes(a, 1), A.args)
Base.size(A::ProductArray) = map(length, A.args)

Base.@propagate_inbounds function Base.getindex(A::ProductArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    return A.fn(@inbounds(_coords(A.args, I))...)
end

# the coordinates of point I, by tuple recursion: `map` does not pass @inbounds
# on to the reads
_coords(::Tuple{}, ::Tuple{}) = ()
Base.@propagate_inbounds _coords(xs::Tuple, is::Tuple) =
    (first(xs)[first(is)], _coords(Base.tail(xs), Base.tail(is))...)
