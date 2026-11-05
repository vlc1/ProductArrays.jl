# ProductArrays

Lazy, read-only arrays over the Cartesian product of vectors:
`ProductArray(f, x, y)[i, j]` is `f(x[i], y[j])`, computed when read. Only `f`
and the vectors are stored. The package has no dependencies.

## Usage

A type in place of `f` constructs the elements, so `SVector` gives the nodes of
a rectilinear grid:

```julia
using ProductArrays, StaticArrays

x = range(0.0, 1.0; length = 5)
y = range(0.0, 2.0; length = 3)

X = ProductArray(SVector, x, y)
X[2, 3] === SVector(x[2], y[3])     # true
eltype(X) === SVector{2, Float64}   # true
size(X) == (5, 3)                   # true

u(a, b) = sin(π * a) * b
U = ProductArray(u, x, y)
U[2, 3] == u(x[2], y[3])            # true
```

The constructor takes one vector per axis. Splat coordinates held in a tuple:

```julia
using ProductArrays

xs = (1:2, 1:3, 1:4)
A = ProductArray(+, xs...)
A[2, 3, 4] == 9                     # true
```

The arguments must be vectors. To include a higher-dimensional array, take the
product over indices and capture the arrays in `f`, inside a function so that
the eltype is inferred:

```julia
using ProductArrays, StaticArrays

function extrude(x, y)
    f(i, j, k) = SVector(x[i], y[j, k])
    return ProductArray(f, axes(x, 1), axes(y, 1), axes(y, 2))
end

x, y = rand(10), rand(20, 30)
A = extrude(x, y)
A[2, 3, 4] === SVector(x[2], y[3, 4])   # true
eltype(A) === SVector{2, Float64}       # true
size(A) == (10, 20, 30)                 # true
```

Such an array is not `isbits`, and `Base.mightalias` does not see the captured
arrays, so do not write it into one of them.

## Behavior

- **Axes and eltype.** The axes are the vectors' axes, and the eltype is the
  return type of `f` on the vectors' eltypes. `f` that cannot return for those
  eltypes throws `ArgumentError`.
- **Devices.** The array is `isbits` when `f` and the vectors are (a function
  without captured state, a type, a range), so it passes to a GPU kernel by
  value. For vectors held in device memory, `using Adapt` loads an extension:
  `adapt(to, A)` adapts `f` and the vectors and recomputes the eltype. Write to
  a device array with `d .= A`: `copyto!(d, A)` falls back on scalar indexing.
  On a device without `Float64`, such as Metal, `isbits` is not enough, as Base's
  ranges may compute in `Float64`. `mtl(A)` converts `Float32` `range`s and
  `Float64` vectors, but not `LinRange` or `Float64` ranges.
  `StepRangeLen(0f0, h, n)` needs no conversion.
- **Aliasing.** The array reads its vectors, so writing to one changes the
  array. `Base.mightalias` sees this, and operations that unalias, such as
  `copyto!`, stay correct when the destination is one of the vectors. Ranges
  hold no memory.

## With StencilAlgebra

A `ProductArray` binds to a [StencilAlgebra](https://github.com/vlc1/StencilAlgebra.jl)
field of the same eltype. Bindings must be indexed from 1, so the vectors must be
too:

```julia
using ProductArrays, StaticArrays, StencilAlgebra

x, y = 0.0:0.25:1.0, 0.0:0.5:2.0
X = ProductArray(SVector, x, y)

@field C SVector{2, Float64}
A = materialize(Unsplit(C), (C = X,), ((1:5, 1:5),))
A[2, 3] === SVector(x[2], y[3])     # true
```

## Installation

ProductArrays is registered in the private `VLC` registry, not in General.
General holds an unrelated package of the same name, so name this one by UUID:

```
pkg> registry add git@github.com:vlc1/VLCRegistry.git
pkg> add ProductArrays=47983261-ebc5-4293-86fb-c7fd2acc601e
```

## Changes from 0.1

Version 0.2 is a rewrite, and it breaks existing code:

- `f` is called as `f(x[i], y[j])`, not `f((x[i], y[j]))`. For tuples, use
  `tuple`, not `identity`.
- `ProductArray(f, (x, y))` becomes `ProductArray(f, x, y)`.
- `product`, `arguments`, `diag`, `CartesianPlane` and the `CartesianIndices`
  shortcuts are gone, along with the OnlyArrays, ZippedArrays and LinearAlgebra
  dependencies.
- Julia 1.12 or later is required.

## Development

The tests use StencilAlgebra, so they need access to the `VLC` registry:

```
pkg> test
```
