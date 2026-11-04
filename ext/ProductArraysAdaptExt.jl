module ProductArraysAdaptExt

using Adapt: Adapt, adapt
using ProductArrays: ProductArray

# through the constructor: an adaptor may change the vectors' eltype, hence T
Adapt.adapt_structure(to, A::ProductArray) = ProductArray(A.fn, adapt(to, A.args)...)

end
