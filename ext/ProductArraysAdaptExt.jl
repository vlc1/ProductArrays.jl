module ProductArraysAdaptExt

using Adapt: Adapt, adapt
using ProductArrays: ProductArray

# through the constructor: an adaptor may change the vectors' eltype, hence T;
# the function too, as it may capture arrays
Adapt.adapt_structure(to, A::ProductArray) = ProductArray(adapt(to, A.fn), adapt(to, A.args)...)

end
