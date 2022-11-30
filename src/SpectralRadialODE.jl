"""
Ordinary differential equation in radial direction for the
hyperboloidally compactified Teukolsky equation.
"""
module SpectralRadialODE

export radial_operator

using ApproxFun
using SpecialFunctions

using LinearAlgebra
using SparseArrays
using PyCall
qnm = pyimport("qnm")

"""
    function radial_operator(
        nr::Integer,
        s::Integer,
        l::Integer,
        m::Integer,
        n::Integer,
        a::T,
        bhm::T,
        rmin::T,
        rmax::T
    ) where T<:Real
"""
function radial_operator(
    nr::Integer,
    s::Integer,
    l::Integer,
    m::Integer,
    n::Integer,
    a::T,
    bhm::T,
    rmin::T,
    rmax::T
) where {T<:Real}
    mode_seq = qnm.modes_cache(s=s,l=l,m=m,n=n);
    om, lambda, C = mode_seq(a=a);

    d = rmin..rmax;
    D1 = Derivative(d);
    D2 = D1^2;
    x = Fun(identity,d);

    A = (
        (2 * im) * om - 2 * (1 + s) * x +
        2 * (im * om * ((a^2) - 8 * (bhm^2)) + im * m * a + (s + 3) * bhm) * x^2 +
        4 * (2 * im * om * bhm - 1) * (a^2) * x^3
    )
    B = (
        (((a^2) - 16 * (bhm^2)) * (om^2) + 2 * (m * a + 2 * im * s * bhm) * om) +
        2 *
        (
            4 * ((a^2) - 4 * (bhm^2)) * bhm * (om^2) +
            (4 * m * a * bhm - 4 * im * (s + 2) * (bhm^2) + im * (a^2)) * om +
            im * m * a +
            (s + 1) * bhm
        ) *
        x +
        2 * (8 * (bhm^2) * (om^2) + 6 * im * bhm * om - 1) * (a^2) * x^2
    )
    return (-(x^2 - 2 * bhm * x^3 + (a^2) * x^4) * D2 + A * D1 + (B + lambda))[1:nr,1:nr]
end

end #module
