"""
Ordinary differential equation in radial direction for the
hyperboloidally compactified Teukolsky equation.
"""
module RadialODE

export radial_discretized_eqn_c

include("Chebyshev.jl")
import .Chebyshev as CH

include("Gegenbauer.jl")
import .Gegenbauer as GE
#import IterativeSolvers: invpowm
using LinearAlgebra
using GenericSchur
using SparseArrays

"""
    function radial_discretized_eqn_c(
        nr::Integer,
        s::Integer,
        m::Integer,
        a::T,
        bhm::T,
        om::Complex{T},
        rmin::T,
        rmax::T
    ) where T<:Real
"""
function radial_discretized_eqn_c(
    nr::Integer,
    s::Integer,
    m::Integer,
    a::T,
    bhm::T,
    om::Complex{T},
    gamma::Complex{T},
    rmin::T,
    rmax::T,
) where {T<:Real}

    ### conversion from Chebyshev to ultraspherical
    S1 = GE.compute_S(T,nr,1);
    S0 = GE.compute_S(T,nr,0);

    ### derivative operator that returns C2 ultraspherical
    D1 = S1*GE.compute_D(T,nr,2,rmin,rmax)
    D2 = GE.compute_D(T,nr,1,rmin,rmax)

    Id = sparse(I, nr, nr)

    ### multiplication matrices (by rho) in C2 spectral space
    X1 = GE.compute_M(T,nr,2,rmin,rmax)
    #CH.mat_X(rmin, rmax, nr)
    X2 = X1 * X1
    X3 = X1 * X2
    X4 = X1 * X3

    ### coefficients in real space
    A = (
        (2 * im) * om * Id - 2 * (1 + s) * X1 +
        2 * (im * om * ((a^2) - 8 * (bhm^2)) + im * m * a + (s + 3) * bhm) * X2 +
        4 * (2 * im * om * bhm - 1) * (a^2) * X3
    )
    B = (
        (((a^2) - 16 * (bhm^2)) * (om^2) + 2 * (m * a + 2 * im * s * bhm) * om) * Id +
        2 *
        (
            4 * ((a^2) - 4 * (bhm^2)) * bhm * (om^2) +
            (4 * m * a * bhm - 4 * im * (s + 2) * (bhm^2) + im * (a^2)) * om +
            im * m * a +
            (s + 1) * bhm
        ) *
        X1 +
        2 * (8 * (bhm^2) * (om^2) + 6 * im * bhm * om - 1) * (a^2) * X2
    )
    ### convert coefficients to spectral space
    ### -(X2 - 2 * bhm * X3 + (a^2) * X4) in C2
    ### A in C2
    ### B in T but convert to C2

    return (-(X2 - 2 * bhm * X3 + (a^2) * X4) * D2 + A * D1 + (B - gamma*Id) * S1 * S0)
end
end # module
