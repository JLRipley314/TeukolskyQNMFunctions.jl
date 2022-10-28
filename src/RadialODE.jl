"""
Ordinary differential equation in radial direction for the
hyperboloidally compactified Teukolsky equation.
"""
module RadialODE

export eig_vals_vecs_c

include("Chebyshev.jl")
import .Chebyshev as CH

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
    rmin::T,
    rmax::T,
) where {T<:Real}

    D1 = CH.mat_D1(rmin, rmax, nr)
    D2 = D1 * D1

    Id = sparse(I, nr, nr)

    X1 = CH.mat_X(rmin, rmax, nr)
    X2 = X1 * X1
    X3 = X1 * X2
    X4 = X1 * X3

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

    return (-(X2 - 2 * bhm * X3 + (a^2) * X4) * D2 + A * D1 + B)
end

"""
    eig_vals_vecs_c(
            nr::Integer,
            s::Integer,
            m::Integer,
            a::T,
            om::Complex{T}
        ) where T<:Real

Compute eigenvectors and eigenvalues for the radial equation
using a pseudospectral Chebyshev polynomial method.
The black hole mass is always one.
"""
function eig_vals_vecs_c(
    nr::Integer,
    s::Integer,
    m::Integer,
    a::T,
    om::Complex{T},
) where {T<:Real}

    TR = typeof(a)

    bhm = TR(1) ## always have unit black hole mass
    rmin = TR(0) ## location of future null infinity (1/r = ∞)
    rmax = abs(a) > 0 ? (bhm / (a^2)) * (1 - sqrt(1 - ((a / bhm)^2))) : 0.5 / bhm

    Mat = radial_discretized_eqn_c(nr, s, m, a, bhm, om, rmin, rmax)
    t = eigen(Matrix(Mat), permute = true, scale = true, sortby = abs)

    return -t.values[1], t.vectors[:, 1], CH.cheb_pts(rmin, rmax, nr)
end

end # module
