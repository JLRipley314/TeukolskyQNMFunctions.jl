"""
TeukolskyQNMFunctions.jl computes the quasinormal modes and eigenfunctions 
for the spin s Teukolsky equation
using a horizon penetrating, hyperboloidally compactified coordinate system.
The main advantage of using these coordinates is that the quasinormal
wavefunctions are finite valued from the black hole to future null infinity.
"""
module TeukolskyQNMFunctions

export F, compute_om

include("Spheroidal.jl")
include("RadialODE.jl")

import .Spheroidal as SH
import .RadialODE as RD

"""
    F(
            nr::Integer,
            nl::Integer,
            s::Integer,
            l::Integer,
            m::Integer,
            a::T,
            om::Complex{T}
        ) where T<:Real

Compute the absolute difference of Lambda seperation constant for radial and angular ODEs.
"""
function F(
    nr::Integer,
    nl::Integer,
    s::Integer,
    l::Integer,
    m::Integer,
    a::T,
    om::Complex{T},
) where {T<:Real}

    lmin = max(abs(s), abs(m))
    neig = l - lmin + 1 # number of eigenvalues

    la_s, _ = SH.eig_vals_vecs(nl, neig, s, m, a * om)
    la_r, _, _ = RD.eig_vals_vecs_c(nr, s, m, a, om)

    ## The Lambdas are ordered in size of smallest magnitude
    ## to largest magnitude, we ASSUME this is the same as the
    ## ordering of the l-angular indexing (this may not
    ## hold when a->1; need to check).

    return abs(la_s[l-lmin+1] - la_r)
end

"""
    compute_om(
        nr::Integer,
        nl::Integer,
        s::Integer,
        l::Integer,
        m::Integer,
        a::Real,
        om::Complex;
        tolerance::Real = 1e6,
        epsilon::Real = 1e-6,
        gamma::Real = 1.0,
        verbose::Bool = false
    ) where T<:Real

Search for quasinormal mode frequency in the complex plane using Newton's method.

# Arguments

* `nr`       : number of radial Chebyshev collocation points
* `nl`       : number of spherical harmonic terms
* `s`        : spin of the field in the Teukolsky equation
* `l`        : l angular number
* `m`        : m angular number
* `a`        : black hole spin
* `om`       : guess for the initial quasinormal mode
* `tolerance`: tolerance for root finder
* `epsilon`  : derivative finite difference length
* `gamma`    : search gamma
* `verbose`  : true: print out intermediate results as searches for root 

"""
function compute_om(
    nr::Integer,
    nl::Integer,
    s::Integer,
    l::Integer,
    m::Integer,
    a::T,
    om::Complex{T};
    tolerance::T = 1e6,
    epsilon::T = 1e-6,
    gamma::T = 1.0,
    verbose::Bool = false,
) where {T<:Real}

    om_n = -1000.0 + 0.0im
    om_np1 = om

    f(omega) = F(nr, nl, s, l, m, a, omega)

    ## Newton search with 2nd order finite differences

    newgamma = gamma
    iterations = 1
    while abs(om_np1 - om_n) > tolerance

        ## if too many iterations, reduce search step size
        iterations += 1
        if iterations % 100 == 0
            newgamma /= 2
        end

        om_n = om_np1

        df = (
            (f(om_n + epsilon) - f(om_n - epsilon)) / (2 * epsilon) +
            (f(om_n + im * epsilon) - f(om_n - im * epsilon)) / (2 * im * epsilon)
        )
        om_np1 = om_n - newgamma * f(om_n) / df

        if verbose == true
            println("om_np1=$om_np1\tdf=$df\tdiff=$(abs(om_np1-om_n))")
        end
    end

    lmin = max(abs(s), abs(m))
    neig = l - lmin + 1 # number of eigenvalues

    la_s, v_s = SH.eig_vals_vecs(nl, neig, s, m, a * om_np1)
    la_r, v_r, rvals = RD.eig_vals_vecs_c(nr, s, m, a, om_np1)

    return om_np1, la_r, v_s[:, l-lmin+1], v_r, rvals
end

end # module
