"""
Methods to compute spin-weighted spheroidal harmonics
"""
module Spheroidal

using SparseArrays

import LinearAlgebra: eigen
using GenericSchur

"""
    compute_l_min(s::Integer, m::Integer)

Compute the minimum l value
"""
@inline function compute_l_min(s::Integer, m::Integer)
    return max(abs(s), abs(m))
end

"""
    compute_A(s::Integer, l::Integer, m::Integer)

See Appendix A of 
J. Ripley, Class.Quant.Grav. 39 (2022) 14, 145009, Class.Quant.Grav. 39 (2022) 145009
arXiv:2202.03837
"""
@inline function compute_A(s::Integer, l::Integer, m::Integer)
    return sqrt(
        (l + 1 - s) *
        (l + 1 + s) *
        (l + 1 + m) *
        (l + 1 - m) // (((l + 1)^2) * (2 * l + 1) * (2 * l + 3)),
    )
end
"""
    compute_B(s::Integer, l::Integer, m::Integer)

See Appendix A of 
J. Ripley, Class.Quant.Grav. 39 (2022) 14, 145009, Class.Quant.Grav. 39 (2022) 145009
arXiv:2202.03837
"""
@inline function compute_B(s::Integer, l::Integer, m::Integer)
    if (l == 0)
        return 0
    else
        return -(m * s) // (l * (l + 1))
    end
end
"""
    compute_C(s::Integer, l::Integer, m::Integer)

See Appendix A of 
J. Ripley, Class.Quant.Grav. 39 (2022) 14, 145009, Class.Quant.Grav. 39 (2022) 145009
arXiv:2202.03837
"""
@inline function compute_C(s::Integer, l::Integer, m::Integer)
    if (l == 0)
        return 0
    else
        return sqrt(
            (l - s) * (l + s) * (l + m) * (l - m) // ((l^2) * (2 * l + 1) * (2 * l - 1)),
        )
    end
end
"""
    mat_Y(nl::Integer, s::Integer, m::Integer)

Compute the matrix for y in spectral space.
"""
function mat_Y(nl::Integer, s::Integer, m::Integer)

    TI = typeof(nl)

    I = Vector{TI}(undef, 0)
    J = Vector{TI}(undef, 0)
    V = Vector{Rational{TI}}(undef, 0)

    lmin = compute_l_min(s, m)

    for i = 1:nl

        l = (i - 1) + lmin

        ## (l,l) component 
        append!(I, i)
        append!(J, i)
        append!(V, compute_B(s, l, m))

        ## (l,l-1) component 
        if (i > 1)
            append!(I, i)
            append!(J, i - 1)
            append!(V, compute_C(s, l, m))
        end
        ## (l,l+1) component 
        if (i + 1 <= nl)
            append!(I, i)
            append!(J, i + 1)
            append!(V, compute_A(s, l, m))
        end
    end
    return sparse(I, J, V)
end

"""
    function mat_L(nl::Integer, s::Integer, m::Integer)

Compute the matrix for spherical laplacian (s=0) in spectral space.
"""
function mat_L(nl::Integer, s::Integer, m::Integer)
    TI = typeof(nl)
    
    I = Vector{TI}(undef, 0)
    J = Vector{TI}(undef, 0)
    V = Vector{TI}(undef, 0)

    lmin = compute_l_min(s, m)

    for i = 1:nl

        l = (i - 1) + lmin

        ## (l,l) component 
        append!(I, i)
        append!(J, i)
        append!(V, (l - s) * (l + s + 1))
    end
    return sparse(I, J, V)
end
"""
    compute_M_matrix(nl::Integer, s::Integer, m::Integer, c::Complex)

Compute the matrix for computing spheroidal-spherical mixing and seperation coefficients.
"""
function compute_M_matrix(
    nl::Integer,
    s::Integer,
    m::Integer,
    c::Complex
)
    Y1 = Matrix{typeof(c)}(mat_Y(nl + 2, s, m)) # look at larger to capture last term in matrix mult
    # at higher l
    Y2 = Y1 * Y1
    Y1 = Y1[1:nl, 1:nl]
    Y2 = Y2[1:nl, 1:nl]
    L = mat_L(nl, s, m)

    return sparse(transpose(-(c^2) * Y2 + 2 * c * s * Y1 + L))

end

"""
    eig_vals_vecs(nl::Integer, neig::Integer, s::Integer, m::Integer, c::Complex)

Compute the eigenvectors and eigenvalues for the spheroidal equation.  
Returns eigenvalues λs (smallest to largest), and eigenvectors as
an array (access nth eigenvector through v[:,n]); i.e. returns
(λs, vs)
"""
function eig_vals_vecs(
    nl::Integer,
    neig::Integer,
    s::Integer,
    m::Integer,
    c::Complex
)

    Mat = compute_M_matrix(nl, s, m, c)

    for i = 1:nl
        Mat[i, i] += 1 # shift to avoid issues with a zero eigenvalue (shift back at the end)
    end
    # λs, vs
    t = eigen(Matrix(Mat), permute = true, scale = true, sortby = abs)
    return t.values .- 1, t.vectors
end

end # module
