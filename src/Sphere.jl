"""
Spin-weighted spherical harmonics and associated functions and operators.
"""
module Sphere

setprecision(2048) # BigFloat precision in bits

import QuadGK

export Y_vals,
    cos_vals,
    sin_vals,
    swal,
    swal_laplacian_matrix,
    swal_filter_matrix,
    angular_matrix_mult!

function cBF(v::Number)
    return convert(BigFloat, v)
end

"""
    jacobi(x::T, n::Integer, a::T, b::T)::T where {T<:Number}

    The Jacobi polynomial.
"""
function jacobi(x::T, n::Integer, a::T, b::T)::T where {T<:Number}
    ox = one(x)
    zx = zero(x)
    if n == 0
        return ox
    elseif n == 1
        return (ox / 2) * (a - b + (a + b + 2) * x)
    end

    p0 = ox
    p1 = (ox / 2) * (a - b + (a + b + 2) * x)
    p2 = zx

    for i = 1:(n-1)
        a1 = 2 * (i + 1) * (i + a + b + 1) * (2 * i + a + b)
        a2 = (2 * i + a + b + 1) * (a * a - b * b)
        a3 = (2 * i + a + b) * (2 * i + a + b + 1) * (2 * i + a + b + 2)
        a4 = 2 * (i + a) * (i + b) * (2 * i + a + b + 2)
        p2 = (ox / a1) * ((a2 + a3 * x) * p1 - a4 * p0)

        p0 = p1
        p1 = p2
    end

    return p2
end

"""
    num_l(ny::Integer, max_s::Integer=2, max_m::Integer=6)

Number of l angular values given ny theta values

num_l = ny - 2*(max_s+max_m), where max_s and max_m
are hard coded in Sphere.jl
"""
function num_l(ny::Integer, max_s::Integer = 2, max_m::Integer = 6)
    @assert ny > (2 * max_s + 2 * max_m) + 4
    return ny - 2 * max_s - 2 * max_m
end

"""
    inner_product(
       v1::Vector{T},
       v2::Vector{T},
       )::T where T<:Number

Compute the inner product over the interval [-1,1] using
Gauss quadrature:

return: ∫ dy v1 conj(v2)

"""
function inner_product(v1::Vector{T}, v2::Vector{T})::T where {T<:Number}
    ny = size(v1)[1]

    _, weights = QuadGK.gauss(typeof(real(v1[1])), ny)

    s = 0.0
    for j = 1:ny
        s += weights[j] * v1[j] * conj(v2[j])
    end

    return s
end

"""
    Y_vals(ny::Integer, T::Type{<:Real}=Float64)

Compute the Gauss-Legendre points (y) over the interval [-1,1].
"""
function Y_vals(ny::Integer, T::Type{<:Real} = Float64)

    roots, _ = QuadGK.gauss(T, ny)

    return roots
end

"""
    cos_vals(ny::Integer, T::Type{<:Real}=Float64)

Compute cos(y) at Gauss-Legendre points over interval [-1,1].
"""
function cos_vals(ny::Integer, T::Type{<:Real} = Float64)

    yv = Y_vals(ny, T)

    return -yv
end

"""
    sin_vals(ny::Integer, T::Type{<:Real}=Float64)

Compute sin(y) at Gauss-Legendre points over interval [-1,1].
"""
function sin_vals(ny::Integer, T::Type{<:Real} = Float64)

    yv = Y_vals(ny, T)

    return [sqrt(1 - y) * sqrt(1 + y) for y in yv]
end

"""
    swal(
       spin::Integer,
       m_ang::Integer,
       l_ang::Integer,
       y::Real
       )

Compute the spin-weighted associated Legendre function Y^s_{lm}(y).
"""
function swal(spin::Integer, m_ang::Integer, l_ang::Integer, y::Real)
    @assert l_ang >= abs(m_ang)

    TR = typeof(y)

    al = abs(m_ang - spin)
    be = abs(m_ang + spin)
    @assert((al + be) % 2 == 0)
    n = l_ang - (al + be) / 2

    if n < 0
        return convert(TR, 0)
    end

    norm = sqrt(
        (2 * n + al + be + 1) * (2^(-al - be - 1.0)) * factorial(cBF(n + al + be)) /
        factorial(cBF(n + al)) * factorial(cBF(n)) / factorial(cBF(n + be)),
    )
    norm *= (-1)^(max(m_ang, -spin))

    return convert(
        TR,
        norm *
        ((1 - y)^(al / 2)) *
        ((1 + y)^(be / 2)) *
        jacobi(cBF(y), Integer(n), cBF(al), cBF(be)),
    )
end

"""
    swal_vals(ny::Integer, spin::Integer, m_ang::Integer, T::Type{<:Real}=Float64)

Compute the matrix swal^s_{lm}(y) at Gauss-Legendre points,
over a grid of l values (and fixed m).
"""
function swal_vals(ny::Integer, spin::Integer, m_ang::Integer, T::Type{<:Real} = Float64)

    yv = Y_vals(ny, T)

    nl = num_l(ny)

    vals = zeros(T, ny, nl)

    lmin = max(abs(spin), abs(m_ang))

    for k = 1:nl
        l_ang = k - 1 + lmin
        for (j, y) in enumerate(yv)
            vals[j, k] = swal(spin, m_ang, l_ang, y)
        end
    end
    return vals
end

"""
    swal_laplacian_matrix(ny::Integer, spin::Integer, m_ang::Integer, T::Type{<:Real}=Float64)

Compute the matrix to compute spin-weighted spherical harmonic laplacian.
To compute the spherical laplacian use the function 
set_lap!(v_lap,v,lap_matrix)
"""
function swal_laplacian_matrix(
    ny::Integer,
    spin::Integer,
    m_ang::Integer,
    T::Type{<:Real} = Float64,
)

    rv, wv = QuadGK.gauss(T, ny)

    nl = num_l(ny)

    lap = zeros(T, ny, ny)

    lmin = max(abs(spin), abs(m_ang))

    for j = 1:ny
        for i = 1:ny
            for k = 1:nl
                l = k - 1 + lmin
                lap[j, i] -=
                    (l - spin) *
                    (l + spin + 1) *
                    (swal(spin, m_ang, l, rv[i]) * swal(spin, m_ang, l, rv[j]))
            end
            lap[j, i] *= wv[j]
        end
    end

    return lap
end

"""
    swal_filter_matrix(ny::Integer, spin::Integer, m_ang::Integer, T::Type{<:Real}=Float64)

Compute the matrix to compute low pass filter.
Multiply the matrix on the left: v[i]*M[i,j] -> v[j]
"""
function swal_filter_matrix(
    ny::Integer,
    spin::Integer,
    m_ang::Integer,
    T::Type{<:Real} = Float64,
)

    rv, wv = QuadGK.gauss(T, ny)

    nl = num_l(ny)

    filter = zeros(T, ny, ny)

    lmin = max(abs(spin), abs(m_ang))

    for j = 1:ny
        for i = 1:ny
            for l = lmin:(nl-1+lmin)
                filter[j, i] +=
                    exp(-30.0 * (l / (nl - 1.0))^10) *
                    (swal(spin, m_ang, l, rv[i]) * swal(spin, m_ang, l, rv[j]))
            end
            filter[j, i] *= wv[j]
        end
    end

    return filter
end

"""
    angular_matrix_mult!(
       f_m,
       f,
       mat
       )

Compute matrix multiplication in the angular direction.
"""
function angular_matrix_mult!(f_m, f, mat)
    nx, ny = size(f)
    for j = 1:ny
        for i = 1:nx
            f_m[i, j] = 0
            for k = 1:ny
                f_m[i, j] += f[i, k] * mat[k, j]
            end
        end
    end
    return nothing
end

end
