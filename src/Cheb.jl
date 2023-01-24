"""
Methods for position space Chebyshev methods. 
"""
module Cheb

export chep_pts, mat_X, mat_D1, to_cheb, to_real

using SparseArrays

"""
    cheb_pts([T,] nx::Integer)

Computes Generates nx Chebyshev points in [-1,1]
"""
function cheb_pts(::Type{T}, nx::Integer) where {T<:AbstractFloat}
    return [cos(pi * i / (nx - T(1))) for i = 0:(nx-1)]
end

"""
    cheb_pts(xmin::Real, xmax::Real, nx::Integer)

Computes Chebyshev points on interval [xmin,xmax] 
"""
function cheb_pts(xmin::T, xmax::T, nx::Integer) where {T<:Real}
    pts = cheb_pts(T, nx)
    m = (xmax - xmin) / 2
    b = (xmax + xmin) / 2
    return [m * pts[i] + b for i = 1:nx]
end

"""
    mat_X(
          xmin::Real,
          xmax::Real,
          nx::Integer
       )

Computes matrix for multiplication of x in real space. 
"""
function mat_X(xmin::TR, xmax::TR, nx::TI) where {TR<:AbstractFloat,TI<:Integer}
    @assert nx > 4

    X = Vector{TI}(undef, 0)
    Y = Vector{TI}(undef, 0)
    V = Vector{TR}(undef, 0)

    pts = cheb_pts(xmin, xmax, nx)

    for i = 1:nx
        push!(X, i)
        push!(Y, i)
        push!(V, pts[i])
    end

    return sparse(X, Y, V)
end

"""
    mat_D1(
          xmin::Real,
          xmax::Real,
          nx::Integer,
       )

Computes derivative matrix D1 in real space.
"""
function mat_D1(xmin::TR, xmax::TR, nx::TI) where {TR<:AbstractFloat,TI<:Integer}
    @assert nx > 4

    M = Matrix{TR}(undef, nx, nx)
    n = nx - 1
    pts = cheb_pts(TR, nx)

    M[1, 1] = (2 * (n^2) + 1) // 6
    M[nx, nx] = -M[1, 1]

    M[1, nx] = (1 // 2) * ((-1)^n)
    M[nx, 1] = -M[1, nx]

    for i = 2:(nx-1)
        M[1, i] = (2) * ((-1)^(i + 1) / (1 - pts[i]))
        M[nx, i] = (-2) * ((-1)^(i + nx) / (1 + pts[i]))
        M[i, 1] = (-1 // 2) * ((-1)^(i + 1) / (1 - pts[i]))
        M[i, nx] = (1 // 2) * ((-1)^(i + nx) / (1 + pts[i]))

        for j = 2:(nx-1)
            if i != j
                M[i, j] = (((-1))^(i + j)) / (pts[i] - pts[j])
            end
        end
    end

    for i = 1:(nx-1)
        M[i, i] = 0.0
        for j = 1:nx
            if i != j
                M[i, i] -= M[i, j]
            end
        end
    end

    M .*= 2 / (xmax - xmin)

    return M
end

"""
    to_cheb(f::Vector{<:Number})

Convert to Chebyshev space. We assume we are working with Chebyshev-Gauss-Lobatto points.
"""
function to_cheb(f::Vector{T}) where {T<:Number}

    N = length(f) - 1
    c = zeros(T, N + 1)

    for i = 1:(N+1)
        n = i - 1

        c[i] += f[1] / N
        c[i] += ((-1)^n) * f[end] / N

        for k = 2:N
            c[i] += (2.0 / N) * f[k] * cos(n * (k - 1) * pi / N)
        end
    end
    c[1] /= 2 ## from normalization of inner product 

    return c
end

"""
    to_real(c::Vector{<:Number})

Convert to Real space. We assume we are working with Chebyshev-Gauss-Lobatto points.
"""
function to_real(c::Vector{T}) where {T<:Number}

    N = length(c) - 1
    f = zeros(T, N + 1)

    for i = 1:(N+1)
        n = i - 1
        for j = 1:(N+1)
            f[j] += c[i] * (cos(n * (j - 1) * pi / N))
        end
    end
    return f
end

end
