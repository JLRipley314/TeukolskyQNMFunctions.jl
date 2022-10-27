module TestCHLaplacian

export interval_laplacian_ch, interval_laplacian_fd, interval_laplacian_chs

const tolerance = 1e-6 ## tolerance we compare to

include("../src/Chebyshev.jl")
import .Chebyshev as CH

using LinearAlgebra: I, eigvals
using SparseArrays

import Test: @test

##============================================================
"""
    interval_laplacian_ch(nx::Integer, neig::Integer, xmin::Real, xmax::Real)

Compare Eigenvalues on an interval using D2 
"""
function interval_laplacian_ch(
    nx::Integer,
    neig::Integer,
    xmin::Real,
    xmax::Real
)

    D1 = CH.mat_D1(xmin, xmax, nx)
    D2 = -D1 * D1

    D2[1, 1] = 1.0
    D2[1, 2:end] .= 0.0

    D2[nx, nx] = 1.0
    D2[nx, 1:nx-1] .= 0.0

    t = eigvals(D2, sortby = abs)

    L = xmax - xmin

    for (n, la) in enumerate(t[3:div(end, 2, RoundNearest)])
        v = (pi * n / L)^2
        @test abs((la - v) / v) < tolerance
    end
end

"""
    interval_laplacian_fd(nx::Integer, neig::Integer, xmin::Real, xmax::Real)

Compare Eigenvalues on an interval using finite difference D2 
"""
function interval_laplacian_fd(
    nx::Integer,
    neig::Integer,
    xmin::Real,
    xmax::Real
)

    D2 = -CH.mat_fd_D2(xmin, xmax, nx)

    D2[1, 1] = 1.0
    D2[1, 2:end] .= 0.0

    D2[end, end] = 1.0
    D2[nx, 1:end-1] .= 0.0

    t = eigvals(Matrix(D2), sortby = abs)

    L = xmax - xmin

    for (n, la) in enumerate(t[3:10])
        v = (pi * n / L)^2
        @test abs((la - v) / v) < 1e-3
    end
end

end
