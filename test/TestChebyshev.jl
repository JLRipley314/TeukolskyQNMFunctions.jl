module TestChebyshev

export test_convergence,
    test_convergence_fd, test_X_matrices, test_to_cheb, test_to_cheb_to_real

include("../src/Chebyshev.jl")

import Test: @test

const tolerance = 1e-8 ## tolerance we compare to

##============================================================
import .Chebyshev as CH

"""
One norm 
"""
function one_norm(v)
    norm = 0.0
    n = length(v)
    for val in v
        norm += abs(val)
    end
    return norm / n
end

"""
    evaluate_difference(
        nx::Integer,
        xmin::Real,
        xmax::Real,
        f::Function,
        d1f::Function,
        d2f::Function,
        T::Type{<:Real}=Float64
    )

Evaluate derivative compared to analytic value.
"""
function evaluate_difference(
    nx::Integer,
    xmin::Real,
    xmax::Real,
    f::Function,
    d1f::Function,
    d2f::Function,
    T::Type{<:Real}=Float64
)
    xvals = CH.cheb_pts(xmin, xmax, nx, T)

    D1 = CH.mat_D1(xmin, xmax, nx, T)
    D2 = D1 * D1

    fv = [f(x) for x in xvals]
    d1fv = [d1f(x) for x in xvals]
    d2fv = [d2f(x) for x in xvals]

    compare_d1fv = D1 * fv
    compare_d2fv = D2 * fv

    norm_d1 = one_norm(compare_d1fv .- d1fv)
    norm_d2 = one_norm(compare_d2fv .- d2fv)

    return norm_d1, norm_d2
end

"""
    evaluate_difference_fd(
        nx::Integer,
        xmin::Real,
        xmax::Real,
        f::Function,
        d1f::Function,
        d2f::Function,
        T::Type{<:Real}=Float64
    )

Evaluate finite difference derivative compared to analytic value.
"""
function evaluate_difference_fd(
    nx::Integer,
    xmin::Real,
    xmax::Real,
    f::Function,
    d1f::Function,
    d2f::Function,
    T::Type{<:Real}=Float64
)
    xvals = CH.cheb_pts(xmin, xmax, nx, T)

    D1 = CH.mat_fd_D1(xmin, xmax, nx, T)
    D2 = CH.mat_fd_D2(xmin, xmax, nx, T)

    fv = [f(x) for x in xvals]
    d1fv = [d1f(x) for x in xvals]
    d2fv = [d2f(x) for x in xvals]

    compare_d1fv = D1 * fv
    compare_d2fv = D2 * fv

    norm_d1 = one_norm(compare_d1fv .- d1fv)
    norm_d2 = one_norm(compare_d2fv .- d2fv)

    return norm_d1, norm_d2
end

"""
    test_convergence(
        nx::Integer,
        xmin::Real,
        xmax::Real,
        f::Function,
        d1f::Function,
        d2f::Function,
        T::Type{<:Real}=Float64
    )

Evaluate convergence of the derivative matrices.
"""
function test_convergence(
    nx::Integer,
    xmin::Real,
    xmax::Real,
    f::Function,
    d1f::Function,
    d2f::Function,
    T::Type{<:Real}=Float64
)
    
    norm_d1_1, norm_d2_1 = evaluate_difference(nx, xmin, xmax, f, d1f, d2f, T)
    norm_d1_2, norm_d2_2 =
        evaluate_difference(convert(Int64, ceil(1.1 * nx)), xmin, xmax, f, d1f, d2f, T)

    @test norm_d1_1 / norm_d1_2 > 5.0
    @test norm_d2_1 / norm_d2_2 > 5.0

    return nothing
end

"""
    test_convergence_fd(
        nx::Integer,
        xmin::Real,
        xmax::Real,
        f::Function,
        d1f::Function,
        d2f::Function,
        T::Type{<:Real}=Float64
    )

Evaluate convergence of the finite difference derivative matrices.
"""
function test_convergence_fd(
    nx::Integer,
    xmin::Real,
    xmax::Real,
    f::Function,
    d1f::Function,
    d2f::Function,
    T::Type{<:Real}=Float64
)

    norm_d1_1, norm_d2_1 = evaluate_difference_fd(nx, xmin, xmax, f, d1f, d2f, T)
    norm_d1_2, norm_d2_2 =
        evaluate_difference_fd(convert(Int64, ceil(2 * nx)), xmin, xmax, f, d1f, d2f, T)

    @test norm_d1_1 / norm_d1_2 > 3.5
    @test norm_d2_1 / norm_d2_2 > 3.5

    return nothing
end

"""
    test_X_matrices(nx::Integer, xmin::Real, xmax::Real, T::Type{<:Real}=Float64)

Evaluate derivative on the X matrices
"""
function test_X_matrices(nx::Integer, xmin::Real, xmax::Real, T::Type{<:Real}=Float64)

    i_1 = ones(T, nx)
    D_1 = CH.mat_D1(xmin, xmax, nx, T)
    X_1 = CH.mat_X(xmin, xmax, nx, T)

    i_2 = ones(T, convert(Int64, ceil(1.1 * nx)))
    D_2 = CH.mat_D1(xmin, xmax, convert(Int64, ceil(1.1 * nx)))
    X_2 = CH.mat_X(xmin, xmax, convert(Int64, ceil(1.1 * nx)))

    norm_1 = one_norm(D_1 * sin.(X_1 * i_1) .- cos.(X_1 * i_1))
    norm_2 = one_norm(D_2 * sin.(X_2 * i_2) .- cos.(X_2 * i_2))

    @test norm_1 / norm_2 > 2.0

    return nothing
end

"""
    test_to_cheb(nx::Integer, T::Type{<:Real}=Float64)

Test can recover Chebyshev coefficients from real space data. 
"""
function test_to_cheb(nx::Integer, T::Type{<:Real}=Float64)

    i_1 = ones(T, nx)
    X_1 = CH.mat_X(-1.0, 1.0, nx, T)

    for n = 0:(nx-5)
        values = cos.(n * acos.(X_1 * i_1)) ## nth Chebyshev polynomial
        c_1 = CH.to_cheb(values, T) ## coefficients, only nth should be nonzero
        for i = 1:nx
            if i == n + 1
                @test abs(c_1[i] - 1.0) < 1e-14
            else
                @test abs(c_1[i]) < 1e-14
            end
        end
    end
    return nothing
end

"""
    test_to_cheb_to_real(nx::Integer, xmin::Real, xmax::Real, T::Type{<:Real}=Float64)

Test can go to and from Chebyshev space correctly. 
"""
function test_to_cheb_to_real(nx::Integer, xmin::Real, xmax::Real, T::Type{<:Real}=Float64)

    nx_1 = nx
    nx_2 = convert(Int64, ceil(1.1 * nx))

    i_1 = ones(T, nx_1)
    X_1 = CH.mat_X(xmin, xmax, nx_1, T)
    xv = X_1 * i_1
    values = tanh.(xv)
    c_1 = CH.to_cheb(values, T)
    r_1 = CH.to_real(c_1, T)
    n_1 = one_norm(values .- r_1)

    i_2 = ones(T, nx_2)
    X_2 = CH.mat_X(xmin, xmax, nx_2, T)
    xv = X_2 * i_2
    values = tanh.(xv)
    c_2 = CH.to_cheb(values, T)
    r_2 = CH.to_real(c_2, T)

    n_2 = one_norm(values .- r_2)

    @test n_2 < n_1

    return nothing
end

end
