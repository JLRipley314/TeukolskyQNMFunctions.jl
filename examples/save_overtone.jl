using ApproxFun
using SpecialFunctions

using LinearAlgebra
using SparseArrays

include("../src/Chebyshev.jl")
import .Chebyshev as CH

import HDF5

include("../src/SpectralRadialODE.jl")
import .SpectralRadialODE as RODE
using GenericLinearAlgebra # for svd of bigfloat matrix

# input params

nr = 160;
s = -2;
l = 2;
m = 2;
nmin = 8;
nmax = 12;
T=BigFloat
a = T(0.0);
bhm = T(1);

rmin = T(0); ## location of future null infinity (1/r = ∞)
rmax = abs(a) > 0 ? (bhm / (a^2)) * (1 - sqrt(1 - ((a / bhm)^2))) : 0.5 / bhm;


function save_to_file!(
    prename::String,
    n::Integer,
    l::Integer,
    m::Integer,
    a::T,
    #omega::ComplexF64,
    #lambda::ComplexF64,
    vr::Vector{<:Complex{T}},
    rs::Vector{<:T},
    amp::Vector{<:T},
    damp::Vector{<:T},
) where {T<:Real}
    fname = "$(pwd())/qnmfiles/$(prename)a$(a)_l$(l)_m$(m)_n$(n)"
    HDF5.h5open("$fname.h5", "cw") do file
        g = HDF5.create_group(file, "[a=$(convert(Float64,a)),l=$(convert(Int64,l))]")
        g["nr"] = convert(Int64, length(rs))
        # g["omega"] = convert(ComplexF64, omega)
        # g["lambda"] = convert(ComplexF64, lambda)
        g["radial_func"] = [convert(ComplexF64, v) for v in vr]
        g["rvals"] = [convert(Float64, v) for v in rs]
        g["amp"] = [convert(Float64, v) for v in amp]
        g["damp"] = [convert(Float64, v) for v in damp]
    end
    return nothing
end;

for n=0:12
    M = RODE.radial_operator(nr,s,l,m,n,a,bhm,rmin,rmax);
    println(n)
    rs = CH.cheb_pts(rmin,rmax,nr);
    null = nullspace(Matrix(M),rtol=1e-10);
    vect = CH.to_real(null[:,end]);
    amp = [abs(v/vect[1]) for v in vect];
    D = CH.mat_D1(rmin,rmax,nr);
    damp = D*amp;
    save_to_file!("",n,l,m,a,vect,rs,amp,damp)
end
