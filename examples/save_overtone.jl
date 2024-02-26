using ApproxFun
using SpecialFunctions

using LinearAlgebra
using SparseArrays

using PyCall
@pyimport qnm

include("../src/Chebyshev.jl")
import .Chebyshev as CH

import HDF5

include("../src/SpectralRadialODE.jl")
import .SpectralRadialODE as RODE
using GenericLinearAlgebra # for svd of bigfloat matrix

# input params

nr = 160;
nr_interp = 300;
s = -2;
l = 2;
m = -2;
nmin = 0;
nmax = 7;
T=BigFloat
a = T(0);
bhm = T(1);

rmin = T(0); ## location of future null infinity (1/r = ∞)
rmax = abs(a) > 0 ? (bhm / (a^2)) * (1 - sqrt(1 - ((a / bhm)^2))) : 0.5 / bhm;


function save_to_file!(
    prename::String,
    n::Integer,
    l::Integer,
    m::Integer,
    a::T,
    omega::ComplexF64,
    lambda::ComplexF64,
    C::Vector{ComplexF64},
    radial_coef::Vector{<:Complex{T}},
    vr::Vector{<:Complex{T}},
    rs::Vector{<:T},
    amp::Vector{<:T},
    damp::Vector{<:T},
) where {T<:Real}
    fname = "$(pwd())/qnmfiles/$(prename)a$(Float32(a))_l$(l)_m$(m)"
    HDF5.h5open("$fname.h5", "cw") do file
        g = HDF5.create_group(file, "[n=$(convert(Int32,n))]")
        g["nr"] = convert(Int64, length(rs))
        g["omega"] = [convert(ComplexF64, omega)]
        g["lambda"] = [convert(ComplexF64, lambda)]
        g["angular_coef"] = C
        g["radial_coef"] = [convert(ComplexF64, v) for v in radial_coef]
        g["radial_func"] = [convert(ComplexF64, v) for v in vr]
        g["rvals"] = [convert(Float64, v) for v in rs]
        g["amp"] = [convert(Float64, v) for v in amp]
        g["damp"] = [convert(Float64, v) for v in damp]
    end
    return nothing
end;

for n=nmin:nmax
    mode_seq = qnm.modes_cache(s=s,l=l,m=m,n=n);
    omega, lambda, C = mode_seq(a=Float64(a));
    M = RODE.radial_operator(nr,s,l,m,n,a,bhm,rmin,rmax);
    rs = CH.cheb_pts(rmin,rmax,nr_interp);
    radial_coef = nullspace(Matrix(M),rtol=1e-10)[:,end];
    null_interp = zeros(Complex{T},nr_interp);
    null_interp[1:nr] = radial_coef;
    vect = CH.to_real(null_interp);
    amp = [abs(v/vect[end]) for v in vect];
    println(n)
    D = CH.mat_D1(rmin,rmax,nr_interp);
    damp = D*amp;
    save_to_file!("",n,l,m,a,omega,lambda,C,radial_coef,vect,rs,amp,damp)
end;
