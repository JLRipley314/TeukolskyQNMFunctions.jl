module QNMFileUtils

include("../qnmtables/ReadQNM.jl")
include("../src/Cheb.jl")
include("../src/TeukolskyQNMFunctions.jl")
import .Cheb as CH
import .TeukolskyQNMFunctions as QNM
import .ReadQNM

import HDF5

export generate

function save_to_file!(
    fname::String,
    nr::Integer,
    nl::Integer,
    s::Integer,
    n::Integer,
    l::Integer,
    m::Integer,
    a::T,
    omega::Complex{T},
    lambda::Complex{T},
    vs::Vector{<:Complex{T}},
    vr::Vector{<:Complex{T}},
    vrc::Vector{<:Complex{T}},
    rs::Vector{<:T},
    tolerance::T,
) where {T<:Real}
    HDF5.h5open("$fname.h5", "cw") do file
        g = HDF5.create_group(file, "[n=$(convert(Int64,n))]")
        g["nr"] = convert(Int64, nr)
        g["nl"] = convert(Int64, nl)
        g["omega"] = convert(ComplexF64, omega)
        g["lambda"] = convert(ComplexF64, lambda)
        g["angular_coef"] = [convert(ComplexF64, v) for v in vs]
        g["radial_func"] = [convert(ComplexF64, v) for v in vr]
        g["radial_coef"] = [convert(ComplexF64, v) for v in vrc]
        g["rvals"] = [convert(Float64, v) for v in rs]
        g["tolerance"] = convert(Float64, tolerance)
    end
    return nothing
end


"""
   Starts from guess from qnm code:

      L. Stein, 
      J.Open Source Softw. 4 (2019) 42, 1683,
      arXiv:1908.10377
   
   and then searches for closest qnm to that mode
"""
function generate_data(
    fname::String,
    nr::Integer,
    nl::Integer,
    s::Integer,
    n::Integer,
    l::Integer,
    m::Integer,
    a::T;
    qnm_tolerance::T = T(1e-6),
    coef_tolerance::T = T(1e-6),
    epsilon::T = T(1e-6),
) where {T<:Real}
    nrtmp = nr
    nltmp = nl
    while true
        println("Trying nr=$nrtmp\tnl=$nltmp")
        om, la = ReadQNM.qnm(n, s, m, l, a)
       
        omega, lambda, vs, vr, rs = QNM.compute_om(
            nrtmp,
            nltmp,
            s,
            l,
            m,
            a,
            convert(Complex{T},om),
            tolerance = qnm_tolerance,
            epsilon = epsilon,
            gamma = 1.0 - a,
            verbose = false,
        )
        println("read in ω=$om")
        println("read in Λ=$la")
        
        println("cheb coef tolerance=$(abs(vr[end]))\tswal coef tolerance=$(abs(vs[end]))")
        if ((abs(vr[end]) < coef_tolerance) && (abs(vs[end]) < coef_tolerance))
            println("compute Λ=$lambda")
            save_to_file!(
                fname,
                nrtmp,
                nltmp,
                s,
                n,
                l,
                m,
                a,
                omega,
                lambda,
                vs,
                vr,
                vr,
                rs,
                coef_tolerance,
            )
            break
        end
        if (abs(vr[end]) > coef_tolerance)
            println("-------------------------------")
            println("not enough tolerance, adding nr")
            println("-------------------------------")
            nrtmp += Int64(ceil(nrtmp / 2))
        end
        if (abs(vs[end]) > coef_tolerance)
            println("-------------------------------")
            println("not enough tolerance, adding nl")
            println("-------------------------------")
            nltmp += Int64(ceil(nltmp / 2))
        end
    end
    return nothing
end

function generate(
    prename::String,
    nr::Integer,
    nl::Integer,
    s::Integer,
    m::Integer,
    l::Integer,
    n::Integer,
    avals::Vector{<:T},
    qnm_tolerance::T,
    coef_tolerance::T,
    epsilon::T,
) where {T<:Real}
    lmin = max(abs(s), abs(m))
    for a in avals
        try
            println("=======================================")
            println("s=$s,l=$l,m=$m,n=$n,a=$a")
            generate_data(
                "$(pwd())/qnmfiles/$(prename)s$(s)_l$(l)_m$(m)_a$(a)",
                nr,
                nl,
                s,
                n,
                l,
                m,
                a,
                qnm_tolerance = qnm_tolerance,
                coef_tolerance = coef_tolerance,
                epsilon = epsilon,
            )
        catch err
            println(err)
        end
    end
    return nothing
end

end
