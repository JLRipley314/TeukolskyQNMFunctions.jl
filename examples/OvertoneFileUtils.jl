module OvertoneFileUtils

include("../qnmtables/ReadQNM.jl")
include("../src/Chebyshev.jl")
include("../src/TeukolskyQNMFunctions.jl")
include("../src/RadialODE.jl")

import .Chebyshev as CH
import .TeukolskyQNMFunctions as QNM
import .ReadQNM
import .RadialODE as RD
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
    vr::Vector{<:Complex{T}},
    vrc::Vector{<:Complex{T}},
    rs::Vector{<:T},
) where {T<:Real}
    HDF5.h5open("$fname.h5", "cw") do file
        g = HDF5.create_group(file, "[a=$(convert(Float64,a)),l=$(convert(Int64,l))]")
        g["nr"] = convert(Int64, nr)
        g["nl"] = convert(Int64, nl)
        g["omega"] = convert(ComplexF64, omega)
        g["lambda"] = convert(ComplexF64, lambda)
        g["radial_func"] = [convert(ComplexF64, v) for v in vr]
        g["radial_coef"] = [convert(ComplexF64, v) for v in vrc]
        g["rvals"] = [convert(Float64, v) for v in rs]
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
    avals::Vector{<:T};
    qnm_tolerance::T = T(1e-6),
    coef_tolerance::T = T(1e-6),
    epsilon::T = T(1e-6),
) where {T<:Real}
    for a in avals
        nrtmp = nr
        nltmp = nl
        while true
            println("nr=$nrtmp\tnl=$nltmp\ta=$a")
            om, la = ReadQNM.qnm(n, s, m, l, a)
           
            lambda, vr, rs = RD.eig_vals_vecs_c(
                nrtmp,
                s,
                m,
                a,
                convert(Complex{T},om),
            )
            chebcoef = CH.to_cheb(vr)
            if ((abs(chebcoef[end]) < coef_tolerance))
                println("ω=$om, Λ=$lambda")
                save_to_file!(
                    fname,
                    nrtmp,
                    nltmp,
                    s,
                    n,
                    l,
                    m,
                    a,
                    om,
                    lambda,
                    vr,
                    chebcoef,
                    rs,
                )
                break
            end
            if (abs(chebcoef[end]) > coef_tolerance)
                nrtmp += Int64(ceil(nrtmp / 2))
            end
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
    try
        println("s=$s,m=$m,n=$n,l=$l")
        generate_data(
            "$(pwd())/qnmfiles/$(prename)s$(s)_m$(m)_n$(n)_nr$(nr)",
            nr,
            nl,
            s,
            n,
            l,
            m,
            avals,
            qnm_tolerance = qnm_tolerance,
            coef_tolerance = coef_tolerance,
            epsilon = epsilon,
        )
    catch err
        println(err)
    end
    return nothing
end

end
