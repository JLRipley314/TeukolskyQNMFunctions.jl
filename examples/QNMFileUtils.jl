module QNMFileUtils 

include("../src/CustomTypes.jl")
include("../src/Chebyshev.jl")
using  .CustomTypes 
import .Chebyshev as CH
import TeukolskyQNMFunctions as QNM

import HDF5
import PyCall

export generate_data

function save_to_file!(
      fname::String,
      nr::myI,
      nl::myI, 
      s::myI,
      n::myI,
      l::myI,
      m::myI,
      a::myF,
      omega::myC,
      lambda::myC,
      vs::Vector{myC},
      vr::Vector{myC},
      vrc::Vector{myC},
      rs::Vector{myF},
      tolerance::myF
   )
   HDF5.h5open("$fname.h5", "cw") do file
      g = HDF5.create_group(file, 
         "[a=$(convert(Float64,a)),l=$(convert(Int64,l))]"
        )
      g["nr"]           = convert(Int64,nr)
      g["nl"]           = convert(Int64,nl)
      g["omega"]        = convert(ComplexF64,omega)
      g["lambda"]       = convert(ComplexF64,lambda) 
      g["angular_coef"] = [convert(ComplexF64,v) for v in vs]
      g["radial_func"]  = [convert(ComplexF64,v) for v in vr]
      g["radial_coef"]  = [convert(ComplexF64,v) for v in vrc] 
      g["rvals"]        = [convert(Float64,v)    for v in rs]
      g["tolerance"]    = convert(Float64,tolerance)
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
      nr::myI,
      nl::myI, 
      s::myI,
      n::myI,
      l::myI,
      m::myI,
      avals::Vector{myF};
      qnm_tolerance::myF=tomyF(1e-6),
      coef_tolerance::myF=tomyF(1e-6),
      epsilon::myF=tomyF(1e-6)
   )
   qnmlib = PyCall.pyimport("qnm")

   for a in avals
      nrtmp = nr
      nltmp = nl
      while true
         println("nr=$nrtmp\tnl=$nltmp\ta=$a")
         lib = qnmlib.modes_cache(
                  s=convert(Int64,s),
                  l=convert(Int64,l),
                  m=convert(Int64,m),
                  n=convert(Int64,n)
                 )
         om, la, cs = lib(a=convert(Float64,min(a,0.9999999)))
         omega, lambda, vs, vr, rs = QNM.compute_om(
            nrtmp, nltmp, s, l, m, a, tomyC(om),
            tolerance=tomyF(qnm_tolerance),
            epsilon=tomyF(epsilon),
            gamma=tomyF(1.0-a)
         )
         chebcoef = CH.to_cheb(vr)
         if ((abs(chebcoef[end]) < coef_tolerance) 
         &&  (abs(vs[end])       < coef_tolerance)
         )
            println("ω=$omega, Λ=$lambda")
            save_to_file!(
               fname,
               nrtmp, nltmp, 
               s, n, l, m, a,
               omega, lambda,
               vs, vr, chebcoef, rs,
               coef_tolerance
            )
            break 
         end
         if (abs(chebcoef[end]) > coef_tolerance) 
            nrtmp += tomyI(ceil(nrtmp/2))
         end
         if (abs(vs[end]) > coef_tolerance) 
            nltmp += tomyI(ceil(nltmp/2))
         end
      end
   end
   return nothing
end

end
