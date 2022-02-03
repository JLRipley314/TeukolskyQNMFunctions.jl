module TestRadialODE

export compare_to_qnm

const tolerance = 1e-3 ## tolerance we compare to

include("../src/CustomTypes.jl")
include("ReadQNM.jl")
using .CustomTypes
using .ReadQNM
import Test: @test

##============================================================
import RadialODE as RO

"""
Compare against values computed by Leo Stein's qnm code,

J.Open Source Softw. 4 (2019) 42, 1683

This is written in python so we need to bind to the library
"""
function compare_to_qnm(
      nr::myI,
      s::myI,
      n::myI,
      l::myI,
      m::myI,
      avals::Vector{myF}
   )
   
   println("Comparison to qnm library: s=$s, n=$n, l=$l, m=$m") 

   for a in avals
      om, la = qnm(n,s,m,l,a)

      println("testing: a=$a, ω=$om, Λ=$la")
      ls, vs = RO.eig_vals_vecs(nr, s, m, a, tomyC(om))

      println("testing: a=$a, ω=$om, Λ=$la, numerical Λ=$(ls[1])")

      @test abs(ls[1]-la)/max(1,abs(la))<tolerance
   end
end

end
