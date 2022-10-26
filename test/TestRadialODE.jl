module TestRadialODE

export compare_to_qnm

const tolerance = 1e-3 ## tolerance we compare to

include("../src/RadialODE.jl")
include("ReadQNM.jl")
using .ReadQNM
import .RadialODE as RO
import Test: @test

##============================================================

"""
    compare_to_qnm(
          nr::Integer,
          s::Integer,
          n::Integer,
          l::Integer,
          m::Integer,
          avals::Vector{<:Real},
          T::Type{<:Real}=Float64
       )

Compare against values computed by Leo Stein's qnm code,
J.Open Source Softw. 4 (2019) 42, 1683
"""
function compare_to_qnm(
      nr::Integer,
      s::Integer,
      n::Integer,
      l::Integer,
      m::Integer,
      avals::Vector{<:Real},
      T::Type{<:Real}=Float64
   )
   
   println("Comparison to qnm library: s=$s, n=$n, l=$l, m=$m") 

   for a in avals
      om, la = qnm(n,s,m,l,a)

      ls_c, vs_c = RO.eig_vals_vecs_c(nr, s, m, a, om, T)

      #println("testing: a=$a, ω=$om\nΛ=$la, Cheb Λ=$(ls_c[1])")

      @test abs(ls_c - la)/max(1,abs(la))<tolerance
   end
end

end
