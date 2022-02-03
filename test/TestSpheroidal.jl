module TestSpheroidal

export test_m_symmetry, test_s_symmetry, test_conj_symmetry, compare_to_qnm

include("../src/CustomTypes.jl")
include("ReadQNM.jl")
using .CustomTypes
using .ReadQNM

import Test: @test

const tolerance = 1e-8 ## tolerance we compare to

##============================================================
import Spheroidal as SH

"""
Test la_{s,l,m,n}(-c) = la_{s,l,-m,n}(c) 
"""
function test_m_symmetry(
      nl::myI,
      neig::myI,
      s::myI,
      m::myI,
      a::myF,
      om::myC
   ) 
   la_pm, ph_pm = SH.eig_vals_vecs(nl, neig, s,  m, -a*om)
   la_nm, ph_nm = SH.eig_vals_vecs(nl, neig, s, -m,  a*om)
     
   for (i,_) in enumerate(la_pm)
      @test abs(la_pm[i] - la_nm[i]) < tolerance
   end

end
"""
Test la_{-s,l,m,n}(c) = 2s + la_{s,l,m,n}(c) 
"""
function test_s_symmetry(
      nl::myI,
      neig::myI,
      s::myI,
      m::myI,
      a::myF,
      om::myC
   ) 
   la_ps, ph_ps = SH.eig_vals_vecs(nl, neig,  s, m, a*om)
   la_ns, ph_ns = SH.eig_vals_vecs(nl, neig, -s, m, a*om)
     
   for (i,_) in enumerate(la_ps)
      @test abs(la_ps[i] + 2.0*s - la_ns[i]) < tolerance
   end
end
"""
Test (la_{s,l,m,n}(c))* = 2s + la_{s,l,m,n}(c*) 
"""
function test_conj_symmetry(
      nl::myI,
      neig::myI,
      s::myI,
      m::myI,
      a::myF,
      om::myC
   ) 
   la,   ph   = SH.eig_vals_vecs(nl, neig, s, m, a*     om )
   la_c, ph_c = SH.eig_vals_vecs(nl, neig, s, m, a*conj(om))
     
   for (i,_) in enumerate(la)
      @test abs(conj(la[i]) - la_c[i]) < tolerance
   end
end

"""
Compare against values computed by Leo Stein's qnm code,

J.Open Source Softw. 4 (2019) 42, 1683

This is written in python so we need to bind to the library
"""
function compare_to_qnm(
      nl::myI,
      neig::myI,
      s::myI,
      n::myI,
      l::myI,
      m::myI,
      avals::Vector{myF}
   )
   
   lmin = SH.compute_l_min(s,m)

   println("Comparison to qnm library: s=$s, n=$n, l=$l, m=$m") 

   for a in avals
      println("testing a=$a")
      om, la = qnm(n,s,m,l,a)
      ls, vs = SH.eig_vals_vecs(nl, neig, s, m, a*om)

      @test abs(ls[l-lmin+1]-la)<tolerance
   end
end

end
