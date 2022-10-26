module TestSpheroidal

export test_m_symmetry, test_s_symmetry, test_conj_symmetry, compare_to_qnm

include("../src/Spheroidal.jl")
include("ReadQNM.jl")
using .ReadQNM
import .Spheroidal as SH

import Test: @test

const tolerance = 1e-8 ## tolerance we compare to

##============================================================
"""
    test_m_symmetry(
          nl::Integer,
          neig::Integer,
          s::Integer,
          m::Integer,
          a::Real,
          om::Complex,
          T::Type{<:Real}=Float64
       ) 

Test la_{s,l,m,n}(-c) = la_{s,l,-m,n}(c) 
"""
function test_m_symmetry(
      nl::Integer,
      neig::Integer,
      s::Integer,
      m::Integer,
      a::Real,
      om::Complex,
      T::Type{<:Real}=Float64
   ) 
   la_pm, ph_pm = SH.eig_vals_vecs(nl, neig, s,  m, -a*om, T)
   la_nm, ph_nm = SH.eig_vals_vecs(nl, neig, s, -m,  a*om, T)
     
   for (i,_) in enumerate(la_pm)
      @test abs(la_pm[i] - la_nm[i]) < tolerance
   end

end
"""
    test_s_symmetry(
          nl::Integer,
          neig::Integer,
          s::Integer,
          m::Integer,
          a::Real,
          om::Complex,
          T::Type{<:Real}=Float64
       ) 

Test la_{-s,l,m,n}(c) = 2s + la_{s,l,m,n}(c) 
"""
function test_s_symmetry(
      nl::Integer,
      neig::Integer,
      s::Integer,
      m::Integer,
      a::Real,
      om::Complex,
      T::Type{<:Real}=Float64
   ) 
   la_ps, ph_ps = SH.eig_vals_vecs(nl, neig,  s, m, a*om, T)
   la_ns, ph_ns = SH.eig_vals_vecs(nl, neig, -s, m, a*om, T)
     
   for (i,_) in enumerate(la_ps)
      @test abs(la_ps[i] + 2.0*s - la_ns[i]) < tolerance
   end
end
"""
    test_conj_symmetry(
          nl::Integer,
          neig::Integer,
          s::Integer,
          m::Integer,
          a::Real,
          om::Complex,
          T::Type{<:Real}=Float64
       ) 


Test (la_{s,l,m,n}(c))* = 2s + la_{s,l,m,n}(c*) 
"""
function test_conj_symmetry(
      nl::Integer,
      neig::Integer,
      s::Integer,
      m::Integer,
      a::Real,
      om::Complex,
      T::Type{<:Real}=Float64
   ) 
   la,   ph   = SH.eig_vals_vecs(nl, neig, s, m, a*     om , T)
   la_c, ph_c = SH.eig_vals_vecs(nl, neig, s, m, a*conj(om), T)
     
   for (i,_) in enumerate(la)
      @test abs(conj(la[i]) - la_c[i]) < tolerance
   end
end

"""
    compare_to_qnm(
          nl::Integer,
          neig::Integer,
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
      nl::Integer,
      neig::Integer,
      s::Integer,
      n::Integer,
      l::Integer,
      m::Integer,
      avals::Vector{<:Real},
      T::Type{<:Real}=Float64
   )
   
   lmin = SH.compute_l_min(s,m)

   println("Comparison to qnm library: s=$s, n=$n, l=$l, m=$m") 

   for a in avals
      println("testing a=$a")
      om, la = qnm(n,s,m,l,a)
      ls, vs = SH.eig_vals_vecs(nl, neig, s, m, a*om, T)

      @test abs(ls[l-lmin+1]-la)<tolerance
   end
end

end
