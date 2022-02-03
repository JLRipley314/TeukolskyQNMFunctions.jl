module TestSphere

export test_swal_inner_product, test_norm_swal_lap

include("../src/CustomTypes.jl")
include("../src/Sphere.jl")
include("Norms.jl")

import Test: @test

using .CustomTypes
import .Norms 
import .Sphere


const tol = 5e-13

"""
Computes inner product between different Y^s_{lm} functions
"""
function test_swal_inner_product(
      ny::myI,
      spin::myI,
      m_ang::myI
   )::Nothing

   nl = Sphere.num_l(ny)

   vals = Sphere.swal_vals(ny,spin,m_ang)

   println("Test: Y^s_{lm} orthogonality:\tny=$ny\tspin=$spin\tm_ang=$m_ang\tl=[0..$nl]")
   for i=1:nl
      for j=i:nl
         val = Sphere.inner_product(vals[:,i],vals[:,j])
         if i==j && i>max(abs(spin),abs(m_ang))
            @test abs(val-1.0) < tol 
         else
            @test abs(val) < tol 
         end
      end
   end
   return nothing
end

"""
Computes the norm of the difference between the numerical and
exact spin-weighted spherical laplacian for a given
spin-weighted spherical harmonic.
"""
function test_norm_swal_lap(
      ny::myI,
      spin::myI,
      m_ang::myI,
      l_ang::myI
   )::Nothing
   
   Yv = Sphere.Y_vals(ny)

   swal = [Sphere.swal(spin,m_ang,l_ang,y) for y in Yv]

   swal_lap_v1 = [-(l_ang-spin)*(l_ang+spin+1.0)*v for v in swal]
   swal_lap_v2 = zeros(myF,ny) 
   
   lap = Sphere.swal_laplacian_matrix(ny,spin,m_ang)
   for j=1:ny
      for k=1:ny
         swal_lap_v2[j] += swal[k]*lap[k,j]
      end
   end

   ## compute integral over interval [-1,1] of difference
   n = Sphere.inner_product(ones(ny),swal_lap_v1 .- swal_lap_v2)

   println("Test: Δ_{S2}Y^s_{lm}:\tny=$ny\tspin=$spin\tm_ang=$m_ang\tl=$l_ang")
   @test abs(n) < tol 

   return nothing
end

end
