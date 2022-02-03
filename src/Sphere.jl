"""
Spin-weighted spherical harmonics and associated functions and operators.
We set y ≡ -cosθ

num_l = ny - 2*(max_s+max_m), where max_s and max_m
are hard coded in Sphere.jl
"""
module Sphere

include("CustomTypes.jl")
using .CustomTypes

import FastGaussQuadrature as FGQ
import Jacobi: jacobi

const max_s = 2
const max_m = 6

function num_l(ny::myI)::myI
   @assert ny>(2*max_s+2*max_m)+4
   return ny-2*max_s-2*max_m
end
"""
    Y_vals(ny::myI)::Vector{myF}

Computes Gauss-Legendre points (y) over the interval [-1,1].
"""
function Y_vals(ny::myI)::Vector{myF}

   roots, weights= FGQ.gausslegendre(ny) 

   return roots 
end
"""
    inner_product(
      v1::Vector{myF},
      v2::Vector{myF}
    )::myF

Computes inner product over the interval [-1,1] using
Gauss quadrature:

return: ∫ dy v1 conj(v2)
"""
function inner_product(
      v1::Vector{myF},
      v2::Vector{myF}
   )::myF
   ny = size(v1)[1]

   roots, weights= FGQ.gausslegendre(ny) 

   s = tomyF(0)
   for j=1:ny
      s += weights[j]*v1[j]*conj(v2[j])
   end

   return s 
end
"""
    cos_vals(ny::myI)::Vector{myF}

Computes cos(y) at Gauss-Legendre points over interval [-1,1].
"""
function cos_vals(ny::myI)::Vector{myF}

   roots, weights= FGQ.gausslegendre(ny) 

   return [-pt for pt in roots]
end
"""
    sin_vals(ny::myI)::Vector{myF}

Computes sin(y) at Gauss-Legendre points over interval [-1,1].
"""
function sin_vals(ny::myI)::Vector{myF}

   roots, weights= FGQ.gausslegendre(ny) 
   
   return [convert(myF,
                   sqrt(tomyF(1)-pt)*sqrt(tomyF(1)+pt)
                  )
           for pt in roots
          ]
end
"""
    swal(
      spin::myI,
      m_ang::myI,
      l_ang::myI,
      y::myF
    )::myF

Computes the spin-weighted associated Legendre function Y^s_{lm}(y).
"""
function swal(
      spin::myI,
      m_ang::myI,
      l_ang::myI,
      y::myF
   )::myF
   
   al = abs(m_ang-spin)
   be = abs(m_ang+spin)
   @assert((al+be)%2==0)
   n = tomyF(l_ang - ((al+be)/tomyF(2)))

   if n<0
      return tomyF(0)
   end

   norm = sqrt(
      (2*n+al+be+1)*(tomyF(2)^(-al-be-1.0))
   *  factorial(tomyF(n+al+be))/factorial(tomyF(n+al))
   *  factorial(tomyF(n      ))/factorial(tomyF(n+be))
   )
   norm *= tomyF(-1)^(max(m_ang,-spin))

   return convert(
      myF,
      norm
      *(tomyF(1-y)^(al/2.))
      *(tomyF(1+y)^(be/2.))
      *jacobi(tomyF(y),tomyF(n),tomyF(al),tomyF(be))
   )
end
"""
    swal_vals(
      ny::myI,
      spin::myI,
      m_ang::myI,
    )::Matrix{myF}

Computes matrix swal^s_{lm}(y) at Gauss-Legendre points,
over a grid of l values (and fixed m).
"""
function swal_vals(
      ny::myI,
      spin::myI,
      m_ang::myI,
   )::Matrix{myF}

   roots, weights= FGQ.gausslegendre(ny) 

   nl = num_l(ny)
   
   vals = zeros(myF,ny,nl)
   
   for k=1:nl
      l_ang = k-1
      for j=1:length(roots)
         vals[j,k] = swal(spin,m_ang,l_ang,roots[j])
      end
   end
   return vals
end
"""
    swal_laplacian_matrix(
      ny::myI,
      spin::myI,
      m_ang::myI
    )::Matrix{myF}

Computes matrix to compute spin-weighted spherical harmonic laplacian.
To compute the spherical laplacian use the function 
set_lap!(v_lap,v,lap_matrix)
"""
function swal_laplacian_matrix(
      ny::myI,
      spin::myI,
      m_ang::myI
   )::Matrix{myF}
   
   rv, wv = FGQ.gausslegendre(ny) 

   nl = num_l(ny) 

   lap = zeros(myF,ny,ny)

   for j=1:ny
      for i=1:ny
         for k=1:nl
            l = k-1
            lap[j,i] -= (l-spin)*(l+spin+tomyF(1))*(swal(spin,m_ang,l,rv[i])*
                                                    swal(spin,m_ang,l,rv[j])
                                                   )
         end
         lap[j,i] *= wv[j]
      end
   end

   return lap
end
"""
    swal_filter_matrix(
      ny::myI,
      spin::myI,
      m_ang::myI
    )::Matrix{myF}

Computes matrix to compute low pass filter
of spin-weighted spherical harmonic laplacian.
To compute the spherical laplacian use the function 
set_lap!(v_lap,v,lap_matrix)
"""
function swal_filter_matrix(
      ny::myI,
      spin::myI,
      m_ang::myI
   )::Matrix{myF}
   
   rv, wv = FGQ.gausslegendre(ny) 

   nl = num_l(ny) 

   filter = zeros(myF,ny,ny)

   for j=1:ny
      for i=1:ny
         for k=1:nl
            l = k-1
            filter[j,i] += exp(-tomyF(30)*(l/(nl-tomyF(1)))^10)*(
                           swal(spin,m_ang,l,rv[i])*
                           swal(spin,m_ang,l,rv[j])
                        )
         end
         filter[j,i] *= wv[j]
      end
   end

   return filter
end
"""
    angular_matrix_mult!(
      f_m,
      f,
      mat
    )

Matrix multiplication in the angular direction
"""
function angular_matrix_mult!(
      f_m,
      f,
      mat
   )
   nx, ny = size(f)
   for j=1:ny
      for i=1:nx
         f_m[i,j] = tomyF(0)
         for k=1:ny
            f_m[i,j] += f[i,k]*mat[k,j]
         end
      end
   end
   return nothing
end

end
