"""
Spin-weighted spherical harmonics and associated functions and operators.
"""
module Sphere

setprecision(2048) # BigFloat precision in bits

import FastGaussQuadrature as FGQ
import Jacobi: jacobi

export Y_vals, cos_vals, sin_vals, swal, swal_laplacian_matrix, swal_filter_matrix, angular_matrix_mult!

function convertBF(v)
   return parse(BigFloat,"$v")
end

"""
number of l angular values given ny theta values

num_l = ny - 2*(max_s+max_m), where max_s and max_m
are hard coded in Sphere.jl

num_l(
   ny::Int64
   )::Int64
"""
const max_s = 2
const max_m = 6

function num_l(ny::Int64)::Int64
   @assert ny>(2*max_s+2*max_m)+4
   return ny-2*max_s-2*max_m
end

"""
Computes Gauss-Legendre points (y) over the interval [-1,1].

Y_vals(
   ny::Int64
   )::Vector{Float64}
"""
function Y_vals(ny::Int64)::Vector{Float64}

   roots, weights= FGQ.gausslegendre(ny) 

   return roots 
end

"""
Computes inner product over the interval [-1,1] using
Gauss quadrature:

return: ∫ dy v1 conj(v2)

inner_product(
   v1::Vector{T},
   v2::Vector{T},
   )::T where T<:Number
"""
function inner_product(
      v1::Array{T,1},
      v2::Array{T,1}
   )::T where T<:Number
   ny = size(v1)[1]

   roots, weights= FGQ.gausslegendre(ny) 

   s = 0.0
   for j=1:ny
      s += weights[j]*v1[j]*conj(v2[j])
   end

   return s 
end

"""
Computes cos(y) at Gauss-Legendre points over interval [-1,1].

cos_vals(
   ny::Int64
   )::Vector{Float64}
"""
function cos_vals(ny::Int64)::Vector{Float64}

   roots, weights= FGQ.gausslegendre(ny) 

   return [-pt for pt in roots]
end

"""
Computes sin(y) at Gauss-Legendre points over interval [-1,1].

sin_vals(
   ny::Int64
   )::Vector{Float64}
"""
function sin_vals(ny::Int64)::Vector{Float64}

   roots, weights= FGQ.gausslegendre(ny) 
   
   return [convert(Float64,
                   sqrt(1.0-convertBF(pt))*sqrt(1.0+convertBF(pt))
                  )
           for pt in roots
          ]
end

"""
Computes the spin-weighted associated Legendre function Y^s_{lm}(y).

swal(
   spin::Int64,
   m_ang::Int64,
   l_ang::Int64,
   y::Float64
   )::Float64
"""
function swal(
      spin::Int64,
      m_ang::Int64,
      l_ang::Int64,
      y::Float64
   )::Float64
   @assert l_ang>=abs(m_ang)

   al = abs(m_ang-spin)
   be = abs(m_ang+spin)
   @assert((al+be)%2==0)
   n = l_ang - (al+be)/2

   if n<0
      return convert(Float64,0)
   end

   norm = sqrt(
      (2*n+al+be+1)*(convertBF(2)^(-al-be-1.0))
   *  factorial(convertBF(n+al+be))/factorial(convertBF(n+al))
   *  factorial(convertBF(n      ))/factorial(convertBF(n+be))
   )
   norm *= convertBF(-1)^(max(m_ang,-spin))

   return convert(
      Float64,
      norm
      *(convertBF(1-y)^(al/2.))
      *(convertBF(1+y)^(be/2.))
      *jacobi(convertBF(y),convertBF(n),convertBF(al),convertBF(be))
   )
end

"""
Computes matrix swal^s_{lm}(y) at Gauss-Legendre points,
over a grid of l values (and fixed m).

swal_vals(
   ny::Int64,
   spin::Int64,
   m_ang::Int64,
   )::Matrix{Float64}
"""
function swal_vals(
      ny::Int64,
      spin::Int64,
      m_ang::Int64,
   )::Matrix{Float64}

   roots, weights= FGQ.gausslegendre(ny) 

   nl = num_l(ny)
   
   vals = zeros(Float64,ny,nl)
   
   lmin = max(abs(spin),abs(m_ang))
   
   for k=1:nl
      l_ang = k-1+lmin
      for j=1:length(roots)
         vals[j,k] = swal(spin,m_ang,l_ang,roots[j])
      end
   end
   return vals
end

"""
Computes matrix to compute spin-weighted spherical harmonic laplacian.
To compute the spherical laplacian use the function 
set_lap!(v_lap,v,lap_matrix)

swal_laplacian_matrix(
   ny::Int64,
   spin::Int64,
   m_ang::Int64
   )::Matrix{Float64}
"""
function swal_laplacian_matrix(
      ny::Int64,
      spin::Int64,
      m_ang::Int64
   )::Matrix{Float64}
   
   rv, wv = FGQ.gausslegendre(ny) 

   nl = num_l(ny) 

   lap = zeros(Float64,ny,ny)

   lmin = max(abs(spin),abs(m_ang))

   for j=1:ny
      for i=1:ny
         for k=1:nl
            l = k-1+lmin
            lap[j,i] -= (l-spin)*(l+spin+1.0)*(swal(spin,m_ang,l,rv[i])*
                                               swal(spin,m_ang,l,rv[j])
                                              )
         end
         lap[j,i] *= wv[j]
      end
   end

   return lap
end

"""
Computes matrix to compute low pass filter.

swal_filter_matrix(
   ny::Int64,
   spin::Int64,
   m_ang::Int64
   )::Matrix{Float64}

Multiply the matrix on the left: v[i]*M[i,j] -> v[j]
"""
function swal_filter_matrix(
      ny::Int64,
      spin::Int64,
      m_ang::Int64
   )::Matrix{Float64}
   
   rv, wv = FGQ.gausslegendre(ny) 

   nl = num_l(ny) 

   filter = zeros(Float64,ny,ny)

   lmin = max(abs(spin),abs(m_ang))

   for j=1:ny
      for i=1:ny
         for l=lmin:(nl-1+lmin)
            filter[j,i] += exp(-30.0*(l/(nl-1.0))^10)*(
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
Matrix multiplication in the angular direction

angular_matrix_mult!(
   f_m,
   f,
   mat
   )
"""
function angular_matrix_mult!(
      f_m,
      f,
      mat
   )
   nx, ny = size(f)
   for j=1:ny
      for i=1:nx
         f_m[i,j] = 0.0
         for k=1:ny
            f_m[i,j] += f[i,k]*mat[k,j]
         end
      end
   end
   return nothing
end

end
