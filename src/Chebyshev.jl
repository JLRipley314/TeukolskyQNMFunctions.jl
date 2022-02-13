"""
Methods for position space Chebyshev methods. 
"""
module Chebyshev 

export chep_pts, mat_X, mat_D1, to_cheb, to_real

include("CustomTypes.jl")

using .CustomTypes
using SparseArrays

"""
    cheb_pts(nx::myI)::Vector{myF}

Computes Generates nx Chebyshev points in [-1,1]
"""
function cheb_pts(nx::myI)::Vector{myF}
   return [cos(pi*i/(nx-tomyF(1))) for i=0:(nx-1)]
end

"""
    cheb_pts(xmin::myF, xmax::myF, nx::myI)::Vector{myF}

Computes Chebyshev points on interval [xmin,xmax] 
"""
function cheb_pts(xmin::myF, xmax::myF, nx::myI)::Vector{myF}
   pts = cheb_pts(nx)
   m = (xmax - xmin)/tomyF(2)
   b = (xmax + xmin)/tomyF(2) 
   return [m*pts[i] + b for i=1:nx] 
end

"""
    mat_X(
      xmin::myF,
      xmax::myF,
      nx::myI
    )::SparseMatrixCSC{myF, myI}

Computes matrix for multiplication of x in real space. 
"""
function mat_X(
      xmin::myF,
      xmax::myF,
      nx::myI
   )::SparseMatrixCSC{myF, myI}
   @assert nx>4

   X = Vector{myI}(undef,0)
   Y = Vector{myI}(undef,0)
   V = Vector{myF}(undef,0)

   pts = cheb_pts(xmin, xmax, nx)
   
   for i=1:nx
      push!(X,i)
      push!(Y,i)
      push!(V,pts[i])
   end
   
   return sparse(X,Y,V)
end

"""
    mat_D1(
      xmin::myF,
      xmax::myF,
      nx::myI
    )::Matrix{myF}

Computes derivative matrix D1 in real space.
"""
function mat_D1(
      xmin::myF,
      xmax::myF,
      nx::myI
   )::Matrix{myF}
   @assert nx>4

   M = Matrix{myF}(undef,nx,nx)
   n = nx-1
   pts = cheb_pts(nx)
   
   M[1,1]   = (tomyF(2)*(n^2) + tomyF(1)) / tomyF(6)
   M[nx,nx] = -M[1,1]

   M[1,nx] = tomyF(0.5)*((-1.0)^n) 
   M[nx,1] = -M[1,nx] 

   for i=2:(nx-1)
      M[1 ,i] = tomyF(2.0) *((-1)^(i+1 )/(tomyF(1) - pts[i]))
      M[nx,i] = tomyF(-2.0)*((-1)^(i+nx)/(tomyF(1) + pts[i])) 
      M[i, 1] = tomyF(-0.5)*((-1)^(i+1 )/(tomyF(1) - pts[i]))
      M[i,nx] = tomyF(0.5) *((-1)^(i+nx)/(tomyF(1) + pts[i])) 

      M[i,i] = tomyF(-0.5)*pts[i]/((tomyF(1) + pts[i])*(tomyF(1) - pts[i]))

      for j=2:(nx-1)
         if i!=j
            M[i,j] = ((tomyF(-1))^(i+j))/(pts[i] - pts[j])
         end
      end
   end

   M .*= tomyF(2)/(xmax-xmin)

   return M 
end

"""
    to_cheb(f::Vector{T})::Vector{T} where T<:Number

Convert to Chebyshev space.
We assume we are working with Chebyshev-Gauss-Lobatto points.
"""
function to_cheb(f::Vector{T})::Vector{T} where T<:Number
  
   N = length(f) - 1
   c = zeros(T,N+1) 

   for i=1:(N+1)
      n = i-1

      c[i] += f[1]/tomyF(N)
      c[i] += ((-1)^n)*f[end]/tomyF(N)

      for k=2:N
         c[i] += (2.0/N)*f[k]*cos(n*(k-1)*pi/tomyF(N))
      end
   end
   c[1] /= tomyF(2) ## from normalization of inner product 

   return c
end

"""
    to_real(c::Vector{T})::Vector{T} where T<:Number 

Convert to Real space.
We assume we are working with Chebyshev-Gauss-Lobatto points.
"""
function to_real(c::Vector{T})::Vector{T} where T<:Number 
  
   N = length(c) - 1
   f = zeros(T,N+1) 

   for i=1:(N+1)
      n = i-1
      for j=1:(N+1)
         f[j] += c[i]*(cos(n*(j-1)*pi/tomyF(N))) 
      end
   end 
   return f
end

end
