"""
Methods for position space Chebyshev methods. 
"""
module Chebyshev 

export chep_pts, mat_X, mat_D1, mat_D2, mat_fd_D1, mat_fd_D2, to_cheb, to_real

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

      for j=2:(nx-1)
         if i!=j
            M[i,j] = ((tomyF(-1))^(i+j))/(pts[i] - pts[j])
         end
      end
   end

   for i=1:(nx-1)
      M[i,i] = 0.0 
      for j=1:nx
         if i!=j
            M[i,i] -= M[i,j]
         end
      end
   end

   M .*= tomyF(2)/(xmax-xmin)

   return M 
end

"""
    mat_D2(
      xmin::myF,
      xmax::myF,
      nx::myI
    )::Matrix{myF}

Computes derivative matrix D1 in real space.
"""
function mat_D2(
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

      for j=2:(nx-1)
         if i!=j
            M[i,j] = ((tomyF(-1))^(i+j))/(pts[i] - pts[j])
         end
      end
   end

   for i=1:(nx-1)
      M[i,i] = 0.0 
      for j=1:nx
         if i!=j
            M[i,i] -= M[i,j]
         end
      end
   end

   M .*= tomyF(2)/(xmax-xmin)

   return M 
end

"""
    mat_fd_D1(
      xmin::myF,
      xmax::myF,
      nx::myI
    )::Matrix{myF}

Compute 2nd order finite difference in Chebyshev points.
"""
function mat_fd_D1(
      xmin::myF,
      xmax::myF,
      nx::myI
   )::Matrix{myF}
   @assert nx>4

   P = zeros(myF,nx,nx)
   pts = cheb_pts(xmin,xmax,nx)
   
   h(i) = pts[i+1] - pts[i]
  
   X = Vector{myF}() 
   Y = Vector{myF}() 
   V = Vector{myF}() 
   
   append!(X,1)
   append!(Y,1)
   append!(V,
      (2*pts[1]-pts[2]-pts[3])
      /
      ((pts[1]-pts[2])*(pts[1]-pts[3]))
     )
   append!(X,1)
   append!(Y,2)
   append!(V,
      (1.0/(pts[2]-pts[1])) + (1.0/(pts[3]-pts[2]))
     )
   append!(X,1)
   append!(Y,3)
   append!(V,
      (pts[1]-pts[2])
      /
      ((pts[3]-pts[1])*(pts[3]-pts[2]))
     )

   for i=2:(nx-1)
      append!(X,i)
      append!(Y,i-1)
      append!(V,-(h(i)/h(i-1))/(h(i-1)+h(i)))
      append!(X,i)
      append!(Y,i)
      append!(V,(1.0/h(i-1)) - (1.0/h(i)))
      append!(X,i)
      append!(Y,i+1)
      append!(V,(h(i-1)/h(i))/(h(i-1)+h(i)))
   end 
   
   append!(X,nx)
   append!(Y,nx)
   append!(V,
      (1.0/(pts[nx]-pts[nx-2])) + (1.0/(pts[nx]-pts[nx-1]))
     )
   append!(X,nx)
   append!(Y,nx-1)
   append!(V,
      (pts[nx-2]-pts[nx])
      /
      ((pts[nx-2]-pts[nx-1])*(pts[nx-1]-pts[nx]))
     )
   append!(X,nx)
   append!(Y,nx-2)
   append!(V,
      (pts[nx]-pts[nx-1])
      /
      ((pts[nx-2]-pts[nx-1])*(pts[nx-2]-pts[nx]))
     )

   return sparse(X,Y,V) 
end

"""
    mat_fd_D2(
      xmin::myF,
      xmax::myF,
      nx::myI
    )::Matrix{myF}

Compute 2nd order finite difference in Chebyshev points.
"""
function mat_fd_D2(
      xmin::myF,
      xmax::myF,
      nx::myI
   )::Matrix{myF}
   @assert nx>4

   P = zeros(myF,nx,nx)
   pts = cheb_pts(xmin,xmax,nx)
   
   h(i) = pts[i+1] - pts[i]
  
   X = Vector{myF}() 
   Y = Vector{myF}() 
   V = Vector{myF}() 
   
   append!(X,1)
   append!(Y,1)
   append!(V,
      2*(3*pts[1]-pts[2]-pts[3]-pts[4])
      /
      ((pts[1]-pts[2])*(pts[1]-pts[3])*(pts[1]-pts[4]))
     )
   append!(X,1)
   append!(Y,2)
   append!(V,
      2*(-2*pts[1]+pts[3]+pts[4])
      /
      ((pts[1]-pts[2])*(pts[2]-pts[3])*(pts[2]-pts[4]))
     )
   append!(X,1)
   append!(Y,3)
   append!(V,
      2*(-2*pts[1]+pts[2]+pts[4])
      /
      ((pts[1]-pts[3])*(-pts[2]+pts[3])*(pts[3]-pts[4]))
     )
   append!(X,1)
   append!(Y,4)
   append!(V,
      2*(-2*pts[1]+pts[2]+pts[3])
      /
      ((pts[2]-pts[4])*(-pts[1]+pts[4])*(-pts[3]+pts[4]))
     )

   for i=2:(nx-1)
      append!(X,i)
      append!(Y,i-1)
      append!(V,(2.0/h(i-1))/(h(i-1)+h(i)))
      append!(X,i)
      append!(Y,i)
      append!(V,- 2.0/(h(i-1)*h(i)))
      append!(X,i)
      append!(Y,i+1)
      append!(V,(2.0/h(i))/(h(i-1)+h(i)))
   end 
   append!(X,nx)
   append!(Y,nx)
   append!(V,
      2*(3*pts[nx]-pts[nx-1]-pts[nx-2]-pts[nx-3])
      /
      ((pts[nx]-pts[nx-1])*(pts[nx]-pts[nx-2])*(pts[nx]-pts[nx-3]))
     )
   append!(X,nx)
   append!(Y,nx-1)
   append!(V,
      2*(-2*pts[nx]+pts[nx-2]+pts[nx-3])
      /
      ((pts[nx]-pts[nx-1])*(pts[nx-1]-pts[nx-2])*(pts[nx-1]-pts[nx-3]))
     )
   append!(X,nx)
   append!(Y,nx-2)
   append!(V,
      2*(-2*pts[nx]+pts[nx-1]+pts[nx-3])
      /
      ((pts[nx]-pts[nx-2])*(-pts[nx-1]+pts[nx-2])*(pts[nx-2]-pts[nx-3]))
     )
   append!(X,nx)
   append!(Y,nx-3)
   append!(V,
      2*(-2*pts[nx]+pts[nx-1]+pts[nx-2])
      /
      ((pts[nx-1]-pts[nx-3])*(-pts[nx]+pts[nx-3])*(-pts[nx-2]+pts[nx-3]))
     )

   return sparse(X,Y,V) 
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
