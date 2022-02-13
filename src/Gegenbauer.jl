"""
   Implements basic ideas in 

      S. Olver and A. Townsend, 
      A fast and well-conditioned spectral method, 
      SIAM Review, 55 (2013), pp. 462–489
      arXiv:1409.2789
"""
module Ultraspherical

include("CustomTypes.jl")
using .CustomTypes
using LinearAlgebra, SparseArrays

"""
   compute_S(n::myI, lam::myI)

   Computes n×n S_{λ} matrix.
"""
function compute_S(n::myI, lam::myI)
   @assert lam>=0
   @assert n>=3
   X = Vector{myI}(undef,0)
   Y = Vector{myI}(undef,0)
   V = Vector{myF}(undef,0)
   push!(X,1)
   push!(Y,1)
   push!(V,tomyF(1.0))
   if lam==0
      push!(X,1)
      push!(Y,3)
      push!(V,tomyF(-0.5))
      for i=2:(n-2)
         push!(X,i)
         push!(Y,i)
         push!(V,tomyF(0.5))
         push!(X,i)
         push!(Y,i+2)
         push!(V,tomyF(-0.5))
      end
      push!(X,n-1)
      push!(Y,n-1)
      push!(V,tomyF(0.5))
      push!(X,n)
      push!(Y,n)
      push!(V,tomyF(0.5))
   else
      push!(X,1)
      push!(Y,3)
      push!(V,tomyF(-lam/(lam+2.0)))
      for i=2:(n-2)
         push!(X,i)
         push!(Y,i)
         push!(V,tomyF(lam/(lam+2.0)))
         push!(X,i)
         push!(Y,i+2)
         push!(V,tomyF(-lam/(lam+2.0)))
      end
      push!(X,n-1)
      push!(Y,n-1)
      push!(V,tomyF(lam/(lam+2.0)))
      push!(X,n)
      push!(Y,n)
      push!(V,tomyF(lam/(lam+2.0)))
   end
   return sparse(X,Y,V)
end

"""
   compute_D(n::myI, lam::myI)

   Computes n×n D_{λ} matrix.
"""
function compute_D(n::myI, lam::myI)
   @assert lam>=1
   @assert n>=lam
   X = Vector{myI}(undef,0)
   Y = Vector{myI}(undef,0)
   V = Vector{myF}(undef,0)
   for i=1:(n-lam)
      push!(X,i)
      push!(Y,i+1+lam)
      push!(V,tomyF(lam+1.0))
   end
   return dropzeros(tomyF((2^(lam-1))*factorial(lam-1)).*sparse(X,Y,V))
end

"""
   compute_M(n::myI, lam::myI)

   Computes n×n M_{λ} matrix for polynomial x.
   For a simple derivation see Sec. 6.3.1 of
      Computing with functions in two dimensions,
      Townsend, Alex
      PhD Thesis, University of Oxford
   This can be accessed at
   https://pi.math.cornell.edu/~ajt/
"""
function compute_M(n::myI, lam::myI)
   @assert lam>=1
   @assert n>=lam
   X = Vector{myI}(undef,0)
   Y = Vector{myI}(undef,0)
   V = Vector{myF}(undef,0)
   push!(X,1)
   push!(Y,2)
   push!(V,tomyF(lam/(lam+1.0)))
   for i=2:(n-1)
      push!(X,i)
      push!(Y,i+1)
      push!(V,tomyF(2.0*(lam+i-1)/(2.0*(lam+i))))
      push!(X,i)
      push!(Y,i-1)
      push!(V,tomyF(1.0/(lam+i-2)))
   end
   push!(X,n)
   push!(Y,n-1)
   push!(V,tomyF(lam/(lam+n-2)))
   return sparse(X,Y,V)
end


"""
   compute_D(n::myI, lam::myI)

   Computes n×n D_{λ} matrix over the interval [rmin,rmax].
"""
function compute_D(n::myI, lam::myI, rmin::myF, rmax::myF)
   D = compute_D(n,lam)
   
   a = tomyF(0.5*(rmax-rmin))

   return dropzeros((a^(-lam)).*D) 
end

"""
   compute_M(n::myI, lam::myI, rmin::myF, rmax::myF)

   Computes n×n M_{λ} matrix for polynomial x over
   the interval [rmin,rmax].
"""
function compute_M(n::myI, lam::myI, rmin::myF, rmax::myF)
   M = compute_M(n,lam)

   a = tomyF(0.5*(rmax-rmin))
   b = tomyF(0.5*(rmax+rmin))

   return dropzeros(a.*M .+ b.*sparse(I,n,n)) 
end

end
