"""
   Implements basic ideas in 

      S. Olver and A. Townsend, 
      SIAM review, 2013,
      arXiv:1202.1347

   For more reviews, see Sec. 3 of 

      A. Townsend and S. Olver,
      The automatic solution of partial differential 
      equations using a global spectral method,
      Journal of Computational Physics, 2015
      arXiv:1409.2789
   
   and/or Sec. 6 of 
      
      A. Townsend,
      Computing with functions in two dimensions,
      PhD Thesis, University of Oxford
      (can be accessed at https://pi.math.cornell.edu/~ajt/)

"""
module Gegenbauer 

export compute_S, compute_D, compute_MX1, compute_MXp

include("CustomTypes.jl")
include("Chebyshev.jl")
using .CustomTypes
using .Chebyshev
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
         push!(V,tomyF(lam/(lam+i-1.0)))
         push!(X,i)
         push!(Y,i+2)
         push!(V,tomyF(-lam/(lam+i+1.0)))
      end
      push!(X,n-1)
      push!(Y,n-1)
      push!(V,tomyF(lam/(lam+n-2.0)))
      push!(X,n)
      push!(Y,n)
      push!(V,tomyF(lam/(lam+n-1.0)))
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
      push!(Y,i+lam)
      push!(V,tomyF(lam+i-1))
   end
   for i=1:lam
      push!(X,n-lam+i)
      push!(Y,n)
      push!(V,tomyF(0))
   end
   return dropzeros(tomyF((2^(lam-1))*factorial(lam-1)).*sparse(X,Y,V))
end

"""
   compute_MX1(n::myI, lam::myI)

   Computes n×n M_{λ} matrix for polynomial x.
   For a simple derivation see Sec. 6.3.1 of
      Computing with functions in two dimensions,
      Townsend, Alex
      PhD Thesis, University of Oxford
   This can be accessed at
   https://pi.math.cornell.edu/~ajt/
"""
function compute_MX1(n::myI, lam::myI)
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
      push!(V,tomyF(((2.0*lam)+i-1)/(2.0*(lam+i))))
      push!(X,i)
      push!(Y,i-1)
      push!(V,tomyF((i-1)/(2.0*(lam+i-2))))
   end
   push!(X,n)
   push!(Y,n-1)
   push!(V,tomyF((n-1)/(2.0*(lam+n-2))))
   return sparse(X,Y,V)
end

"""
   compute_MCp(n::myI, lam::myI, p::myI)

   Computes n×n M_{λ} matrix for polynomial C^{λ}_p.
   For a simple derivation see Sec. 6.3.1 of
      Computing with functions in two dimensions,
      Townsend, Alex
      PhD Thesis, University of Oxford
   This can be accessed at
   https://pi.math.cornell.edu/~ajt/
"""
function compute_MCp(n::myI, lam::myI, p::myI)
   @assert p>=0

   MX0 = sparse(I,n,n)
   MX1 = compute_MX1(n,lam)
   
   if p==0
      return MX0
   elseif p==1
      return (2.0*lam).*MX1
   else
      MCp1 = deepcopy(sparse(I,n,n))
      MC   = deepcopy((2.0*lam).*MX1) 
      MCm1 = deepcopy(MX0) 

      for k=1:(p-1)
         MCp1 = (
            (2.0*(k+lam)/(k+1)).*(MX1*MC)
            -
            ((k-1+2*lam)/(k+1)).*MCm1
           )
         MCm1 = deepcopy(MC)
         MC   = deepcopy(MCp1)
      end
      return MCp1
   end
end

"""
   compute_MXp(n::myI, lam::myI, p::myI)

   Computes n×n M_{λ} matrix for polynomial x_p.
   For a simple derivation see Sec. 6.3.1 of
      Computing with functions in two dimensions,
      Townsend, Alex
      PhD Thesis, University of Oxford
   This can be accessed at
   https://pi.math.cornell.edu/~ajt/
"""
function compute_MXp(n::myI, lam::myI, p::myI)
   @assert p>=0

   MCp = compute_MCp(n,lam,p)

   if p==0
      return dropzeros(MCp)
   elseif p==1
      return dropzeros((0.5/lam).*MCp)
   elseif p==2
      MC0 = compute_MCp(n,lam,0)
      return dropzeros(((0.5/lam)/(1.0+lam)).*(MCp + lam.*MC0)) 
   elseif p==3
      return nothing
   elseif p==4
      return nothing
   end
end

"""
   compute_D(n::myI, lam::myI)

   Computes n×n D_{λ} matrix over the interval [xmin,xmax].
"""
function compute_D(n::myI, lam::myI, xmin::myF, xmax::myF)
   D = compute_D(n,lam)
   
   a = tomyF(0.5*(xmax-xmin))

   return dropzeros((a^(-lam)).*D) 
end

"""
   compute_MXp(n::myI, lam::myI, p::myI, xmin::myF, xmax::myF)

   Computes n×n M_{λ} matrix for polynomial x_p over the interval [xmin,xmax].
"""
function compute_MXp(n::myI, lam::myI, p::myI, xmin::myF, xmax::myF)

   a = tomyF(0.5*(xmax-xmin))
   b = tomyF(0.5*(xmax+xmin))

   if p==0
      return compute_MXp(n,lam,0)
   elseif p==1
      return dropzeros(
               a*compute_MXp(n,lam,1) 
               + 
               b*compute_MXp(n,lam,0)
              )
   elseif p==2
      return dropzeros(
               (a^2)*compute_MXp(n,lam,2) 
               + 
               2*a*b*compute_MXp(n,lam,1)
               + 
               (b^2)*compute_MXp(n,lam,0)
              )
   end
end

end
