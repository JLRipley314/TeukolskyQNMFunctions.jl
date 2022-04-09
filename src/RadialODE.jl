"""
Ordinary differential equation in radial direction for the
hyperboloidally compactified Teukolsky equation.
"""
module RadialODE 

export eig_vals_vecs_c

include("CustomTypes.jl")
include("Chebyshev.jl")
using .CustomTypes
import .Chebyshev as CH 

#import IterativeSolvers: invpowm
using LinearAlgebra
using GenericSchur
using SparseArrays
using Printf: @printf 

using ApproxFun

"""
    radial_discretized_eqn(
      nr::myI,
      s::myI,
      m::myI,
      a::myF,
      bhm::myF,
      om::myC,
      rmin::myF,
      rmax::myF
    )
"""
function radial_discretized_eqn(
      nr::myI,
      s::myI,
      m::myI,
      a::myF,
      bhm::myF,
      om::myC,
      rmin::myF,
      rmax::myF
)
   ## Using position space method 
   #=
   D1 = CH.mat_D1(rmin,rmax,nr)
   D2 = D1*D1    

   Id = sparse(I,nr,nr)

   X1 = CH.mat_X(rmin,rmax,nr)
   X2 = X1*X1
   X3 = X1*X2
   X4 = X1*X3
   =#
  
   ## Using Ultraspherical polynomial method
   X1 = ApproxFun.Fun(rmin..rmax);
   X2 = X1^2
   X3 = X1^3
   X4 = X1^4
   D1 = ApproxFun.Derivative()
   D2 = D1^2 
   
   A = (
        tomyC(2*im)*om*I
        -
        tomyF(2)*(tomyF(1) + s)*X1
        +
        tomyF(2)*(
             im*om*((a^2) - tomyF(8)*(bhm^2))
             +
             im*m*a
             +
             (s + tomyF(3))*bhm
         )*X2
        +
        tomyF(4)*(tomyF(2)*im*om*bhm - tomyF(1))*(a^2)*X3
   )
   B = (
     (
      ((a^2) - tomyF(16)*(bhm^2))*(om^2) 
      + 
      tomyF(2)*(m*a + tomyF(2)*im*s*bhm)*om
     )*I
     +
     tomyF(2)*(
      tomyF(4)*((a^2) - tomyF(4)*(bhm^2))*bhm*(om^2)
      +
      (tomyF(4)*m*a*bhm - tomyF(4)*im*(s + tomyF(2))*(bhm^2) + im*(a^2))*om
      +
      im*m*a
      +
      (s + tomyF(1))*bhm
     )*X1
     +
     tomyF(2)*(
          tomyF(8)*(bhm^2)*(om^2)
          +
          tomyF(6)*im*bhm*om
          -
          tomyF(1)
     )*(a^2)*X2
   )

   return (
      - 
      (X2 - tomyF(2)*bhm*X3 + (a^2)*X4)*D2
      +
      A*D1
      +
      B
   ) : ApproxFun.space(X1)
end

"""
    radial_discretized_eqn_p(
      np::myI,
      s::myI,
      m::myI,
      a::myF,
      bhm::myF,
      om::myC
    )
"""
function radial_discretized_eqn_p(
      np::myI,
      s::myI,
      m::myI,
      a::myF,
      bhm::myF,
      om::myC
)
   X = Vector{myI}()
   Y = Vector{myI}()
   V = Vector{myC}()

   A0 = tomyF(2)*im*om
   A1 = -tomyF(2)*(tomyF(1)+s)
   A2 = tomyF(2)*(im*om*((a^2) - tomyF(8)*(bhm^2)) + im*m*a + (s+tomyF(3))*bhm)
   A3 = tomyF(4)*(tomyF(2)*im*om*bhm - tomyF(1))*(a^2)
      
   B0 = ((a^2) - tomyF(16)*(bhm^2))*(om^2) + tomyF(2)*(m*a + tomyF(2)*im*s*bhm)*om 
   B1 = tomyF(2)*(
           tomyF(4)*((a^2) - tomyF(4)*(bhm^2))*bhm*(om^2) 
           + 
           (tomyF(4)*m*a*bhm - tomyF(4)*im*(s+2)*(bhm^2) + im*(a^2))*om
           +
           im*m*a
           +
           (s + tomyF(1))*bhm
          )
   B2 = tomyF(2)*(tomyF(8)*((bhm*om)^2) + tomyF(6)*im*om*bhm - tomyF(1))*(a^2) 

   push!(X,1)
   push!(Y,1)
   push!(V,B0)
   
   push!(X,1)
   push!(Y,2)
   push!(V,A0)
   
   push!(X,2)
   push!(Y,1)
   push!(V,B1)
   
   push!(X,2)
   push!(Y,2)
   push!(V,A1+B0)
   
   push!(X,2)
   push!(Y,3)
   push!(V,2*A0)
   
   for i=3:(np-1)
      n = i-1
      push!(X,i) 
      push!(Y,i-2) 
      push!(V,-(n-2)*(n-3)*(a^2) + (n-2)*A3 + B2)
      
      push!(X,i) 
      push!(Y,i-1) 
      push!(V,2*(n-1)*(n-2)*bhm + (n-1)*A2 + B1)

      push!(X,i)
      push!(Y,i)
      push!(V,-n*(n-1) + n*A1 + B0)

      push!(X,i)
      push!(Y,i+1)
      push!(V,(n+1)*A0)
   end
   
   n = np-1
   push!(X,np) 
   push!(Y,np-2) 
   push!(V,-(n-2)*(n-3)*(a^2) + (n-2)*A3 + B2)
   
   push!(X,np) 
   push!(Y,np-1); 
   push!(V,2*(n-1)*(n-2)*bhm + (n-1)*A2 + B1)

   push!(X,np)
   push!(Y,np)
   push!(V,-n*(n-1) + n*A1 + B0)

   mat = dropzeros(sparse(X,Y,V))
   for i=1:np
      mat[:,i] .*= 2^(i-1)
   end

   mat = dropzeros(sparse(X,Y,V))

   #for i=1:np
   #   mat[:,i] .*= 2^(i-1)
   #end

   return mat
end

"""
    eig_vals_vecs_c(
      nr::myI,
      s::myI,
      m::myI,
      a::myF,
      om::myC
    )

Compute eigenvectors and eigenvalues for the radial equation
using a pseudospectral Chebyshev polynomial method.
The black hole mass is always one.
"""
function eig_vals_vecs_c(
      nr::myI,
      s::myI,
      m::myI,
      a::myF,
      om::myC
   )
   bhm  = tomyF(1) ## always have unit black hole mass
   rmin = tomyF(0)
   rmax = tomyF(abs(a)>0 ? (bhm/(a^2))*(tomyF(1) - sqrt(tomyF(1)- ((a/bhm)^2))) : tomyF(0.5)/bhm)
   
   L = radial_discretized_eqn(nr,s,m,a,bhm,om,rmin,rmax)
   t = ApproxFun.eigs(L,nr,tolerance=1e-12) 

   mini = argmin(abs.(t[1]))

   #nMat = radial_discretized_eqn_p(nr,s,m,a,bhm,om)
   #nt = eigen(Matrix(nMat), permute=true, scale=true, sortby=abs)
   #println(nMat)

   return -t[1][mini], t[2][mini], CH.cheb_pts(rmin,rmax,nr)
end

end # module
