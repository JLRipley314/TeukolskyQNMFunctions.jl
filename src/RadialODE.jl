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

"""
    radial_discretized_eqn_c(
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
function radial_discretized_eqn_c(
      nr::myI,
      s::myI,
      m::myI,
      a::myF,
      bhm::myF,
      om::myC,
      rmin::myF,
      rmax::myF
)
   D1 = CH.mat_D1(rmin,rmax,nr)
   D2 = D1*D1    

   Id = sparse(I,nr,nr)

   X1 = CH.mat_X(rmin,rmax,nr)
   X2 = X1*X1
   X3 = X1*X2
   X4 = X1*X3

   A = (
        tomyC(2*im)*om*Id
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
     )*Id
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
   )
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
   
   Mat = radial_discretized_eqn_c(nr,s,m,a,bhm,om,rmin,rmax)
   t = eigen(Matrix(Mat), permute=true, scale=true, sortby=abs)

   return -t.values[1], t.vectors[:,1], CH.cheb_pts(rmin,rmax,nr)
end

end # module
