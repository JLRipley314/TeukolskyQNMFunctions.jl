module TestGegenbauer

export constant_ode, linear_ode, trig_ode 

include("../src/CustomTypes.jl")
include("../src/Chebyshev.jl")
include("../src/Gegenbauer.jl")
using LinearAlgebra, SparseArrays
using SpecialFunctions: airyai
using Jacobi: jacobi 
using .CustomTypes
import .Chebyshev as CH 
import .Gegenbauer as GE 

import Test: @test

const tolerance = 1e-8 ## tolerance we compare to

##============================================================

"""
Evaluate ODE u' = 0, with u(xmin) = c0.
"""
function constant_sol_ode(
      nx::myI,
      xmin::myF,
      xmax::myF,
      c0::myF
   ) 
   D1 = GE.compute_D(nx,tomyI(1))
   D1 = [transpose([(-1)^i for i=0:(nx-1)]);D1]

   f0 = [c0; [0.0 for i=1:nx]]

   numerical_sol = D1\f0
   analytic_sol  = [c0; [0.0 for i=1:(nx-1)]]

   println("Test: Gegenbauer constant solution ODE")
   @test norm(numerical_sol .- analytic_sol,1) < tolerance
   return nothing 
end

"""
Evaluate ODE u'' = 0, with u(xmin) = c0, u'(xmax) = c1.
"""
function linear_sol_ode(
      nx::myI,
      xmin::myF,
      xmax::myF,
      c0::myF,
      c1::myF
   ) 
   D2 = GE.compute_D(nx,tomyI(2))
   D2 = [transpose([1      for i=0:(nx-1)]);D2]
   D2 = [transpose([(-1)^i for i=0:(nx-1)]);D2]

   f0 = [c0; c1; [0.0 for i=1:nx]]

   numerical_sol = D2\f0
   
   ## coordinate change
   numerical_sol[1] -= numerical_sol[2]*(xmax+xmin)/(xmax-xmin)
   numerical_sol[2] *= 2.0/(xmax-xmin)
   
   analytic_sol  = [
                    (c0*xmax - c1*xmin)/(xmax-xmin); 
                    (c1-c0)/(xmax-xmin); 
                    [0.0 for i=1:(nx-2)]
                   ]

   println("Test: Gegenbauer linear solution ODE")
   @test norm(numerical_sol .- analytic_sol,1) < tolerance
   return nothing 
end

"""
Evaluate ODE u'' - x*u = 0, with u(xmin) = Airy(xmin), u(xmax) = Airy(xmax).
"""
function airy_sol_ode(
      nx::myI,
      xmin::myF,
      xmax::myF,
      om2::myF
   ) 
   D2  = GE.compute_D(nx,tomyI(2),xmin,xmax)
   X1  = GE.compute_MXp(nx,tomyI(2),tomyI(1),xmin,xmax)
   S0  = GE.compute_S(nx,tomyI(0))
   S1  = GE.compute_S(nx,tomyI(1))
   
   Op  = D2 - X1*S1*S0 
   Op  = [transpose([1      for i=0:(nx-1)]);Op]
   Op  = [transpose([(-1)^i for i=0:(nx-1)]);Op]

   f0 = [airyai(xmin); airyai(xmax); [0.0 for i=1:nx]]

   numerical_sol = Op\f0 
   
   pts = CH.cheb_pts(xmin,xmax,nx)
   analytic_sol = CH.to_cheb(airyai.(pts))

   println("Test: Gegenbauer Airy function solution ODE")
   @test norm(numerical_sol .- analytic_sol,1) < tolerance
   return nothing 
end

"""
Evaluate ODE Jacobi equation
"""
function jacobi_sol_ode(
      nx::myI,
      n::myI,
      a::myF,
      b::myF
   ) 
   xmin = tomyF(-1)
   xmax = tomyF(+1)

   f(x) = jacobi(x, n, a, b)

   Id  = sparse(I,nx,nx)
   D1  = GE.compute_D(nx,tomyI(1),xmin,xmax)
   D2  = GE.compute_D(nx,tomyI(2),xmin,xmax)
   X1  = GE.compute_MXp(nx,tomyI(2),tomyI(1),xmin,xmax)
   X2  = GE.compute_MXp(nx,tomyI(2),tomyI(2),xmin,xmax)
   S0  = GE.compute_S(nx,tomyI(0))
   S1  = GE.compute_S(nx,tomyI(1))

   println(Id)
   println(X2)
   println(D2)
   println((Id .- X2)*D2 .+ ((b-a)*Id .- (a+b+2.0)*X1)*S1*D1 .+ n*(n+a+b+1)*S1*S0)

   Op  = (Id .- X2)*D2 .+ ((b-a)*Id .- (a+b+2.0)*X1)*S1*D1 #+ n*(n+a+b+1)*S1*S0

   println(Op)
   println(eigen(Matrix(Op)).values)
   
   f0 = [0.0 for i=1:nx]

   numerical_sol = Op\f0

   pts = CH.cheb_pts(xmin,xmax,nx)
   analytic_sol  = f.(pts) 
   analytic_sol  = CH.to_cheb(analytic_sol) 

   println(f0)
   println(Op)
   println(numerical_sol)
   println(analytic_sol)

   println("Test: Gegenbauer Jacobi function solution ODE")
   @test norm(numerical_sol .- analytic_sol,1) < tolerance
   return nothing 
end

end
