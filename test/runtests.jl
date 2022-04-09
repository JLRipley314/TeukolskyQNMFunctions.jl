using Test 

include("../src/CustomTypes.jl")
include("../src/Spheroidal.jl")
include("../src/TeukolskyQNMFunctions.jl")

include("TestCHLaplacian.jl")
include("TestChebyshev.jl")
include("TestGegenbauer.jl")
include("TestRadialODE.jl")
include("TestSphere.jl")
include("TestSpheroidal.jl")

using .CustomTypes, ..TeukolskyQNMFunctions

import .TestCHLaplacian 
import .TestChebyshev 
import .TestGegenbauer 
import .TestRadialODE
import .TestSphere
import .TestSpheroidal

import .Spheroidal: compute_l_min, eig_vals_vecs

avals = [0.0, 0.354, 0.7, 0.99] ## same as in generate.py

###--------------------------------------------------------------
## Test convergence of Chebyshev derivatives 
##--------------------------------------------------------------
nx   = tomyI(16)
xmin = tomyF(-1.431)
xmax = tomyF(10.3)

f(x)   =      sin(2.0*x)
d1f(x) =  2.0*cos(2.0*x)
d2f(x) = -4.0*sin(2.0*x)
TestChebyshev.test_convergence(   nx, xmin, xmax, f, d1f, d2f) 
TestChebyshev.test_convergence_fd(nx, xmin, xmax, f, d1f, d2f) 

f(x)   =      exp(-2.0*x)
d1f(x) = -2.0*exp(-2.0*x)
d2f(x) =  4.0*exp(-2.0*x)
TestChebyshev.test_convergence(   nx, xmin, xmax, f, d1f, d2f) 
TestChebyshev.test_convergence_fd(nx, xmin, xmax, f, d1f, d2f) 

TestChebyshev.test_X_matrices(nx, xmin, xmax) 

TestChebyshev.test_to_cheb(nx) 
TestChebyshev.test_to_cheb_to_real(nx, xmin, xmax) 
##--------------------------------------------------------------
## Test Chebyshev second derivative 
##--------------------------------------------------------------
nx   = tomyI(40) 
neig = tomyI(4)
xmin = tomyF(-1.0)
xmax = tomyF(1.0)

TestCHLaplacian.interval_laplacian_ch(nx,neig,xmin,xmax)
nx = tomyI(320) 
TestCHLaplacian.interval_laplacian_fd(nx,neig,xmin,xmax)
#nx  = tomyI(40)
#TestCHLaplacian.interval_laplacian_chs(nx,neig,xmin,xmax)
##--------------------------------------------------------------
## Testing spherical function 
##--------------------------------------------------------------
TestSphere.test_swal_inner_product(48,-2,-2)
TestSphere.test_swal_inner_product(48,-1,-2)
TestSphere.test_swal_inner_product(48, 0,-2)
TestSphere.test_swal_inner_product(48, 1,-2)
TestSphere.test_swal_inner_product(48, 2,-2)

TestSphere.test_swal_inner_product(48,-2,0)
TestSphere.test_swal_inner_product(48,-1,0)
TestSphere.test_swal_inner_product(48, 0,0)
TestSphere.test_swal_inner_product(48, 1,0)
TestSphere.test_swal_inner_product(48, 2,0)

TestSphere.test_swal_inner_product(48,-2,3)
TestSphere.test_swal_inner_product(48,-1,3)
TestSphere.test_swal_inner_product(48, 0,3)
TestSphere.test_swal_inner_product(48, 1,3)
TestSphere.test_swal_inner_product(48, 2,3)

TestSphere.test_norm_swal_lap(32, 0,0,2)

TestSphere.test_norm_swal_lap(32,-1,1,2)
TestSphere.test_norm_swal_lap(32, 0,1,2)
TestSphere.test_norm_swal_lap(32, 1,1,2)

TestSphere.test_norm_swal_lap(32,-2,-2,2)
TestSphere.test_norm_swal_lap(32,-1,-2,2)
TestSphere.test_norm_swal_lap(32, 0,-2,2)
TestSphere.test_norm_swal_lap(32, 1,-2,2)
TestSphere.test_norm_swal_lap(32, 2,-2,2)

TestSphere.test_norm_swal_lap(48,-2,5,23)
TestSphere.test_norm_swal_lap(48,-2,5,23)
TestSphere.test_norm_swal_lap(48,-1,5,23)
TestSphere.test_norm_swal_lap(48, 0,5,23)
TestSphere.test_norm_swal_lap(48, 1,5,23)
TestSphere.test_norm_swal_lap(48, 2,5,23)
###--------------------------------------------------------------
## Test Gegenbauer polynomial 
##--------------------------------------------------------------
#=
nx=tomyI(20)
c0=tomyF(1.3243)
c1=tomyF(2.3)
om2=tomyF(1.0)
xmin=tomyF(-2.2321)
xmax=tomyF(+1.932)
TestGegenbauer.constant_sol_ode(nx,xmin,xmax,c0)
TestGegenbauer.linear_sol_ode(nx,xmin,xmax,c0,c1)
TestGegenbauer.airy_sol_ode(nx,xmin,xmax,om2)
n=tomyI(1)
a=tomyF(1.0)
b=tomyF(1.0)
TestGegenbauer.jacobi_sol_ode(6,n,a,b)
=#
##--------------------------------------------------------------
## Testing spheroidal functions 
##--------------------------------------------------------------
nl   =  30
neig =  10

for s in [-2,-1,0]
   for m in [-3, 0]
      for a in avals 
         for omegar in rand(2)
            for omegai in rand(2)
               om = (-1.0 + 2.0*omegar) + (-1.0 + 2.0*omegai)*im
               ## test reduces to spin weighted spherical harmonics
               ## when a*omega = 0
               if abs(a)<1e-16
                  lmin = compute_l_min(tomyI(s),tomyI(m)) 
                  ls, vs = eig_vals_vecs(tomyI(nl), tomyI(neig), tomyI(s), tomyI(m), tomyC(a*om))
                  for (i,la) in enumerate(ls)
                     l = (i-1) + lmin
                     @test abs((l-s)*(l+s+1) - la) < 1e-3 
                  end
               end
               TestSpheroidal.test_m_symmetry(tomyI(nl),    tomyI(neig), tomyI(s), tomyI(m), tomyF(a), tomyC(om)) 
               TestSpheroidal.test_s_symmetry(tomyI(nl),    tomyI(neig), tomyI(s), tomyI(m), tomyF(a), tomyC(om)) 
               TestSpheroidal.test_conj_symmetry(tomyI(nl), tomyI(neig), tomyI(s), tomyI(m), tomyF(a), tomyC(om)) 
            end
         end
      end
   end
end
##--------------------------------------------------------------
## Testing Spheroidal ODE QNM 
##--------------------------------------------------------------
for n=[0,1]
   for s=0:2
      for m=[0,2]
         lmin = compute_l_min(s,m)
         for l=lmin:(lmin+4)
            TestSpheroidal.compare_to_qnm(
                           tomyI(nl),
                           tomyI(neig),
                           tomyI(s),
                           tomyI(n),
                           tomyI(l),
                           tomyI(m),
                           [tomyF(a) for a in avals]
               )
         end
      end
   end
end
##--------------------------------------------------------------
## Testing Radial ODE QNM 
##--------------------------------------------------------------
nr = 144

for n=[0,1]
   for s=-2:0
      mvals = [abs(s)]
      for m=[-2,0,1]
         lmin = compute_l_min(s,m)
         lvals = [lmin]
         if n==0
            mvals = lmin:(lmin+1) 
         end  
         for l=lvals
            TestRadialODE.compare_to_qnm(
                  tomyI(nr),
                  tomyI(s),
                  tomyI(n),
                  tomyI(l),
                  tomyI(m),
                  [tomyF(a) for a in avals]
               )
         end
      end
   end
end
