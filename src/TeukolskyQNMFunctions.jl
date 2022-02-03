"""
TeukolskyQNMFunctions.jl computes the quasinormal modes and eigenfunctions 
for the spin s Teukolsky equation
using a horizon penetrating, hyperboloidally compactified coordinate system.
The main advantage of using these coordinates is that the quasinormal
wavefunctions are finite valued from the black hole to future null infinity.
"""
module TeukolskyQNMFunctions 

export F, compute_om

include("CustomTypes.jl")
include("Spheroidal.jl")
include("RadialODE.jl")

using .CustomTypes

import .Spheroidal as SH
import .RadialODE  as RD 


"""
    F(
      nr::myI,
      nl::myI,
      s::myI,
      lang::myI,
      mang::myI,
      a::myF,
      om::myC
    )::myF

Absolute difference of Lambda seperation constant
for radial and angular ODEs.
"""
function F(
      nr::myI,
      nl::myI,
      s::myI,
      lang::myI,
      mang::myI,
      a::myF,
      om::myC
   )::myF

   lmin = max(abs(s),abs(mang))
   neig = lang - lmin + 1 # number of eigenvalues

   la_s, _ = SH.eig_vals_vecs(nl, neig, s, mang,  a*om)
   la_r, _ = RD.eig_vals_vecs(nr,       s, mang, a, om)

   ## The Lambdas are ordered in size of smallest magnitude
   ## to largest magnitude, we ASSUME this is the same as the
   ## ordering of the l-angular indexing (this may not
   ## hold when a->1; need to check).

   return abs(la_s[lang-lmin+1] - la_r[1])
end

"""
    compute_om(
      nr::myI,
      nl::myI,
      s::myI,
      lang::myI,
      mang::myI,
      a::myF,
      om::myC; 
      tolerance::myF=tomyF(1e-6),
      epsilon::myF=tomyF(1e-6),
      gamma::myF=tomyF(1),
      verbose::Bool=false
   )::Tuple{
            myC,
            myC,
            Vector{myC},
            Vector{myC},
            Vector{myF}}

Search for quasinormal mode frequency in the complex plane
using Newton's method.

# Arguments

* `nr`:   number of radial Chebyshev collocation points
* `nl`:   number of spherical harmonic terms
* `s`:    spin of the field in the Teukolsky equation
* `lang`: l angular number
* `mang`: m angular number
* `a`:    black hole spin
* `om`:   guess for the initial quasinormal mode
"""
function compute_om(
      nr::myI,
      nl::myI,
      s::myI,
      lang::myI,
      mang::myI,
      a::myF,
      om::myC; 
      tolerance::myF=tomyF(1e-6),
      epsilon::myF=tomyF(1e-6),
      gamma::myF=tomyF(1),
      verbose::Bool=false
   )::Tuple{
            myC,
            myC,
            Vector{myC},
            Vector{myC},
            Vector{myF}}

   om_n   = tomyC(-1000)
   om_np1 = om 

   f(omega) = F(nr,nl,s,lang,mang,a,omega)

   ## Newton search with 2nd order finite differences

   newgamma = gamma
   iterations = 1
   while abs(om_np1 - om_n) > tolerance 

      ## if too many iterations, reduce search step size
      iterations += 1
      if iterations%100 == 0
         newgamma /= tomyF(2) 
      end
     
      om_n = om_np1

      df = (
            (f(om_n+epsilon)    - f(om_n-epsilon)   )/(tomyF(2)*epsilon)
            +
            (f(om_n+im*epsilon) - f(om_n-im*epsilon))/(tomyF(2)*im*epsilon)
           )
      om_np1 = om_n - newgamma*f(om_n)/df
      if verbose==true
         println("om_np1=$om_np1\tdf=$df\tdiff=$(abs(om_np1-om_n))")
      end
   end
  
   lmin = max(abs(s),abs(mang))
   neig = lang - lmin + 1 # number of eigenvalues

   la_s, v_s        = SH.eig_vals_vecs(nl, neig, s, mang,  a*om_np1)
   la_r, v_r, rvals = RD.eig_vals_vecs(nr,       s, mang, a, om_np1)

   return om_np1, la_r[1], v_s[:,lang-lmin+1], v_r[:,1], rvals 
end

end # module
