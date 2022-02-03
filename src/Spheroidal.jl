"""
Methods to compute spin-weighted spheroidal harmonics
"""
module Spheroidal

include("CustomTypes.jl")

using .CustomTypes
using SparseArrays

import LinearAlgebra: eigen
using GenericSchur

"""
    compute_l_min(s,m)

Minimum l value
"""
function compute_l_min(s,m)
   return max(abs(s),abs(m))
end

"""
    compute_A(s::myI, l::myI, m::myI)

See the note. 
"""
@inline function compute_A(s::myI, l::myI, m::myI)
   return sqrt(
      (l+tomyF(1)-s)*(l+tomyF(1)+s)*(l+tomyF(1)+m)*(l+tomyF(1)-m)
      /
      (((l+tomyF(1))^2)*(tomyF(2)*l+tomyF(1))*(tomyF(2)*l+tomyF(3)))
   )
end
"""
    compute_B(s::myI, l::myI, m::myI)

See the note. 
"""
@inline function compute_B(s::myI, l::myI, m::myI)
   if (l==0)
      return tomyF(0)
   else
      return -(m*s)/(l*(l+tomyF(1)))
   end
end
"""
    compute_C(s::myI, l::myI, m::myI)

See the note.
"""
@inline function compute_C(s::myI, l::myI, m::myI)
   if (l==0)
      return tomyF(0)
   else
      return sqrt(
         tomyF(l-s)*tomyF(l+s)*tomyF(l+m)*tomyF(l-m)
         /
         ((l^2)*(tomyF(2)*l+tomyF(1))*(tomyF(2)*l-tomyF(1)))
      )
   end
end
"""
    mat_Y(
      nl::myI,
      s::myI,
      m::myI,
    )::SparseMatrixCSC

Returns the matrix for y in spectral space.
"""
function mat_Y(
      nl::myI,
      s::myI,
      m::myI,
   )::SparseMatrixCSC

   I = Vector{myI}(undef,0)
   J = Vector{myI}(undef,0)
   V = Vector{myF}(undef,0)

   lmin = compute_l_min(s,m) 

   for i=1:nl

      l = (i-1) + lmin

      ## (l,l) component 
      append!(I,i)
      append!(J,i)
      append!(V,compute_B(s,l,m))
      
      ## (l,l-1) component 
      if (i>1)
         append!(I,i)
         append!(J,i-1)
         append!(V,compute_C(s,l,m))
      end
      ## (l,l+1) component 
      if (i+1<=nl)
         append!(I,i)
         append!(J,i+1)
         append!(V,compute_A(s,l,m))
      end
   end
   return sparse(I,J,V)
end

"""
    mat_L(
      nl::myI,
      s::myI,
      m::myI,
    )::SparseMatrixCSC

Returns the matrix for spherical laplacian (s=0) in spectral space.
"""
function mat_L(
      nl::myI,
      s::myI,
      m::myI,
   )::SparseMatrixCSC

   I = Vector{myI}(undef,0)
   J = Vector{myI}(undef,0)
   V = Vector{myF}(undef,0)

   lmin = compute_l_min(s,m) 

   for i=1:nl

      l = (i-1) + lmin
      
      ## (l,l) component 
      append!(I,i)
      append!(J,i)
      append!(V,tomyF(l - s)*tomyF(l + s + 1.0))
   end
   return sparse(I,J,V)
end
"""
    compute_M_matrix(
      nl::myI,
      s::myI,
      m::myI,
      c::myC
    )

M matrix for computing spheroidal-spherical mixing and seperation coefficients.
"""
function compute_M_matrix(
      nl::myI,
      s::myI,
      m::myI,
      c::myC
   )
   Y1 = mat_Y(nl+2,s,m) # look at larger to capture last term in matrix mult
                        # at higher l
   Y2 = Y1*Y1 
   Y1 = Y1[1:nl,1:nl]
   Y2 = Y2[1:nl,1:nl]
   L  = mat_L(nl,s,m)
  
   return sparse(
         transpose(
            -
            (c^2)*Y2
            +
            tomyF(2)*c*s*Y1
            +
            L
         )
      )

end

"""
    eig_vals_vecs(
      nl::myI,
      neig::myI,
      s::myI,
      m::myI,
      c::myC
    )

Compute eigenvectors and eigenvalues for the spheroidal equation.  
Returns eigenvalues λs (smallest to largest), and eigenvectors as
an array (access nth eigenvector through v[:,n]); i.e. returns
(λs, vs)
"""
function eig_vals_vecs(
      nl::myI,
      neig::myI,
      s::myI,
      m::myI,
      c::myC
   )

   Mat = compute_M_matrix(nl, s, m, c)

   for i=1:nl
      Mat[i,i] += tomyF(1) # shift to avoid issues with a zero eigenvalue
   end
   # λs, vs
   t = eigen(Matrix(Mat), permute=true, scale=true, sortby=abs)
   return t.values .- tomyF(1), t.vectors 
end

end # module
