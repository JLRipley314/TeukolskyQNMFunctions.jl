module TestCHLaplacian

export interval_laplacian

push!(LOAD_PATH,"../src/")

const tolerance = 1e-6 ## tolerance we compare to

using CustomTypes

using LinearAlgebra: I, eigen
using SparseArrays

import Test: @test

import Chebyshev as CH 

##============================================================
"""
Compare Eigenvalues on an interval 
"""
function interval_laplacian(
      nx::myI,
      neig::myI,
      xmin::myF,
      xmax::myF
   )

   D1 = CH.mat_D1(xmin, xmax, nx)
   D2 = -D1*D1 
   Id = sparse(I,nx,nx)

   D2[1,1]      = 1.0
   D2[1,2:end] .= 0.0
   
   D2[nx,nx]      = 1.0
   D2[nx,1:nx-1] .= 0.0

   t = eigen(D2)

   L = xmax - xmin

   for (n,la) in enumerate(t.values[3:div(end,2,RoundNearest)]) 
      v = (pi*n/L)^2 
      @test abs((la - v)/v) < tolerance
   end
end

end
