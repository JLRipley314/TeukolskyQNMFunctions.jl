""" 
   Implementing the ideas in 

      @ARTICLE{1993JCoPh.104..211D,
          author = {{Dang-Vu}, H. and {Delcarte}, C.},
           title = "{An Accurate Solution of the Poisson Equation by the Chebyshev Collocation Method}",
         journal = {Journal of Computational Physics},
            year = 1993,
           month = jan,
          volume = {104},
          number = {1},
           pages = {211-220},
             doi = {10.1006/jcph.1993.1021},
          adsurl = {https://ui.adsabs.harvard.edu/abs/1993JCoPh.104..211D},
         adsnote = {Provided by the SAO/NASA Astrophysics Data System}
      }
"""
module ChebyshevSpec

include("CustomTypes.jl")
using .CustomTypes
using SparseArrays

export mat_invD1, mat_invD2

function mat_invD1(nx::myI)
   X = Vector{myF}()
   Y = Vector{myF}()
   V = Vector{myF}()
   
   c(n) = (n==0)      ? 2.0 : 1.0
   e(n) = (n<=(nx-1)) ? 1.0 : 0.0
   
   for i=2:(nx-1)
      n=i-1
      push!(X,i)
      push!(Y,i-1)
      push!(V,tomyF(c(n-1)*0.5/n))
      
      push!(X,i)
      push!(Y,i+1)
      push!(V,tomyF(-e(n+1)*0.5/n))
   end
   
   return dropzeros(sparse(X,Y,V))
end

function mat_invD2(nx::myI)
   X = Vector{myF}()
   Y = Vector{myF}()
   V = Vector{myF}()

   c(n) = (n==0)      ? 2.0 : 1.0
   e(n) = (n<=(nx-1)) ? 1.0 : 0.0

   for i=3:(nx-2)
      n=i-1
      push!(X,i)
      push!(Y,i-2)
      push!(V,tomyF(c(n-2)*0.25/(n*(n-1))))
      
      push!(X,i)
      push!(Y,i)
      push!(V,tomyF(-e(n+2)*0.5/((n+1)*(n-1))))
      
      push!(X,i)
      push!(Y,i+2)
      push!(V,tomyF(e(n+4)*0.25/(n*(n+1))))
   end 
   n=nx-1
   push!(X,nx-1)
   push!(Y,nx-3)
   push!(V,tomyF(c(n-2)*0.25/(n*(n-1))))
   
   n=nx
   push!(X,nx)
   push!(Y,nx-2)
   push!(V,tomyF(c(n-2)*0.25/(n*(n-1))))
   
   return dropzeros(sparse(X,Y,V))
end

end
