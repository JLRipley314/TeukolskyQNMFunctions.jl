var documenterSearchIndex = {"docs":
[{"location":"#TeukolskyQNMFunctions","page":"Home","title":"TeukolskyQNMFunctions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for TeukolskyQNMFunctions.","category":"page"},{"location":"","page":"Home","title":"Home","text":"TeukolskyQNMFunctions computes the quasinormal modes and eigenfunctions  for the spin s Teukolsky equation using a horizon penetrating, hyperboloidally compactified coordinate system. The main advantage of using these coordinates is that the quasinormal wavefunctions are finite valued from the black hole to future null infinity.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Currently, this code uses a Chebyshev pseudospectral method to compute the radial part of the eigenfunctions,  and a spectral method to compute the angular part of the eigenfunctions The angular spectral method was originally introduced by  Cook and Zalutskiy).","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TeukolskyQNMFunctions","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TeukolskyQNMFunctions]","category":"page"},{"location":"#TeukolskyQNMFunctions.TeukolskyQNMFunctions","page":"Home","title":"TeukolskyQNMFunctions.TeukolskyQNMFunctions","text":"TeukolskyQNMFunctions.jl computes the quasinormal modes and eigenfunctions  for the spin s Teukolsky equation using a horizon penetrating, hyperboloidally compactified coordinate system. The main advantage of using these coordinates is that the quasinormal wavefunctions are finite valued from the black hole to future null infinity.\n\n\n\n\n\n","category":"module"},{"location":"#TeukolskyQNMFunctions.F-Tuple{Int64, Int64, Int64, Int64, Int64, Float64, ComplexF64}","page":"Home","title":"TeukolskyQNMFunctions.F","text":"F(\n  nr::myI,\n  nl::myI,\n  s::myI,\n  l::myI,\n  m::myI,\n  a::myF,\n  om::myC\n)::myF\n\nAbsolute difference of Lambda seperation constant for radial and angular ODEs.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.compute_om-Tuple{Int64, Int64, Int64, Int64, Int64, Float64, ComplexF64}","page":"Home","title":"TeukolskyQNMFunctions.compute_om","text":"compute_om(\n  nr::myI,\n  nl::myI,\n  s::myI,\n  l::myI,\n  m::myI,\n  a::myF,\n  om::myC; \n  tolerance::myF=tomyF(1e-6),\n  epsilon::myF=tomyF(1e-6),\n  gamma::myF=tomyF(1),\n  verbose::Bool=false\n\n)::Tuple{             myC,             myC,             Vector{myC},             Vector{myC},             Vector{myF}}\n\nSearch for quasinormal mode frequency in the complex plane using Newton's method.\n\nArguments\n\nnr: number of radial Chebyshev collocation points\nnl: number of spherical harmonic terms\ns:  spin of the field in the Teukolsky equation\nl:  l angular number\nm:  m angular number\na:  black hole spin\nom: guess for the initial quasinormal mode\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"Modules = [TeukolskyQNMFunctions.CustomTypes]","category":"page"},{"location":"#TeukolskyQNMFunctions.CustomTypes","page":"Home","title":"TeukolskyQNMFunctions.CustomTypes","text":"Custom number types for extended-precision computations.\n\n\n\n\n\n","category":"module"},{"location":"","page":"Home","title":"Home","text":"Modules = [TeukolskyQNMFunctions.Spheroidal]","category":"page"},{"location":"#TeukolskyQNMFunctions.Spheroidal","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal","text":"Methods to compute spin-weighted spheroidal harmonics\n\n\n\n\n\n","category":"module"},{"location":"#TeukolskyQNMFunctions.Spheroidal.compute_A-Tuple{Int64, Int64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.compute_A","text":"compute_A(s::myI, l::myI, m::myI)\n\nSee the note. \n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.compute_B-Tuple{Int64, Int64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.compute_B","text":"compute_B(s::myI, l::myI, m::myI)\n\nSee the note. \n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.compute_C-Tuple{Int64, Int64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.compute_C","text":"compute_C(s::myI, l::myI, m::myI)\n\nSee the note.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.compute_M_matrix-Tuple{Int64, Int64, Int64, ComplexF64}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.compute_M_matrix","text":"compute_M_matrix(\n  nl::myI,\n  s::myI,\n  m::myI,\n  c::myC\n)\n\nM matrix for computing spheroidal-spherical mixing and seperation coefficients.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.compute_l_min-Tuple{Any, Any}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.compute_l_min","text":"compute_l_min(s,m)\n\nMinimum l value\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.eig_vals_vecs-Tuple{Int64, Int64, Int64, Int64, ComplexF64}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.eig_vals_vecs","text":"eig_vals_vecs(\n  nl::myI,\n  neig::myI,\n  s::myI,\n  m::myI,\n  c::myC\n)\n\nCompute eigenvectors and eigenvalues for the spheroidal equation.   Returns eigenvalues λs (smallest to largest), and eigenvectors as an array (access nth eigenvector through v[:,n]); i.e. returns (λs, vs)\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.mat_L-Tuple{Int64, Int64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.mat_L","text":"mat_L(\n  nl::myI,\n  s::myI,\n  m::myI,\n)::SparseMatrixCSC\n\nReturns the matrix for spherical laplacian (s=0) in spectral space.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.mat_Y-Tuple{Int64, Int64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.mat_Y","text":"mat_Y(\n  nl::myI,\n  s::myI,\n  m::myI,\n)::SparseMatrixCSC\n\nReturns the matrix for y in spectral space.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"Modules = [TeukolskyQNMFunctions.RadialODE]","category":"page"},{"location":"#TeukolskyQNMFunctions.RadialODE","page":"Home","title":"TeukolskyQNMFunctions.RadialODE","text":"Ordinary differential equation in radial direction for the hyperboloidally compactified Teukolsky equation.\n\n\n\n\n\n","category":"module"},{"location":"#TeukolskyQNMFunctions.RadialODE.eig_vals_vecs_c-Tuple{Int64, Int64, Int64, Float64, ComplexF64}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.eig_vals_vecs_c","text":"eig_vals_vecs_c(\n  nr::myI,\n  s::myI,\n  m::myI,\n  a::myF,\n  om::myC\n)\n\nCompute eigenvectors and eigenvalues for the radial equation using a pseudospectral Chebyshev polynomial method. The black hole mass is always one.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.radial_discretized_eqn-Tuple{Int64, Int64, Int64, Float64, Float64, ComplexF64, Float64, Float64}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.radial_discretized_eqn","text":"radial_discretized_eqn(\n  nr::myI,\n  s::myI,\n  m::myI,\n  a::myF,\n  bhm::myF,\n  om::myC,\n  rmin::myF,\n  rmax::myF\n)\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.radial_discretized_eqn_p-Tuple{Int64, Int64, Int64, Float64, Float64, ComplexF64}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.radial_discretized_eqn_p","text":"radial_discretized_eqn_p(\n  np::myI,\n  s::myI,\n  m::myI,\n  a::myF,\n  bhm::myF,\n  om::myC\n)\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"Modules = [TeukolskyQNMFunctions.RadialODE.Chebyshev]","category":"page"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev","text":"Methods for position space Chebyshev methods. \n\n\n\n\n\n","category":"module"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.cheb_pts-Tuple{Float64, Float64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.cheb_pts","text":"cheb_pts(xmin::myF, xmax::myF, nx::myI)::Vector{myF}\n\nComputes Chebyshev points on interval [xmin,xmax] \n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.cheb_pts-Tuple{Int64}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.cheb_pts","text":"cheb_pts(nx::myI)::Vector{myF}\n\nComputes Generates nx Chebyshev points in [-1,1]\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_D1-Tuple{Float64, Float64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_D1","text":"mat_D1(\n  xmin::myF,\n  xmax::myF,\n  nx::myI\n)::Matrix{myF}\n\nComputes derivative matrix D1 in real space.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_D2-Tuple{Float64, Float64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_D2","text":"mat_D2(\n  xmin::myF,\n  xmax::myF,\n  nx::myI\n)::Matrix{myF}\n\nComputes derivative matrix D1 in real space.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_X-Tuple{Float64, Float64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_X","text":"mat_X(\n  xmin::myF,\n  xmax::myF,\n  nx::myI\n)::SparseMatrixCSC{myF, myI}\n\nComputes matrix for multiplication of x in real space. \n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_fd_D1-Tuple{Float64, Float64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_fd_D1","text":"mat_fd_D1(\n  xmin::myF,\n  xmax::myF,\n  nx::myI\n)::Matrix{myF}\n\nCompute 2nd order finite difference in Chebyshev points.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_fd_D2-Tuple{Float64, Float64, Int64}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_fd_D2","text":"mat_fd_D2(\n  xmin::myF,\n  xmax::myF,\n  nx::myI\n)::Matrix{myF}\n\nCompute 2nd order finite difference in Chebyshev points.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.to_cheb-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Number","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.to_cheb","text":"to_cheb(f::Vector{T})::Vector{T} where T<:Number\n\nConvert to Chebyshev space. We assume we are working with Chebyshev-Gauss-Lobatto points.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.to_real-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Number","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.to_real","text":"to_real(c::Vector{T})::Vector{T} where T<:Number\n\nConvert to Real space. We assume we are working with Chebyshev-Gauss-Lobatto points.\n\n\n\n\n\n","category":"method"}]
}
