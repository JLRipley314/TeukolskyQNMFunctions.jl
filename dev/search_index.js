var documenterSearchIndex = {"docs":
[{"location":"#TeukolskyQNMFunctions","page":"Home","title":"TeukolskyQNMFunctions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for TeukolskyQNMFunctions.","category":"page"},{"location":"","page":"Home","title":"Home","text":"TeukolskyQNMFunctions computes the quasinormal modes and eigenfunctions  for the spin s Teukolsky equation using a horizon penetrating, hyperboloidally compactified coordinate system. The main advantage of using these coordinates is that the quasinormal wavefunctions are finite valued from the black hole to future null infinity.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Currently, this code uses a Chebyshev pseudospectral method to compute the radial part of the eigenfunctions,  and a spectral method to compute the angular part of the eigenfunctions The angular spectral method was originally introduced by  Cook and Zalutskiy).","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TeukolskyQNMFunctions","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TeukolskyQNMFunctions]","category":"page"},{"location":"#TeukolskyQNMFunctions.TeukolskyQNMFunctions","page":"Home","title":"TeukolskyQNMFunctions.TeukolskyQNMFunctions","text":"TeukolskyQNMFunctions.jl computes the quasinormal modes and eigenfunctions  for the spin s Teukolsky equation using a horizon penetrating, hyperboloidally compactified coordinate system. The main advantage of using these coordinates is that the quasinormal wavefunctions are finite valued from the black hole to future null infinity.\n\n\n\n\n\n","category":"module"},{"location":"#TeukolskyQNMFunctions.F-Union{Tuple{T}, Tuple{Integer, Integer, Integer, Integer, Integer, T, Complex{T}}} where T<:Real","page":"Home","title":"TeukolskyQNMFunctions.F","text":"function F(\n    nr::Integer,\n    nl::Integer,\n    s::Integer,\n    l::Integer,\n    m::Integer,\n    a::T,\n    om::Complex{T},\n) where {T<:Real}\n\nCompute the absolute difference of Lambda seperation constant for radial and angular ODEs.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.compute_om-Union{Tuple{T}, Tuple{Integer, Integer, Integer, Integer, Integer, T, Complex{T}}} where T<:Real","page":"Home","title":"TeukolskyQNMFunctions.compute_om","text":"compute_om(\n    nr::Integer,\n    nl::Integer,\n    s::Integer,\n    l::Integer,\n    m::Integer,\n    a::Real,\n    om::Complex;\n    tolerance::Real = 1e6,\n    epsilon::Real = 1e-6,\n    gamma::Real = 1.0,\n    verbose::Bool = false\n) where T<:Real\n\nSearch for quasinormal mode frequency in the complex plane using Newton's method.\n\nArguments\n\nnr       : number of radial Chebyshev collocation points\nnl       : number of spherical harmonic terms\ns        : spin of the field in the Teukolsky equation\nl        : l angular number\nm        : m angular number\na        : black hole spin\nom       : guess for the initial quasinormal mode\ntolerance: tolerance for root finder\nepsilon  : derivative finite difference length\ngamma    : search gamma\nverbose  : true: print out intermediate results as searches for root \n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"Modules = [TeukolskyQNMFunctions.CustomTypes]","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TeukolskyQNMFunctions.Spheroidal]","category":"page"},{"location":"#TeukolskyQNMFunctions.Spheroidal","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal","text":"Methods to compute spin-weighted spheroidal harmonics\n\n\n\n\n\n","category":"module"},{"location":"#TeukolskyQNMFunctions.Spheroidal.compute_A-Tuple{Integer, Integer, Integer}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.compute_A","text":"compute_A(s::Integer, l::Integer, m::Integer)\n\nSee Appendix A of  J. Ripley, Class.Quant.Grav. 39 (2022) 14, 145009, Class.Quant.Grav. 39 (2022) 145009 arXiv:2202.03837\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.compute_B-Tuple{Integer, Integer, Integer}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.compute_B","text":"compute_B(s::Integer, l::Integer, m::Integer)\n\nSee Appendix A of  J. Ripley, Class.Quant.Grav. 39 (2022) 14, 145009, Class.Quant.Grav. 39 (2022) 145009 arXiv:2202.03837\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.compute_C-Tuple{Integer, Integer, Integer}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.compute_C","text":"compute_C(s::Integer, l::Integer, m::Integer)\n\nSee Appendix A of  J. Ripley, Class.Quant.Grav. 39 (2022) 14, 145009, Class.Quant.Grav. 39 (2022) 145009 arXiv:2202.03837\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.compute_M_matrix-Tuple{Integer, Integer, Integer, Complex}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.compute_M_matrix","text":"compute_M_matrix(nl::Integer, s::Integer, m::Integer, c::Complex)\n\nCompute the matrix for computing spheroidal-spherical mixing and seperation coefficients.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.compute_l_min-Tuple{Integer, Integer}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.compute_l_min","text":"compute_l_min(s::Integer, m::Integer)\n\nCompute the minimum l value\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.eig_vals_vecs-Tuple{Integer, Integer, Integer, Integer, Complex}","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.eig_vals_vecs","text":"eig_vals_vecs(nl::Integer, neig::Integer, s::Integer, m::Integer, c::Complex)\n\nCompute the eigenvectors and eigenvalues for the spheroidal equation.   Returns eigenvalues λs (smallest to largest), and eigenvectors as an array (access nth eigenvector through v[:,n]); i.e. returns (λs, vs)\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.mat_L-Union{Tuple{T}, Tuple{T, T, T}} where T<:Integer","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.mat_L","text":"function mat_L(nl::Integer, s::Integer, m::Integer)\n\nCompute the matrix for spherical laplacian (s=0) in spectral space.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.Spheroidal.mat_Y-Union{Tuple{T}, Tuple{T, T, T}} where T<:Integer","page":"Home","title":"TeukolskyQNMFunctions.Spheroidal.mat_Y","text":"mat_Y(nl::Integer, s::Integer, m::Integer)\n\nCompute the matrix for y in spectral space.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"Modules = [TeukolskyQNMFunctions.RadialODE]","category":"page"},{"location":"#TeukolskyQNMFunctions.RadialODE","page":"Home","title":"TeukolskyQNMFunctions.RadialODE","text":"Ordinary differential equation in radial direction for the hyperboloidally compactified Teukolsky equation.\n\n\n\n\n\n","category":"module"},{"location":"#TeukolskyQNMFunctions.RadialODE.eig_vals_vecs-Union{Tuple{T}, Tuple{Integer, Integer, Integer, T, Complex{T}}} where T<:Real","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.eig_vals_vecs","text":"eig_vals_vecs(\n        nr::Integer,\n        s::Integer,\n        m::Integer,\n        a::T,\n        om::Complex{T}\n    ) where T<:Real\n\nCompute eigenvectors and eigenvalues for the radial equation using a pseudospectral Chebyshev polynomial method. The black hole mass is always one.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.radial_discretized_eqn_a-Union{Tuple{T}, Tuple{Integer, Integer, Integer, T, T, Complex{T}, T, T}} where T<:Real","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.radial_discretized_eqn_a","text":"function radial_discretized_eqn_a(\n    nr::Integer,\n    s::Integer,\n    m::Integer,\n    a::T,\n    bhm::T,\n    om::Complex{T},\n    rmin::T,\n    rmax::T\n) where T<:Real\n\nDiscretization that uses fast, well conditioned method of\n\n    S. Olver and A. Townsend, \n    SIAM review, 2013,\n    arXiv:1202.1347\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.radial_discretized_eqn_c-Union{Tuple{T}, Tuple{Integer, Integer, Integer, T, T, Complex{T}, T, T}} where T<:Real","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.radial_discretized_eqn_c","text":"function radial_discretized_eqn_c(\n    nr::Integer,\n    s::Integer,\n    m::Integer,\n    a::T,\n    bhm::T,\n    om::Complex{T},\n    rmin::T,\n    rmax::T\n) where T<:Real\n\nDiscretization that uses Chebyshev pseudospectral method.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"Modules = [TeukolskyQNMFunctions.RadialODE.Chebyshev]","category":"page"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev","text":"Methods for position space Chebyshev methods. \n\n\n\n\n\n","category":"module"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.cheb_pts-Union{Tuple{T}, Tuple{T, T, Integer}} where T<:Real","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.cheb_pts","text":"cheb_pts(xmin::Real, xmax::Real, nx::Integer)\n\nComputes Chebyshev points on interval [xmin,xmax] \n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.cheb_pts-Union{Tuple{T}, Tuple{Type{T}, Integer}} where T<:AbstractFloat","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.cheb_pts","text":"cheb_pts([T,] nx::Integer)\n\nComputes Generates nx Chebyshev points in [-1,1]\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_D1-Union{Tuple{TI}, Tuple{TR}, Tuple{TR, TR, TI}} where {TR<:AbstractFloat, TI<:Integer}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_D1","text":"mat_D1(\n      xmin::Real,\n      xmax::Real,\n      nx::Integer,\n   )\n\nComputes derivative matrix D1 in real space.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_X-Union{Tuple{TI}, Tuple{TR}, Tuple{TR, TR, TI}} where {TR<:AbstractFloat, TI<:Integer}","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.mat_X","text":"mat_X(\n      xmin::Real,\n      xmax::Real,\n      nx::Integer\n   )\n\nComputes matrix for multiplication of x in real space. \n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.to_cheb-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Number","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.to_cheb","text":"to_cheb(f::Vector{<:Number})\n\nConvert to Chebyshev space. We assume we are working with Chebyshev-Gauss-Lobatto points.\n\n\n\n\n\n","category":"method"},{"location":"#TeukolskyQNMFunctions.RadialODE.Chebyshev.to_real-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Number","page":"Home","title":"TeukolskyQNMFunctions.RadialODE.Chebyshev.to_real","text":"to_real(c::Vector{<:Number})\n\nConvert to Real space. We assume we are working with Chebyshev-Gauss-Lobatto points.\n\n\n\n\n\n","category":"method"}]
}
