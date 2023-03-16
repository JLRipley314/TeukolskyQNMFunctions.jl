# TeukolskyQNMFunctions.jl

<!--- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JLRipley314.github.io/TeukolskyQNMFunctions.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JLRipley314.github.io/TeukolskyQNMFunctions.jl/dev)
[![Build Status](https://github.com/JLRipley314/TeukolskyQNMFunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JLRipley314/TeukolskyQNMFunctions.jl/actions/workflows/CI.yml?query=branch%3Amain)
<!--- [![Coverage](https://codecov.io/gh/JLRipley314/TeukolskyQNMFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JLRipley314/TeukolskyQNMFunctions.jl) -->

`TeukolskyQNMFunctions` computes the quasinormal modes and eigenfunctions 
for the spin s Teukolsky equation
using a horizon penetrating, hyperboloidally compactified coordinate system.
The main advantage of using these coordinates is that the quasinormal
wavefunctions are finite valued from the black hole to future null infinity.

# Discretization scheme overview

The original version of this code used a Chebyshev pseudospectral method to compute
the radial part of the eigenfunctions, 
and a spectral method to compute the angular part of the eigenfunctions
The angular spectral method was originally introduced by 
[Cook and Zalutskiy](https://arxiv.org/abs/1410.7698)).

The current version of the code makes use of a fully spectral method to
discretize in the radial direction.
This method was originally described in (see also the bibtex entry at the end) 
```
Olver, Sheehan, and Alex Townsend. 
"A fast and well-conditioned spectral method." 
siam REVIEW 55.3 (2013): 462-489.
```

We make use of the 
[ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl) package to
perform the new, spectral discretization.   

# Documentation

The note [arXiv:2202.03837](https://arxiv.org/abs/2202.03837) 
explains the main ideas that went into this code,
and derives the relevant equations of motion.

See also the `examples` directory.

# Warning!

This code does *not* compute quasinormal modes nearly as quickly 
(nor at a given numerical precision as accurately) as other established
quasinormal mode codes, e.g. 
Leo Stein's [qnm code](https://github.com/duetosymmetry/qnm).
If you just want to compute a quasinormal mode, I suggest using that code,
or looking at the publicly available qnm tables, e.g. 
Emanuele Berti's [tables of qnm](https://pages.jh.edu/eberti2/ringdown/).

# Computing overtones

The original version of this code (which made use of a pseudo-spectral discretization
in the radial direction) had some issues computing higher overtones,
since radial points were needed to resolve those modes. 
This likely because Chebyshev
pseudospectral operators are not very well conditioned.

The current version of the code makes use of a fully spectral method via
the ApproxFun package (see above), and can now resolve higher overtones.
See `examples/example_compute_qnm.jl` for example overtone calculations. 

# How to cite

If you end up using this code in a publication, or some of the ideas in
[arXiv:2202.03837](https://arxiv.org/abs/2202.03837), 
which describes the ideas that went into this code, please cite
```
@article{Ripley:2022ypi,
    author = "Ripley, Justin L.",
    title = "{Computing the quasinormal modes and eigenfunctions for the Teukolsky equation using horizon penetrating, hyperboloidally compactified coordinates}",
    eprint = "2202.03837",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1088/1361-6382/ac776d",
    journal = "Class. Quant. Grav.",
    volume = "39",
    number = "14",
    pages = "145009",
    year = "2022"
}
```

If you use the newest version of this code that makes use of the ApproxFun.jl
package, I suggest you cite that package as well, along with 
```
@article{olver2013fast,
  author="Olver, Sheehan and Townsend, Alex",
  title="{A fast and well-conditioned spectral method}",
  eprint = "1202.1347",
  archivePrefix = "arXiv",
  primaryClass = "math.NA",
  doi= "https://doi.org/10.1137/120865458", 
  journal="siam REVIEW",
  volume="55",
  number="3",
  pages="462--489",
  year="2013",
  publisher="SIAM"
}
```

# Contribution/Contact 

If you would like to add a feature, fix a bug, etc, 
you can open a pull request, or email me (Justin) at

ripley [at] illinois [dot] edu
