# TeukolskyQNMFunctions.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JLRipley314.github.io/TeukolskyQNMFunctions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JLRipley314.github.io/TeukolskyQNMFunctions.jl/dev)
[![Build Status](https://github.com/JLRipley314/TeukolskyQNMFunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JLRipley314/TeukolskyQNMFunctions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JLRipley314/TeukolskyQNMFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JLRipley314/TeukolskyQNMFunctions.jl)

`TeukolskyQNMFunctions` computes the quasinormal modes and eigenfunctions 
for the spin s Teukolsky equation
using a horizon penetrating, hyperboloidally compactified coordinate system.
The main advantage of using these coordinates is that the quasinormal
wavefunctions are finite valued from the black hole to future null infinity.

Currently, this code uses a Chebyshev pseudospectral method to compute
the radial part of the eigenfunctions, 
and a spectral method to compute the angular part of the eigenfunctions
The angular spectral method was originally introduced by 
[Cook and Zalutskiy](https://arxiv.org/abs/1410.7698)).

# Documentation

The note [arXiv:2202.03837](https://arxiv.org/abs/2202.03837) 
explains the main ideas that went into this code,
and derives the relevant equations of motion.
See also the docstrings to the functions in the source code.

# Warning!

This code does *not* compute quasinormal modes nearly as quickly 
(nor at a given numerical precision as accurately) as other established
quasinormal mode codes, e.g. 
Leo Stein's [qnm code](https://github.com/duetosymmetry/qnm).
If you just want to compute a quasinormal mode, I suggest using that code,
or looking at the publicly available qnm tables, e.g. 
Emanuele Berti's [tables of qnm](https://pages.jh.edu/eberti2/ringdown/).

# To Do

* Add spectral (~Leaver) solver for radial equation, which will likely be
  more stable than the Chebyshev pseudospectral solver. 

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
    month = "2",
    year = "2022"
}
```
