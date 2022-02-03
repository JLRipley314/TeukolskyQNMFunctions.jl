# TeukolskyQNMFunctions

Documentation for [TeukolskyQNMFunctions](https://github.com/JLRipley314/TeukolskyQNMFunctions.jl).

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

```@meta
CurrentModule = TeukolskyQNMFunctions
```

```@index
```

```@autodocs
Modules = [TeukolskyQNMFunctions]
```

```@autodocs
Modules = [TeukolskyQNMFunctions.CustomTypes]
```

```@autodocs
Modules = [TeukolskyQNMFunctions.Spheroidal]
```

```@autodocs
Modules = [TeukolskyQNMFunctions.RadialODE]
```

```@autodocs
Modules = [TeukolskyQNMFunctions.RadialODE.Chebyshev]
```
