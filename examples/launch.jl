

include("GenerateFiles.jl")
using .GenerateFiles

println("generating qnm files")

nr=80
nl=24
s=-2
m=0
l=2
n=0
avals=[0.0,0.5,0.7,0.9]
qnm_tolerance=1e-6
coef_tolerance=1e-6
epsilon=1e-6

generate("",nr,nl,s,m,l,n,avals,qnm_tolerance,coef_tolerance,epsilon)
