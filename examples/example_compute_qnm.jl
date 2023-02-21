include("QNMFileUtils.jl")
import .QNMFileUtils as QF

println("generating qnm files")

T = Float64

#====================#

nr = 32 
nl = 8 
s = -2
m = 2
l = 2
avals = [T(0.0), T(0.7), T(0.99)]
qnm_tolerance = T(1e-6)
coef_tolerance = T(1e-6)
epsilon = T(1e-6)

n = 0
QF.generate("", nr, nl, s, m, l, n, avals, qnm_tolerance, coef_tolerance, epsilon)
n = 1
QF.generate("", nr, nl, s, m, l, n, avals, qnm_tolerance, coef_tolerance, epsilon)
n = 2
QF.generate("", nr, nl, s, m, l, n, avals, qnm_tolerance, coef_tolerance, epsilon)

#====================#

nr = 32 
nl = 8 
s = -2
m = 3
l = 3
avals = [T(0.0), T(0.7), T(0.99)]
qnm_tolerance = T(1e-6)
coef_tolerance = T(1e-6)
epsilon = T(1e-6)

n = 0
QF.generate("", nr, nl, s, m, l, n, avals, qnm_tolerance, coef_tolerance, epsilon)
n = 1
QF.generate("", nr, nl, s, m, l, n, avals, qnm_tolerance, coef_tolerance, epsilon)
n = 2
QF.generate("", nr, nl, s, m, l, n, avals, qnm_tolerance, coef_tolerance, epsilon)
