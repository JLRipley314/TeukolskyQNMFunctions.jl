include("QNMFileUtils.jl")
import .QNMFileUtils as QF

println("generating qnm files")

T = Float64

nr = 10 
nl = 8 
s = -2
m = 0
l = 2
avals = [T(0.0)]
qnm_tolerance = T(1e-6)
coef_tolerance = T(1e-6)
epsilon = T(1e-6)

n = 0
QF.generate("", nr, nl, s, m, l, n, avals, qnm_tolerance, coef_tolerance, epsilon)
n = 1
QF.generate("", nr, nl, s, m, l, n, avals, qnm_tolerance, coef_tolerance, epsilon)
n = 2
QF.generate("", nr, nl, s, m, l, n, avals, qnm_tolerance, coef_tolerance, epsilon)
