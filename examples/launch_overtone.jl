include("OvertoneFileUtils.jl")
import .OvertoneFileUtils as OT

println("generating qnm files")

T = Float64

nr = 36
nl = 20
s = -2
m = 2
l = 2
n = 0
avals = [T(0.0)]
qnm_tolerance = T(1e-10)
coef_tolerance = T(1e-10)
epsilon = T(1e-10)

OT.generate("", nr, nl, s, m, l, n, avals, qnm_tolerance, coef_tolerance, epsilon)
