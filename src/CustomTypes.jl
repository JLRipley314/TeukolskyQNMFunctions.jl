"""
Custom number types for extended-precision computations.
"""
module CustomTypes

export myI, myF, myC, tomyI, tomyF, tomyC

#------------------------------------
const myI = Int64
const myF = Float64
const myC = ComplexF64
#------------------------------------
#setprecision(1024)
#const myI = Int64 
#const myF = BigFloat 
#const myC = Complex{BigFloat}
#------------------------------------

function tomyI(v)::myI
   return convert(myI,parse(myF,"$v"))
end

function tomyF(v)::myF
   return parse(myF,"$v")
end

function tomyC(v)::myC
   return parse(myC,"$v")
end

end
