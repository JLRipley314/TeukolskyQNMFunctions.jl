module ReadQNM

export qnm

include("../src/CustomTypes.jl")
using .CustomTypes

export qnm

function eq(x, y)
    return abs(x - y) < 1e-14 ? true : false
end

"""
    qnm(n::myI,s::myI,m::myI,l::myI,a::myF)

Returns omega, lambda
"""
function qnm(n::myI, s::myI, m::myI, l::myI, a::myF)

    for line in eachline("qnmvals.txt")
        line = split(line, ",")
        nl = parse(myF, line[1])
        sl = parse(myF, line[2])
        ml = parse(myF, line[3])
        ll = parse(myF, line[4])
        al = parse(myF, line[5])

        if eq(n, nl) && eq(s, sl) && eq(m, ml) && eq(l, ll) && eq(a, al)
            return parse(myF, line[6]) + im * parse(myF, line[7]),
            parse(myF, line[8]) + im * parse(myF, line[9])
        end
    end
    return NaN, NaN
end

end
