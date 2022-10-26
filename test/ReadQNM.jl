module ReadQNM

export qnm

function eq(x, y)
    return abs(x - y) < 1e-14 ? true : false
end

"""
    qnm(n::Integer,s::Integer,m::Integer,l::Integer,a::Real,T::Type{<:Real}=Float64)

Returns omega, lambda
"""
function qnm(n::Integer, s::Integer, m::Integer, l::Integer, a::Real, T::Type{<:Real}=Float64)

    for line in eachline("qnmvals.txt")
        line = split(line, ",")
        nl = parse(T, line[1])
        sl = parse(T, line[2])
        ml = parse(T, line[3])
        ll = parse(T, line[4])
        al = parse(T, line[5])

        if eq(n, nl) && eq(s, sl) && eq(m, ml) && eq(l, ll) && eq(a, al)
            return parse(T, line[6]) + im * parse(T, line[7]),
            parse(T, line[8]) + im * parse(T, line[9])
        end
    end
    return NaN, NaN
end

end
