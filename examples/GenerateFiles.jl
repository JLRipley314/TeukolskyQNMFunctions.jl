module GenerateFiles

include("QNMFileUtils.jl")
import .QNMFileUtils as QF 

export generate

function generate(
      prename::String,
      nr::Integer,
      nl::Integer,
      s::Integer,
      m::Integer,
      l::Integer,
      n::Integer,
      avals::Vector{<:Real},
      qnm_tolerance::Real,
      coef_tolerance::Real,
      epsilon::Real)
   #println("bash -c 'rm -f $(pwd)/HigherPrecData/*.h5'")
   #run(`bash -c 'rm -f $(pwd)/HigherPrecData/*.h5'`)

   lmin = max(abs(s),abs(m))
   for l=[lmin,lmin+1,lmin+2]
      try 
         println("s=$s,m=$m,n=$n,l=$l")
            QF.generate_data(
               "$(pwd())/qnmfiles/$(prename)s$(s)_m$(m)_n$(n)",
               nr, nl, s, n, l, m,
               [a for a in avals],
               qnm_tolerance=qnm_tolerance,
               coef_tolerance=coef_tolerance,
               epsilon=epsilon
            )
      catch err
         println(err)
         continue
      end
   end
   return nothing
end

end
