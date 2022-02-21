module GenerateFiles

include("../src/CustomTypes.jl")
include("QNMFileUtils.jl")
using  .CustomTypes
import .QNMFileUtils as QF 

export generate

function generate(
      prename::String,
      nr::myI,
      nl::myI,
      s::myI,
      m::myI,
      l::myI,
      n::myI,
      avals::Vector{myF},
      qnm_tolerance::myF,
      coef_tolerance::myF,
      epsilon::myF)
   #println("bash -c 'rm -f $(pwd)/HigherPrecData/*.h5'")
   #run(`bash -c 'rm -f $(pwd)/HigherPrecData/*.h5'`)

   lmin = max(abs(s),abs(m))
   for l=[lmin,lmin+1,lmin+2]
      try 
         println("s=$s,m=$m,n=$n,l=$l")
            QF.generate_data(
               "$(pwd())/qnmfiles/$(prename)s$(s)_m$(m)_n$(n)",
               tomyI(nr), tomyI(nl), 
               tomyI(s), tomyI(n), tomyI(l), tomyI(m),
               [tomyF(a) for a in avals],
               qnm_tolerance=tomyF(qnm_tolerance),
               coef_tolerance=tomyF(coef_tolerance),
               epsilon=tomyF(epsilon)
            )
      catch err
         println(err)
         continue
      end
   end
   return nothing
end

end
