module TestQuasinormalModes

include("../src/CustomTypes.jl")
using CustomTypes

const tolerance = 1e-4 ## tolerance we compare to

##============================================================
import QuasinormalModes as QNM

nr = 48
nl = 16

avals = [0.0, 0.354, 0.7, 0.99] ## same as in generate.py

for s=[-2,-1,0]
   for m=[-2,0,3]
      lmin = max(abs(s),abs(m))
      for l=[lmin, lmin+1, lmin+4]
         for n=[0,1,2]
            for a in avals 
               om, la qnm(n,s,m,l,a)
               println("s=$s, (n=$n,l=$l,m=$m), a=$a, ω=$om, Λ=$la")
                  
               f_om, f_la, f_vs, f_vr, rvals = QNM.compute_om(
                                         myI(nr), 
                                         myI(nl), 
                                         myI(s), 
                                         myI(l), 
                                         myI(m), 
                                         myF(a), 
                                         myC(om+0.1*rand(myC)),
                                         verbose=false
                                        )
               @test abs(f_om-om)/max(1,abs(om)) < tolerance
            end
         end
      end
   end
end

end
