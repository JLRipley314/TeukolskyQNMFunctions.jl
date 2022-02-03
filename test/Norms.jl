module Norms

function one_norm(v::Array{Number,1})
   nx = size(v)[1]
   n = 0.0
   for i=1:nx
      n += abs(v[i])
   end
   return n/nx
end

function one_norm(v::Array{Number,2})
   nx, ny = size(v)
   n = 0.0
   for j=1:ny
      for i=1:nx
         n += abs(v[i,j])
      end
   end
   return n/(nx*ny)
end

function total_variation(v::Array{Number,1})
   nx = size(v)[1]
   tv = 0.0
   for i=1:nx-1
      tv += abs(v[i+1]-v[i])
   end
   return tv
end

function total_variation(v::Array{Number,2})
   nx, ny = size(v)
   tv = 0.0
   for j=1:ny-1
      for i=1:nx-1
         tv += abs(v[i+1,j]-v[i,j]) + abs(v[i,j+1]-v[i,j])
      end
   end
   return tv
end

end
