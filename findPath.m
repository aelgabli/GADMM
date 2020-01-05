function [path, pathCost, d_square]=findPath(N)

for n=1:N
  x(n)=50*rand;
  y(n)=50*rand;
end

for n=1:N
    for m=1:N
        if(n~=m)
            d_square(n,m)=(x(n)-x(m))^2+(y(n)-y(m))^2;
        end
    end
end
    
path=[1];
pathCost=[];
while(length(path) < N)
    n=path(length(path));
    min=1E12;
    for m=1:N
        temp=d_square(n,m);
        if(m~=n && ~ismember(m,path) && temp < min)
            min=temp;
            min_idx=m;
        end
    end
    path=[path, min_idx];
    pathCost=[pathCost, min];
end

%kkk=1;

        