function pathCost_static = calc_cost(grid, path0)
N=length(path0);
for n=1:N-1
    n1=path0(n);
    n2=path0(n+1);
    pathCost_static(n)=grid(n1,n2);
end
    

end


        