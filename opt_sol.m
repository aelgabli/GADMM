function out=opt_sol(XX,YY, size)
s = size;
cvx_begin %quiet
%cvx_precision high
%cvx_solver mosek
variable x(s)
minimize(0.5*sum_square(XX*x - YY))
cvx_end
out =0.5*sum_square(XX*x - YY);
