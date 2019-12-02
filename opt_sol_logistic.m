function out=opt_sol_logistic(XX,YY, size, lambda, num_workers)
s = size;
cvx_begin %quiet
cvx_precision low%high%low
cvx_solver SDPT3%mosek%SeDuMi%mosek%SDPT3
variable x(s)

    obj=num_workers*lambda*0.5*norm(x)^2+sum(log(1+exp(-YY.*(XX*x))));

minimize(obj)
cvx_end


out =num_workers*lambda*0.5*norm(x)^2+sum(log(1+exp(-YY.*(XX*x))));

