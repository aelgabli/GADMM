function out=opt_sol_logistic(XX,YY, dim, lambda, num_workers)
s = dim;

%sym x(dim,1)
%sym('x',[dim 1])
%temp=(YY./(1+exp(YY.*(XX*ones(dim,1)))))
%temp2=-XX'*temp
%eqn = -(XX'*(YY./(1+exp(YY.*(XX*x)))))+lambda*x == zeros(dim,1);
%solution2 = solve(eqn)
%out2 =num_workers*lambda*0.5*sum_square(x)+sum(log(1+exp(-YY.*(XX*x))));


cvx_begin %quiet
%cvx_precision %low%high%low
%cvx_solver SDPT3%mosek%SeDuMi%mosek%SDPT3
variable x(s)

    obj=num_workers*lambda*0.5*sum_square(x)+sum(log(1+exp(-YY.*(XX*x))));

minimize(obj)
cvx_end

%solution1=x
out =num_workers*lambda*0.5*sum_square(x)+sum(log(1+exp(-YY.*(XX*x))));



end

