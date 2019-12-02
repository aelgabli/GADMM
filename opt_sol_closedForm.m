function out=opt_sol_closedForm(XX,YY)
x = (XX'*XX)\(XX'*YY);
out =0.5*sum_square(XX*x - YY);
