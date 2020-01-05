function x=logReg_GD(X,Y, x, C1, C2, term1, term2,lambda_logistic, gdStep)

for j=1:100
%     for i=1:size(X,1)
%         mu(i,1)=1/(1+exp(X(i,:)*x));
%     end
    %mu=mu';
    %temp=(Y./(1+exp(Y.*(X*x))));
    g=-(X'*(Y./(1+exp(Y.*(X*x)))))+lambda_logistic*x-C1+C2+term1+term2;
    %g=X*(mu-Y)+lambda_logistic*x-C1+C2+term1+term2;
    %S=diag(mu);
    %H=X*S*X'+diag(lambda_logistic)+diag(2*rho);

    %x=x-H\g;

    x_prev=x;

    
    x=x-gdStep*g;
    
    if(abs(x-x_prev) < 1E-4)
        break;
    end

end
    


%kkk=1;
end

%lambda_logistic*0.5*sum_square(x)+sum(log(1+exp(-YY(first:last).*(XX(first:last,1:s1)*x))))- C1'*x+C2'*x + term_1...
           % +term_2;
