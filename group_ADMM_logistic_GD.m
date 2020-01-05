function [obj_GADMM, loss_GADMM, Iter, gadmm_time]=group_ADMM_logistic_GD(XX,YY, rho, no_workers, num_feature,...
    per_samples, num_iter, obj0, lambda_logistic, acc, gdStep)

s1=num_feature;
s2=per_samples;
Iter = num_iter;

lambda = zeros(s1,no_workers);

%x1 = zeros(50,1);
%x2 = zeros(50,1);
out=zeros(s1,no_workers);
flagg =0;
max_iter = num_iter;
gadmm_time(1)=0;
 for i = 1:max_iter
     %i
     for ii =1:2:no_workers
         %cvx_begin quiet
         %cvx_solver SDPT3%mosek
     %cvx_precision low%high%low
     %variable x(s1)
     
         if ii==1
             C1 = zeros(s1,1);
             term_1 =zeros(s1,1);
         else
             C1= lambda(:,ii-1);
             %term_1=rho/2*sum_square(x-out(:,ii-1));
             term_1=rho*(out(:,ii)-out(:,ii-1));
         end
         
         if ii == no_workers
             C2 = zeros(s1,1);
             term_2 = zeros(s1,1);
         else
             C2= lambda(:,ii);
             %term_2 = rho/2*sum_square(x-out(:,ii+1));
             term_2=rho*(out(:,ii)-out(:,ii+1));
             
         end
         first = (ii-1)*s2+1;
         last = first+s2-1;
        
         %syms a b c x
         %eqn = a*x^2 + b*x + c == 0
         if(ii==1 && i > 1)
                tic
         end

         out(:,ii)=logReg_GD(XX(first:last,1:s1),YY(first:last), out(:,ii),...
             C1, C2, term_1, term_2,lambda_logistic, gdStep);
         
        %objFun=lambda_logistic*0.5*sum_square(x)+sum(log(1+exp(-YY(first:last).*(XX(first:last,1:s1)*x))))- C1'*x+C2'*x + term_1...
            %+term_2;
        %minimize(0.5*sum_square(XX(first:last,1:s)*x - YY(first:last))- C1'*x+C2'*x + term_1+term_2)
        
            %minimize(objFun)
         %cvx_end
         
        if(ii==1 && i > 1)
            gadmm_time(i)=gadmm_time(i-1)+2*toc;
        end
        %out(:,ii) =x;
        
     end
    
    
        
     for ii =2:2:no_workers
          %cvx_begin quiet 
        %cvx_precision low %high%low
        %cvx_solver SDPT3%mosek
        %variable x(s1)
         C1= lambda(:,ii-1);
         if ii == no_workers
             C2 = zeros(s1,1);
             term_2 = zeros(s1,1);
         else
             C2= lambda(:,ii);
             %term_2 = rho/2*sum_square(x-out(:,ii+1));
             term_2=rho*(out(:,ii)-out(:,ii+1));
         end
         %term_1=rho/2*sum_square(x-out(:,ii-1));
         term_1=rho*(out(:,ii)-out(:,ii-1));
         first = (ii-1)*s2+1;
         last = first+s2-1;
         
         out(:,ii)=logReg_GD(XX(first:last,1:s1),YY(first:last), out(:,ii),...
             C1, C2, term_1, term_2,lambda_logistic, gdStep);
         
       % objFun=lambda_logistic*0.5*sum_square(x)+sum(log(1+exp(-YY(first:last).*(XX(first:last,1:s1)*x))))- C1'*x+C2'*x + term_1...
            %+term_2;
        %minimize(objFun)
        %cvx_end
        %out(:,ii) =x;
     end
     
        for ii=1:no_workers-1
            lambda(:,ii) = lambda(:,ii) + rho*(out(:,ii)-out(:,ii+1));
            %lambda_2(:,ii) = lambda_2(:,ii) + rho*(out(:,ii)-out(:,ii+1));
            %gap(ii)=norm(out(:,ii)-out(:,ii+1))
        end
        
%         for ii=2:no_workers-1
%             lambda_2(:,ii) = lambda_2(:,ii) + rho*(out(:,ii)-out(:,ii+1));
%         end
        
        final_obj = 0;
        for ii =1:no_workers
            first = (ii-1)*s2+1;
            last = first+s2-1;
            %final_obj = final_obj + 0.5*sum_square(XX(first:last,1:s)*out(:,ii) - YY(first:last));
            final_obj = final_obj+lambda_logistic*0.5*norm(out(:,ii))^2+sum(log(1+exp(-YY(first:last).*(XX(first:last,1:s1)*out(:,ii)))));
        end
        i
        obj_GADMM(i)=final_obj;
        loss_GADMM(i)=abs(final_obj-obj0)
       
        
        if(loss_GADMM(i) < acc)
            Iter = i;
            break;
        end
    

end
     




