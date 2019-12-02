function [obj_GADMM, loss_GADMM, Iter]=group_ADMM(XX,YY, rho, no_workers, num_feature, noSamples, num_iter, obj0, acc)
Iter= num_iter;           

s1=num_feature;
s2=noSamples;


lambda = zeros(s1,no_workers);

out=zeros(s1,no_workers);

max_iter = num_iter;
 for i = 1:max_iter
     
     for ii =1:2:no_workers
         cvx_begin quiet 
     %cvx_precision high
     %cvx_solver  mosek%SDPT3 %mosek
     variable x(s1)
     
         if ii==1
             C1 = zeros(s1,1);
             term_1 =0;
         else
             C1= lambda(:,ii-1);
             term_1=rho/2*sum_square(x-out(:,ii-1));
         end
         
         if ii == no_workers
             C2 = zeros(s1,1);
             term_2 = 0;
         else
             C2= lambda(:,ii);
             term_2 = rho/2*sum_square(x-out(:,ii+1));
             
         end
         first = (ii-1)*s2+1;
         last = first+s2-1;
        
        
        %cvx_solver mosek
        
        minimize(0.5*sum_square(XX(first:last,1:s1)*x - YY(first:last))- C1'*x+C2'*x + term_1...
            +term_2)
         cvx_end
        out(:,ii) =x;
        
     end
    
    
        
     for ii =2:2:no_workers
          cvx_begin quiet 
        %cvx_precision high
        %cvx_solver mosek%SDPT3 %mosek
        variable x(s1)
         C1= lambda(:,ii-1);
         if ii == no_workers
             C2 = zeros(s1,1);
             term_2 = 0;
         else
             C2= lambda(:,ii);
             term_2 = rho/2*sum_square(x-out(:,ii+1));
         end
         term_1=rho/2*sum_square(x-out(:,ii-1));
         first = (ii-1)*s2+1;
         last = first+s2-1;
         
        
        minimize(0.5*sum_square(XX(first:last,1:s1)*x - YY(first:last))- C1'*x+C2'*x + term_1...
            +term_2)
        cvx_end
        out(:,ii) =x;
     end
     
        for ii=1:no_workers-1
            lambda(:,ii) = lambda(:,ii) + rho*(out(:,ii)-out(:,ii+1));
        end
        
        
        final_obj = 0;
        for ii =1:no_workers
            first = (ii-1)*s2+1;
            last = first+s2-1;
            final_obj = final_obj + 0.5*sum_square(XX(first:last,1:s1)*out(:,ii) - YY(first:last));
        end
        obj_GADMM(i)=final_obj;
        loss_GADMM(i)=abs(final_obj-obj0);
        
        if(loss_GADMM(i) < acc)
            Iter = i;
            break;
        end
       
    

end
     




