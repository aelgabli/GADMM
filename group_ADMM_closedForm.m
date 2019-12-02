function [obj_GADMM, loss_GADMM, Iter, gadmm_time]=group_ADMM_closedForm(XX,YY, rho, no_workers, num_feature, noSamples, num_iter, obj0, acc)
Iter= num_iter;   
           
s1=num_feature;
s2=noSamples;
lambda = zeros(s1,no_workers);
out=zeros(s1,no_workers);

max_iter = num_iter;
gadmm_time(1)=0;
 for i = 1:max_iter
     
     for ii =1:2:no_workers
         if ii==1
             C1 = zeros(s1,1);
             term_1 =zeros(s1,1);
         else
             C1= lambda(:,ii-1);
             term_1=rho*out(:,ii-1);
         end
         
         if ii == no_workers
             C2 = zeros(s1,1);
             term_2 = zeros(s1,1);
         else
             C2= lambda(:,ii);
             term_2 = rho*out(:,ii+1);
             
         end
         first = (ii-1)*s2+1;
         last = first+s2-1;
        
        H=XX(first:last,1:s1);
        Y=YY(first:last);
        %q=H'*Y+C1-C2+term_1+term_2;
        %[L, U] = factor(H, rho);
        %x=(H'*H+2*rho*eye(s1,s1))\(H'*Y+C1-C2+term_1+term_2);
        %x = U \ (L \ q);
        if(ii==1||ii==no_workers)
            if(ii==1 && i > 1)
                tic
            end
            x=(H'*H+rho*eye(s1,s1))\(H'*Y+C1-C2+term_1+term_2);   
        else
            x=(H'*H+2*rho*eye(s1,s1))\(H'*Y+C1-C2+term_1+term_2);
        end
        
        %mat=H'*H+2*rho*eye(s1,s1);
        %vec=H'*Y+C1-C2+term_1+term_2;
        %x=inv(mat)*(vec);
        out(:,ii) =x;
        
        if(ii==1 && i > 1)
            gadmm_time(i)=gadmm_time(i-1)+2*toc;
        end
        
     end
    
    
        
     for ii =2:2:no_workers
         C1= lambda(:,ii-1);
         if ii == no_workers
             C2 = zeros(s1,1);
             term_2 = zeros(s1,1);
         else
             C2= lambda(:,ii);
             term_2 = rho*out(:,ii+1);
         end
         term_1=rho*out(:,ii-1);
         first = (ii-1)*s2+1;
         last = first+s2-1;
         
        H=XX(first:last,1:s1);
        Y=YY(first:last);
        %[L, U] = factor(H, rho);
        %x=(H'*H+2*rho*eye(s1,s1))\(H'*Y+C1-C2+term_1+term_2);
        %x = U \ (L \ (H'*Y+C1-C2+term_1+term_2));
        %q=H'*Y+C1-C2+term_1+term_2;
        %[L, U] = factor(H, rho);
        if(ii==no_workers)
            x=(H'*H+rho*eye(s1,s1))\(H'*Y+C1-C2+term_1+term_2);   
        else
            x=(H'*H+2*rho*eye(s1,s1))\(H'*Y+C1-C2+term_1+term_2);
        end
        %x = U \ (L \ q);
        %mat=H'*H+2*rho*eye(s1,s1);
        %vec=H'*Y+C1-C2+term_1+term_2;
        %x=inv(mat)*(vec);
        
        out(:,ii) =x;
     end
        for ii=1:no_workers-1
            lambda(:,ii) = lambda(:,ii) + rho*(out(:,ii)-out(:,ii+1));
        end
        final_obj = 0;
        for ii =1:no_workers
            first = (ii-1)*s2+1;
            last = first+s2-1;
            final_obj = final_obj + 0.5*norm(XX(first:last,1:s1)*out(:,ii) - YY(first:last))^2;
        end
        obj_GADMM(i)=final_obj;
        loss_GADMM(i)=abs(final_obj-obj0);
        
        if(loss_GADMM(i) < acc)
            Iter = i;
            break;
        end
       
    

end
     




