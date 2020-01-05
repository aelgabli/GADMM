function [obj_dualAvg, loss_dualAvg, Iter, dualAvg_time]=dual_averaging_logisticReg(XX,YY, no_workers, num_feature, noSamples, num_iter, obj0, acc, alpha, eta)
Iter= num_iter;   
           
s1=num_feature;
s2=noSamples;
%lambda = zeros(s1,no_workers);
out=zeros(s1,no_workers);
%out_prev=zeros(s1,no_workers);
Z=zeros(s1,no_workers);
Z_prev=zeros(s1,no_workers);
%alpha=1E-6;
max_iter = num_iter;
dualAvg_time(1)=0;

 for i = 1:max_iter
     
     for ii =1:no_workers

         first = (ii-1)*s2+1;
         last = first+s2-1;
        
        H=XX(first:last,1:s1);
        Y=YY(first:last);
        
         if(ii==1)
            if(i > 1)
                tic
            end
         end

        %grads(:,ii)=H'*H*out(:,ii)-H'*Y;
        
        grads(:,ii)=-(H'*(Y./(1+exp(Y.*(H*out(:,ii))))))...
                +eta*out(:,ii);
        
        
        if(ii==1)
           Z(:,ii)=Z_prev(:,ii+1)+grads(:,ii);
        elseif(ii==no_workers)
           Z(:,ii)=Z_prev(:,ii-1)+grads(:,ii);
        else
           Z(:,ii)=0.5*Z_prev(:,ii+1)+0.5*Z_prev(:,ii-1)+grads(:,ii);
        end
        
        out(:,ii)=-alpha*Z(:,ii);
        
        Z_prev(:,ii)=Z(:,ii);
        
        if(ii==1 && i > 1)
            dualAvg_time(i)=dualAvg_time(i-1)+2*toc;
        end
            
        
     end
    
    
        
        final_obj=0;
        for ii =1:no_workers
            first = (ii-1)*s2+1;
            last = first+s2-1;
            %final_obj = final_obj + 0.5*norm(XX(first:last,1:s1)*out(:,ii) - YY(first:last))^2;
            final_obj=final_obj+eta*0.5*norm(out(:,ii))^2+sum(log(1+exp(-YY(first:last).*(XX(first:last,1:s1)*out(:,ii)))));
        end
        obj_dualAvg(i)=final_obj;
        loss_dualAvg(i)=abs(final_obj-obj0);
        
        if(loss_dualAvg(i) < acc)
            Iter = i;
            break;
        end
       
    

end
     




