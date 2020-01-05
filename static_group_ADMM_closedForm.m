function [obj_GADMM, loss_GADMM, Iter, gadmm_time, com_cost]=static_group_ADMM_closedForm(XX,YY, rho, no_workers, num_feature, noSamples, num_iter, obj0...
    , acc, coherence_Time, pathCost_static_matrix)

Iter= num_iter;   
           
s1=num_feature;
s2=noSamples;
lambda = zeros(s1,no_workers);
out=zeros(s1,no_workers);
out_prev=zeros(s1,no_workers);
max_iter = num_iter;
gadmm_time(1)=0;

%path0=path;
%pathCost0=pathCost
kk=1;
pathCost=pathCost_static_matrix(kk,:);
kk=kk+1;
 for i = 1:max_iter
    % i
     if (i > 1 && mod(i,coherence_Time)== 0)
         %path0=path_matrix(:,kk);
         pathCost=pathCost_static_matrix(kk,:);
         %[path0, pathCost]=findPath(no_workers);
         kk=kk+1;
     end
     
     for ii =1:2:no_workers
         
         
         if(ii~=1)
            l=ii-1;
         end
         if(ii~=no_workers)
            r=ii+1;
         end
         
         first = (ii-1)*s2+1;
         last = first+s2-1;
        
         H=XX(first:last,1:s1);
         Y=YY(first:last);
                
         if(i==1)
            com_cost(i)=sum(pathCost);
         else
            com_cost(i)= com_cost(i-1)+sum(pathCost);
         end
         
         C1=lambda(:,ii);
         
         if ii==1
             %C1 = zeros(s1,1);
             term_1 =zeros(s1,1);
         else
             %C1= lambda(:,ii-1);
             term_1=rho*out(:,l);
         end
         
         if ii == no_workers
             %C2 = zeros(s1,1);
             term_2 = zeros(s1,1);
         else
             %C2= lambda(:,ii);
             term_2 = rho*out(:,r);
             
         end
         
        %q=H'*Y+C1-C2+term_1+term_2;
        %[L, U] = factor(H, rho);
        %x=(H'*H+2*rho*eye(s1,s1))\(H'*Y+C1-C2+term_1+term_2);
        %x = U \ (L \ q);
        if(ii==1||ii==no_workers)
            if(ii==1 && i > 1)
                tic
            end
            x=(H'*H+rho*eye(s1,s1))\(H'*Y-C1+term_1+term_2);   
        else
            x=(H'*H+2*rho*eye(s1,s1))\(H'*Y-C1+term_1+term_2);
        end
        
        out_prev(:,ii)=out(:,ii);
        out(:,ii) =x;
        
        if(ii==1 && i > 1)
            gadmm_time(i)=gadmm_time(i-1)+2*toc;
        end
        
     end
    
    
        
     for ii =2:2:no_workers
         
         if(ii~=1)
            l=ii-1;
         end
         if(ii~=no_workers)
            r=ii+1;
         end

         first = (ii-1)*s2+1;
         last = first+s2-1;
         
         H=XX(first:last,1:s1);
         Y=YY(first:last);
              
         C1= lambda(:,ii);
         if ii == no_workers
             %C2 = zeros(s1,1);
             term_2 = zeros(s1,1);
         else
             %C2= lambda(:,ii);
             term_2 = rho*out(:,r);
         end
         term_1=rho*out(:,l);


        if(ii==no_workers)
            x=(H'*H+rho*eye(s1,s1))\(H'*Y-C1+term_1+term_2);   
        else
            x=(H'*H+2*rho*eye(s1,s1))\(H'*Y-C1+term_1+term_2);
        end

        out_prev(:,ii)=out(:,ii);
        out(:,ii) =x;
     end
     
    
    for ii=1:no_workers
         if(ii~=1)
            l=ii-1;
         end
         if(ii~=no_workers)
            r=ii+1;
         end
        if(ii==1)
            lambda(:,ii) = lambda(:,ii) + rho*(out(:,ii)-out(:,r));
        elseif(ii==no_workers)
            lambda(:,ii) = lambda(:,ii) - rho*(out(:,l)-out(:,ii));
        else
            lambda(:,ii) = lambda(:,ii) - rho*(out(:,l)-out(:,ii)) + rho*(out(:,ii)-out(:,r));
        end
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
     




