function [obj_GADMM, loss_GADMM, Iter, gadmm_time]=standared_ADMM(XX,YY, rho, no_workers, num_feature, noSamples, num_iter, obj0...
    , acc)

Iter= num_iter;   
           
s1=num_feature;
s2=noSamples;
lambda = zeros(s1,no_workers);
out=zeros(s1,no_workers);
out_prev=zeros(s1,no_workers);
max_iter = num_iter;
gadmm_time(1)=0;

 for i = 1:max_iter
    
     
     for ii =1:no_workers-1
         
         
         l=no_workers;
         
         first = (ii-1)*s2+1;
         last = first+s2-1;
        
         H=XX(first:last,1:s1);
         Y=YY(first:last);
                
%          if(i==1)
%             com_cost(i)=sum(pathCost);
%          else
%             com_cost(i)= com_cost(i-1)+sum(pathCost);
%          end
         
         C1=lambda(:,ii);
         

             term_1=rho*out(:,l);

            if(ii==1 && i > 1)
                tic
            end
            x=(H'*H+rho*eye(s1,s1))\(H'*Y-C1+term_1);   


        
        out_prev(:,ii)=out(:,ii);
        out(:,ii) =x;
        
        if(ii==1 && i > 1)
            gadmm_time(i)=gadmm_time(i-1)+2*toc;
        end
        
     end
    
    
        
        ii =no_workers;
         
         first = (ii-1)*s2+1;
         last = first+s2-1;
         
         H=XX(first:last,1:s1);
         Y=YY(first:last);
              
         %C1= lambda(:,ii);
         for j=1:num_feature
            C1(j,1)= sum(lambda(j,1:no_workers-1));
         end
         for j=1:num_feature
            term_1(j,1)=rho*sum(out(j,1:no_workers-1));
         end

         x=(H'*H+(no_workers-1)*rho*eye(s1,s1))\(H'*Y+C1+term_1);
         
         
         %x=((no_workers-1)*rho*eye(s1,s1))\(C1+term_1);
        

        out_prev(:,ii)=out(:,ii);
        out(:,ii) =x;

     
    
    for ii=1:no_workers-1

        lambda(:,ii) = lambda(:,ii) + rho*(out(:,ii)-out(:,no_workers));
        
    end
    
%     ii=no_workers;
%     for j=1:num_feature
%         lambda(j,ii) = lambda(j,ii) + rho*sum(out(j,1:no_workers-1))-rho*out(j,ii);
%     end

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
     




