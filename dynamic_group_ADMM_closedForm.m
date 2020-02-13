function [obj_GADMM, loss_GADMM, Iter, gadmm_time, com_cost]=dynamic_group_ADMM_closedForm(XX,YY, rho, no_workers, num_feature, noSamples, num_iter, obj0...
    , acc, path, pathCost, coherence_Time)

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
 for i = 1:max_iter
     
     if (i > 1 && mod(i,coherence_Time)== 0)
         %[path, pathCost, ~]=findPath(no_workers);
         [path, pathCost, ~, ~, ~]=findPath2(no_workers);
     end
     
     for jj =1:2:no_workers
         
         ii=path(jj);
         
         if(ii~=path(1))
            l=path(jj-1);
         end
         if(ii~=path(no_workers))
            r=path(jj+1);
         end
         
         first = (ii-1)*s2+1;
         last = first+s2-1;
        
         H=XX(first:last,1:s1);
         Y=YY(first:last);
         
%          if (i > 1 && mod(i,coherence_Time)== 0)
%              if(ii==path(1))
%                 lambda(:,ii)=H'*Y-H'*H*out(:,ii)-rho*out(:,ii)+rho*out(:,r); 
%              elseif(ii==path(no_workers))
%                  lambda(:,ii)=H'*Y-H'*H*out(:,ii)-rho*out(:,ii)+rho*out(:,l);
%              else
%                  
%                 lambda(:,ii)=H'*Y-H'*H*out(:,ii)-2*rho*out(:,ii)+rho*out(:,r)+rho*out(:,l);
%              end
%          end
         
         if(i==1)
            com_cost(i)=sum(pathCost);
         else
            com_cost(i)= com_cost(i-1)+sum(pathCost);
         end
         
         C1=lambda(:,ii);
         
         if ii==path(1)
             %C1 = zeros(s1,1);
             term_1 =zeros(s1,1);
         else
             %C1= lambda(:,ii-1);
             term_1=rho*out(:,l);
         end
         
         if ii == path(no_workers)
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
        if(ii==path(1)||ii==path(no_workers))
            if(ii==1 && i > 1)
                tic
            end
            x=(H'*H+rho*eye(s1,s1))\(H'*Y-C1+term_1+term_2);   
        else
            x=(H'*H+2*rho*eye(s1,s1))\(H'*Y-C1+term_1+term_2);
        end
        
        %mat=H'*H+2*rho*eye(s1,s1);
        %vec=H'*Y+C1-C2+term_1+term_2;
        %x=inv(mat)*(vec);
        out_prev(:,ii)=out(:,ii);
        out(:,ii) =x;
        
        if(ii==1 && i > 1)
            gadmm_time(i)=gadmm_time(i-1)+2*toc;
        end
        
     end
    
    
        
     for jj =2:2:no_workers
         ii=path(jj);
         if(ii~=path(1))
            l=path(jj-1);
         end
         if(ii~=path(no_workers))
            r=path(jj+1);
         end
         %C1= lambda(:,ii-1);
         
         first = (ii-1)*s2+1;
         last = first+s2-1;
         
         H=XX(first:last,1:s1);
         Y=YY(first:last);
         
%          if (i > 1 && mod(i,coherence_Time)== 0)
%              if(ii==path(1))
%                 lambda(:,ii)=H'*Y-H'*H*out(:,ii)-rho*out(:,ii)+rho*out_prev(:,r); 
%              elseif(ii==path(no_workers))
%                  lambda(:,ii)=H'*Y-H'*H*out(:,ii)-rho*out(:,ii)+rho*out_prev(:,l);
%              else
%                  
%                 lambda(:,ii)=H'*Y-H'*H*out(:,ii)-2*rho*out(:,ii)+rho*out_prev(:,r)+rho*out_prev(:,l);
%              end
%          end
         
         
         C1= lambda(:,ii);
         if ii == path(no_workers)
             %C2 = zeros(s1,1);
             term_2 = zeros(s1,1);
         else
             %C2= lambda(:,ii);
             term_2 = rho*out(:,r);
         end
         term_1=rho*out(:,l);


        if(ii==path(no_workers))
            x=(H'*H+rho*eye(s1,s1))\(H'*Y-C1+term_1+term_2);   
        else
            x=(H'*H+2*rho*eye(s1,s1))\(H'*Y-C1+term_1+term_2);
        end

        out_prev(:,ii)=out(:,ii);
        out(:,ii) =x;
     end
     
    
    for jj=1:no_workers
        ii=path(jj);
         if(ii~=path(1))
            l=path(jj-1);
         end
         if(ii~=path(no_workers))
            r=path(jj+1);
         end
        if(ii==path(1))
            lambda(:,ii) = lambda(:,ii) + rho*(out(:,ii)-out(:,r));
        elseif(ii==path(no_workers))
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
     




