function [obj_GD, loss_GD, time_GD, obj_DGD, loss_DGD, time_DGD,...
    obj_LAG_PS, loss_LAG_PS, time_lagPS, comm_iter_final_LAG_PS,...
    obj_LAG_WK, loss_LAG_WK, time_lagWK, comm_iter_final_LAG_WK,...
    obj_cyclic_IAG, loss_cyclic_IAG, obj_R_IAG, loss_R_IAG, stepsize2]=GD_DGD_LAG(X,y, num_workers, num_feature, Hmax, num_iter, obj0, X_fede, y_fede, num_sample)

Hmax_sum=sum(Hmax);
hfun=Hmax_sum./Hmax;
nonprob=Hmax/Hmax_sum;

Hmin=zeros(num_workers,1);
Hcond=zeros(num_workers,1);
for i=1:num_workers
   Hmin(i)=min(abs(eig(X{i}'*X{i}))); 
   Hcond(i)=Hmax(i)/Hmin(i);
end


triggerslot=10;
Hmaxall=max(eig(X_fede'*X_fede)); 
Hminall=min(eig(X_fede'*X_fede)); 
condnum=Hmaxall/Hminall;
[cdff,cdfx] = ecdf(Hmax*num_workers/Hmaxall);
comm_save=0;
for i=1:triggerslot
    comm_save=comm_save+(1/i-1/(i+1))*cdff(find(cdfx>=min(max(cdfx),2*sqrt(1/(triggerslot*i))),1));
end


%save anis2.mat X_fede y_fede


heterconst=mean(exp(Hmax/Hmaxall));
heterconst2=mean(Hmax/Hmaxall);
rate=1/(1+sum(Hmin)/(4*sum(Hmax)));
%% parameter initialization
%triggerslot=100;
theta_agent_1=zeros(num_feature,num_iter);
theta_agent_2=zeros(num_feature,num_iter);
theta_agent_3=zeros(num_feature,num_iter);


theta=zeros(num_feature,num_iter);
theta_d=zeros(num_feature,num_workers);
grads=ones(num_feature,num_workers);
%stepsize=1/(num_workers*max(Hmax));
stepsize=1/Hmaxall;
thrd=10/(stepsize^2*num_workers^2)/triggerslot;
comm_count=ones(num_workers,1);

theta2=zeros(num_feature,num_iter);
grads2=ones(num_feature,1);
grads_gd=ones(num_feature,num_workers);
grads_d=ones(num_feature,num_workers);
stepsize2=stepsize;

theta3=zeros(num_feature,num_iter);
grads3=ones(num_feature,num_workers);
stepsize3=stepsize2/num_workers; % cyclic access learning

theta4=zeros(num_feature,num_iter);
grads4=ones(num_feature,num_workers);
stepsize4=stepsize/num_workers; % nonuniform-random access learning


thrd5=1/(stepsize^2*num_workers^2)/triggerslot;
theta5=zeros(num_feature,1);
grads5=ones(num_feature,num_workers);
stepsize5=stepsize;
comm_count5=ones(num_workers,1);

%thrd6=2/(stepsize*num_workers);
theta6=zeros(num_feature,1);
grads6=ones(num_feature,1);
stepsize6=0.5*stepsize;
comm_count6=ones(num_workers,1);


theta7=zeros(num_feature,1);
grads7=ones(num_feature,num_workers);
stepsize7=stepsize;
comm_count7=ones(num_workers,1);

lambda=0.000;


%%  GD

 grad_time(1)=0;
 theta_time(1)=0;
 time_GD(1)=0;
for iter=1:num_iter
    if(iter > 1)
        tic
    end
    grads_gd(:,1)=X{1}'*X{1}*theta2(:,iter)-X{1}'*y{1}+lambda*theta2(:,iter);
    
    if iter>1
        grad_time(iter) = grad_time(iter-1)+toc;
    end
    % central server computation
    if iter>1
    grads2=X_fede'*X_fede*theta2(:,iter)-X_fede'*y_fede+num_workers*lambda*theta2(:,iter);
    end
    %grad_error2(iter)=norm(sum(grads2,2),2);
    obj_GD(iter)=0.5*norm(X_fede*theta2(:,iter)-y_fede,2)^2;
    loss_GD(iter)=abs(0.5*norm(X_fede*theta2(:,iter)-y_fede,2)^2-obj0);%+0.5*num_workers*lambda*norm(theta2(:,iter),2)^2;
    
    if(iter > 1)
        tic
    end
    theta2(:,iter+1)=theta2(:,iter)-stepsize2*grads2;
    if iter>1
        theta_time(iter) = theta_time(iter-1)+toc;
    end
    if iter>1
        time_GD(iter) = theta_time(iter)+grad_time(iter);
    end
end




%%  Decentralized GD
comm_error2=[];
comm_grad2=[];
flagg=0;
stepsize_d=stepsize2/100;%1E-4;
grad_time(1)=0;
for iter=1:num_iter
    for ii=1:num_workers
        first = (ii-1)*num_sample+1;
        last = first+num_sample-1;
        if iter>1
            if(ii==1)
                tic
            end
        grads_d(:,ii)=X_fede(first:last,1:num_feature)'*X_fede(first:last,1:num_feature)*theta_d(:,ii)-X_fede(first:last,1:num_feature)'*y_fede(first:last)...
            +num_workers*lambda*theta_d(:,ii);
            if(ii==1)
                grad_time(iter)=grad_time(iter-1)+toc;
            end
        end
    end

        final_obj=0;
        for ii =1:num_workers
            first = (ii-1)*num_sample+1;
            last = first+num_sample-1;
            final_obj = final_obj + 0.5*norm(X_fede(first:last,1:num_feature)*theta_d(:,ii) - y_fede(first:last),2)^2;
        end
        obj_DGD(iter)=final_obj;
        loss_DGD(iter)=abs(final_obj-obj0);
        
        
        for ii=1:num_workers
            if(ii==1)
                tic
                theta_d(:,ii)=theta_d(:,ii)-1/2*stepsize_d*(grads_d(:,ii)+grads_d(:,ii+1));
                if(iter>1)
                    theta_time(iter)=theta_time(iter-1)+toc;
                else
                    theta_time(iter)=toc;
                end
            
            elseif(ii==num_workers)
                theta_d(:,ii)=theta_d(:,ii)-1/2*stepsize_d*(grads_d(:,ii)+grads_d(:,ii-1));
                
            else
                theta_d(:,ii)=theta_d(:,ii)-1/3*stepsize_d*(grads_d(:,ii)+grads_d(:,ii+1)+grads_d(:,ii-1));
            end
        end
        
        time_DGD(iter)=theta_time(iter)+grad_time(iter);
        
        
        
        %comm_error2=[comm_error2;iter*num_workers,loss_GD(iter)]; 
        %comm_grad2=[comm_grad2;iter*num_workers,grad_error2(iter)]; 
    
end


%% LAG-PS
comm_iter_init=1;
comm_index=zeros(num_workers,num_iter);
comm_error=[];
comm_grad=[];
theta_temp=zeros(num_feature,num_workers);
flagg =0;
grad_time_lagPS(1)=0;
theta_time_lagPS(1)=0;
time_lagPS(1)=0;
for iter=1:num_iter
    flag_time=0;
    if(iter == 1)
        comm_iter=comm_iter_init;
    else
        comm_iter=comm_iter_final_LAG_PS(iter-1);
    end
    comm_flag=0;
    % local worker computation
    for i=1:num_workers
        
         if(i==1 && iter > 1)
            %flag_time=1;
            tic
            grads(:,i)=X{i}'*X{i}*theta(:,iter)-X{i}'*y{i}+lambda*theta(:,iter);
            grad_time_lagPS(iter)=grad_time_lagPS(iter-1)+toc;
         end
            
        if iter>triggerslot
                                
            trigger=0;
            for n=1:triggerslot
            trigger=trigger+norm(theta(:,iter-(n-1))-theta(:,iter-n),2)^2;
            end
%             trigger=trigger/triggerslot;
            if Hmax(i)^2*norm(theta_temp(:,i)-theta(:,iter),2)^2>thrd*trigger
                   
                grads(:,i)=X{i}'*X{i}*theta(:,iter)-X{i}'*y{i}+lambda*theta(:,iter);
                theta_temp(:,i)=theta(:,iter);
                comm_index(i,iter)=1;
                comm_count(i)=comm_count(i)+1;
                comm_iter=comm_iter+1;
                comm_flag=1;
            end
        end
    end
    
    % central server computation
    grad_error(iter)=norm(sum(grads,2),2);
    obj_LAG_PS(iter)=0.5*norm(X_fede*theta(:,iter)-y_fede,2)^2;
    loss_LAG_PS(iter)=abs(0.5*norm(X_fede*theta(:,iter)-y_fede,2)^2-obj0);%+0.5*num_workers*lambda*norm(theta(:,iter),2)^2;
    
    if(iter > 1)
        tic
    end
    theta(:,iter+1)=theta(:,iter)-stepsize*sum(grads,2);
    if(iter > 1)
        theta_time_lagPS(iter)=theta_time_lagPS(iter-1)+toc;
    end
    
    if comm_flag==1
        comm_error=[comm_error;comm_iter,loss_LAG_PS(iter)];
        comm_grad=[comm_grad;comm_iter,grad_error(iter)];
    elseif  mod(iter,1000)==0
        iter
        comm_iter=comm_iter+1;
        comm_error=[comm_error;comm_iter,loss_LAG_PS(iter)];
        comm_grad=[comm_grad;comm_iter,grad_error(iter)];
    end
comm_iter_final_LAG_PS(iter)=comm_iter;
if(iter > 1)
    time_lagPS(iter)=theta_time_lagPS(iter)+grad_time_lagPS(iter);
end
end

%% LAG-WK
comm_iter5_init=1;
comm_index5=zeros(num_workers,num_iter);
comm_error5=[];
comm_grad5=[];
flagg = 0;
grad_time_lagWK(1)=0;
theta_time_lagWK(1)=0;
time_lagWK(1)=0;
for iter=1:num_iter
    if(iter == 1)
        comm_iter5=comm_iter5_init;
    else
        comm_iter5=comm_iter_final_LAG_WK(iter-1);
    end
    grad_inexc=[];
    comm_flag=0;
    % local worker computation
    for i=1:num_workers
        
          if(i==1 && iter > 1)
                tic
          end
        grad_temp=X{i}'*X{i}*theta5(:,iter)-X{i}'*y{i}+lambda*theta5(:,iter);
          if(i==1 && iter > 1)
               grad_time_lagWK(iter)=grad_time_lagWK(iter-1)+toc;
          end
        
        
        if iter>triggerslot
            trigger=0;
            for n=1:triggerslot
            trigger=trigger+norm(theta5(:,iter-(n-1))-theta5(:,iter-n),2)^2;
            end
%             trigger=trigger/triggerslot;
            if norm(grad_temp-grads5(:,i),2)^2>thrd5*trigger
                grads5(:,i)=grad_temp;
                comm_count5(i)=comm_count5(i)+1;
                comm_index5(i,iter)=1;
                comm_iter5=comm_iter5+1;
                comm_flag=1;
            end
        end       
    end
    grad_error5(iter)=norm(sum(grads5,2),2);
    obj_LAG_WK(iter)=0.5*norm(X_fede*theta5(:,iter)-y_fede,2)^2;
    loss_LAG_WK(iter)=abs(0.5*norm(X_fede*theta5(:,iter)-y_fede,2)^2-obj0);%+0.5*num_workers*lambda*norm(theta5(:,iter),2)^2;
    if comm_flag==1
       comm_error5=[comm_error5;comm_iter5,loss_LAG_WK(iter)]; 
       comm_grad5=[comm_grad5;comm_iter5,grad_error5(iter)]; 
    elseif  mod(iter,1000)==0
        iter
        comm_iter5=comm_iter5+1; 
        comm_error5=[comm_error5;comm_iter5,loss_LAG_WK(iter)]; 
       comm_grad5=[comm_grad5;comm_iter5,grad_error5(iter)]; 
    end
    
    if(iter > 1)
        tic
    end
    theta5(:,iter+1)=theta5(:,iter)-stepsize5*sum(grads5,2);
    if(iter > 1)
        theta_time_lagWK(iter)=theta_time_lagWK(iter-1)+toc;
    end
     
    comm_iter_final_LAG_WK(iter) = comm_iter5;
    if(iter > 1)
        time_lagWK(iter)=theta_time_lagWK(iter)+grad_time_lagWK(iter);
    end
end

%% cyclic IAG
flagg = 0;
for iter=1:num_iter%*floor(sqrt(num_workers))
    if mod(iter,1000)==0
        iter
    end
    if iter>1
    % local worker computation
    i=mod(iter,num_workers)+1;
    grads3(:,i)=X{i}'*X{i}*theta3(:,iter)-X{i}'*y{i}+lambda*theta3(:,iter);
    end
    % central server computation
    grad_error3(iter)=norm(sum(grads3,2),2);
    obj_cyclic_IAG(iter)=0.5*norm(X_fede*theta3(:,iter)-y_fede,2)^2;
    loss_cyclic_IAG(iter)=abs(0.5*norm(X_fede*theta3(:,iter)-y_fede,2)^2-obj0);%+0.5*num_workers*lambda*norm(theta3(:,iter),2)^2;
    theta3(:,iter+1)=theta3(:,iter)-stepsize3*sum(grads3,2);
     
end

%% non-uniform RANDOMIZED IAG
flagg = 0;
for iter=1:num_iter%*floor(sqrt(num_workers))
    if mod(iter,1000)==0
        iter
    end
    % local worker computation
    workprob=rand;
    for i=1:num_workers
        if workprob<=sum(nonprob(1:i));
           break;
        end
    end
    %i=randi(num_workers);   
    if iter>1
    grads4(:,i)=X{i}'*X{i}*theta4(:,iter)-X{i}'*y{i}+lambda*theta4(:,iter);
    end
    % central server computation
    grad_error4(iter)=norm(sum(grads4,2),2);
    obj_R_IAG(iter)=0.5*norm(X_fede*theta4(:,iter)-y_fede,2)^2;
    loss_R_IAG(iter)=abs(0.5*norm(X_fede*theta4(:,iter)-y_fede,2)^2-obj0);%+0.5*num_workers*lambda*norm(theta4(:,iter),2)^2;
    theta4(:,iter+1)=theta4(:,iter)-stepsize4*sum(grads4,2);
    
end