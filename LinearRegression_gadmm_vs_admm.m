
clear
close all
%long format e
%% data allocation for linear regression
[Xdata_28] = load('data28/data.txt'); % UCLA Housing dataset
[ydata_28] = load('data28/y.txt'); 
[Xdata_29] = load('data29/data.txt'); % Body Fat dataset
[ydata_29] = load('data29/y.txt');   
[Xdata_30] = load('data30/data.txt'); % Age of abalone dataset
[ydata_30] = load('data30/y.txt'); 


num_iter1=10000;
num_iter=num_iter1;
%num_split=3;
num_workers=24;
X=cell(num_workers);
y=cell(num_workers);

num_feature=size(Xdata_29(1:50,:),2);
num_sample=size(Xdata_29(1:50,:),1); 
Xdata=randn(num_sample,num_feature);

ydata=[ydata_29(1:50)];

[Q R]=qr(Xdata);
diagmatrix=diag(ones(num_sample,1));
% [lambda]=eig(Xdata'*Xdata);
Hmax=zeros(num_workers,1);
for i=1:num_workers
   X{i}=1.3^(i-1)*Q(:,i)*Q(:,i)'+diag(ones(num_sample,1));
   Hmax(i)=max(eig(X{i}'*X{i})); 
   y{i}=ydata;
end

num_feature=size(X{1},2);






X_fede=[];
y_fede=[];
for i=1:num_workers
  X_fede=[X_fede;X{i}];
  y_fede=[y_fede;y{i}];
end

%% Optimal solution

XX=X_fede;
YY=y_fede;
%size = num_feature;

obj0 = opt_sol_closedForm(XX,YY);%(0.5*sum_square(XX*z1 - YY));
%obj1 = opt_sol(XX,YY,num_feature);%(0.5*sum_square(XX*z1 - YY));

opt_obj = obj0*ones(1,num_iter);

%% DG, DGD, and LAG baselines
% [obj_GD, loss_GD, time_GD, obj_DGD, loss_DGD, time_DGD,...
%     obj_LAG_PS, loss_LAG_PS, time_lagPS, comm_iter_final_LAG_PS,...
%     obj_LAG_WK, loss_LAG_WK, time_lagWK, comm_iter_final_LAG_WK,...
%     obj_cyclic_IAG, loss_cyclic_IAG, obj_R_IAG, loss_R_IAG, stepsize2]=GD_DGD_LAG(X,y, num_workers, num_feature, Hmax, num_iter, obj0, X_fede, y_fede, num_sample);
% 



%% Dual Averaging baseline
%num_iter=50000;
%acc = 1E-4;

%[obj_dualAvg, loss_dualAvg, Iter_dualAvg, dualAvg_time] = dual_averaging(X_fede,y_fede, num_workers, num_feature, num_sample, num_iter, obj0, acc, stepsize2);

%% Group-ADMM, (The proposed algorithm)
num_iter=20000;
%rho = 3;
acc = 1E-4;

[path, pathCost, grid, P_central, center]=findPath2(num_workers);

cost_star_topology_uplink=sum(P_central);
cost_star_topology_downlink=max(P_central);
cost_star_topology=cost_star_topology_uplink+cost_star_topology_downlink
cost_decentralized=sum(pathCost)
% [obj_GADMM_rho3, loss_GADMM_rho3, Iter_3, gadmm_time3] = group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature, num_sample, num_iter, obj0, acc);
% 

rho = 1;


[obj_ADMM_rho5, loss_ADMM_rho5, Iter_admm_5, admm_time5] = standared_ADMM(X_fede,y_fede, rho, num_workers, num_feature, num_sample, num_iter, obj0...
    , acc);

[obj_GADMM_rho5, loss_GADMM_rho5, Iter_5, gadmm_time5] = group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature, num_sample, num_iter, obj0, acc);



coherence_Time=1;


[obj_GADMM_d, loss_GADMM_d, Iter_d, gadmm_time_d, com_cost_d] = dynamic_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);


coherence_Time=10;
%[path, pathCost, grid]=findPath(num_workers);

[obj_GADMM_d10, loss_GADMM_d10, Iter_d10, gadmm_time_d10, com_cost_d10] = dynamic_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);


coherence_Time=50;
%[path, pathCost, grid]=findPath(num_workers);

[obj_GADMM_d50, loss_GADMM_d50, Iter_d50, gadmm_time_d50, com_cost_d50] = dynamic_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);


%cost_star_topology_uplink=sum(grid(center,:))
%cost_star_topology_downlink=max(grid(center,:))
%cost_star_topology=cost_star_topology_uplink+cost_star_topology_downlink;
%cost_decentralized=sum(pathCost)


num_iter=Iter_admm_5;

for iter=1:num_iter
    
    %cumulative_com_ADMM_rho5(iter)=iter*num_workers;   
    if(iter==1)
        cumulative_com_ADMM_rho5(iter)=cost_star_topology;
    else
        cumulative_com_ADMM_rho5(iter)= cumulative_com_ADMM_rho5(iter-1)+cost_star_topology;
    end        
end




num_iter=Iter_5;
for iter=1:num_iter  
    if(iter==1)
        cumulative_com_GADMM_rho5(iter)=sum(pathCost);
    else
        cumulative_com_GADMM_rho5(iter)= cumulative_com_GADMM_rho5(iter-1)+sum(pathCost);
    end
    
end

num_iter=Iter_d;
for iter=1:num_iter
    if(iter==1)
        cumulative_com_GADMM_d(iter)=sum(pathCost);
    else
        cumulative_com_GADMM_d(iter)= cumulative_com_GADMM_d(iter-1)+sum(pathCost);
    end
end

num_iter=Iter_d10;
for iter=1:num_iter
    if(iter==1)
        cumulative_com_GADMM_d10(iter)=sum(pathCost);
    else
        cumulative_com_GADMM_d10(iter)= cumulative_com_GADMM_d10(iter-1)+sum(pathCost);
    end
end


num_iter=Iter_d50;
for iter=1:num_iter
    if(iter==1)
        cumulative_com_GADMM_d50(iter)=sum(pathCost);
    else
        cumulative_com_GADMM_d50(iter)= cumulative_com_GADMM_d50(iter-1)+sum(pathCost);
    end
end




figure(2);
subplot(1,2,1);
semilogy(loss_ADMM_rho5,'m-','LineWidth',3);
hold on
semilogy(loss_GADMM_rho5,'k--','LineWidth',3);
hold on
semilogy(loss_GADMM_d,'r--','LineWidth',3);
semilogy(loss_GADMM_d10,'b-','LineWidth',3);
semilogy(loss_GADMM_d50,'k-','LineWidth',3);

xlabel({'Number of Iterations';'(a)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('ADMM','GADMM','D-GADMM, refresh rate =1','D-GADMM, refresh rate =10','D-GADMM, refresh rate =50');%,'Batch-GD')
ylim([10^-4 10^3])
%xlim([10 60000])

set(gca,'fontsize',14,'fontweight','bold');



subplot(1,2,2);

semilogy(cumulative_com_ADMM_rho5, loss_ADMM_rho5,'m-','LineWidth',3);
hold on
semilogy(cumulative_com_GADMM_rho5,loss_GADMM_rho5, 'k--','LineWidth',3);
hold on
semilogy(cumulative_com_GADMM_d,loss_GADMM_d, 'r--','LineWidth',3);
hold on
semilogy(cumulative_com_GADMM_d10,loss_GADMM_d10, 'b-','LineWidth',3);
hold on
semilogy(cumulative_com_GADMM_d50,loss_GADMM_d50, 'k-','LineWidth',3);
hold on

xlabel({'Cumulative Communication Cost';'(b)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('ADMM','GADMM','D-GADMM, refresh rate =1','D-GADMM, refresh rate =10','D-GADMM, refresh rate =50');%,'Batch-GD')
ylim([10^-4 10^3])
%xlim([200 400000])
set(gca,'fontsize',14,'fontweight','bold');

