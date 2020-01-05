
clear
close all

%% data allocation for linear regression
[Xdata_28] = load('data28/data.txt'); % UCLA Housing dataset
[ydata_28] = load('data28/y.txt'); 
[Xdata_29] = load('data29/data.txt'); % Body Fat dataset
[ydata_29] = load('data29/y.txt');   
[Xdata_30] = load('data30/data.txt'); % Age of abalone dataset
[ydata_30] = load('data30/y.txt');    

num_feature=size(Xdata_29,2);
total_sample=size(Xdata_29,1);
per_split=25;
num_workers=floor(total_sample/per_split);
total_sample =num_workers*per_split;
num_sample=per_split;

num_iter1=40000;
num_iter=num_iter1;
X=cell(num_workers);
y=cell(num_workers);



diagmatrix=diag(ones(total_sample,1));
Hmax=zeros(num_workers,1);




for n=1:num_workers
        first = (n-1)*per_split+1;
        last = first+per_split-1;
        X{n}=Xdata_29(first:last,1:num_feature);
        y{n}=ydata_29(first:last);
        Hmax(n)=max(eig(X{n}'*X{n}));
end


num_feature=size(X{1},2);

X_fede=[];
y_fede=[];
for i=1:num_workers
  X_fede=[X_fede;X{i}];
  y_fede=[y_fede;y{i}];
end

lambda=0.000;

%% Optimal solution

XX=X_fede;
YY=y_fede;
size = num_feature;

obj0 = opt_sol_closedForm(XX,YY);%(0.5*sum_square(XX*z1 - YY));
%obj1 = opt_sol(XX,YY,num_feature);%(0.5*sum_square(XX*z1 - YY));

opt_obj = obj0*ones(1,num_iter);


% %% DG, DGD, and LAG baselines
% [obj_GD, loss_GD, time_GD, obj_DGD, loss_DGD, time_DGD,...
%     obj_LAG_PS, loss_LAG_PS, time_lagPS, comm_iter_final_LAG_PS,...
%     obj_LAG_WK, loss_LAG_WK, time_lagWK, comm_iter_final_LAG_WK,...
%     obj_cyclic_IAG, loss_cyclic_IAG, obj_R_IAG, loss_R_IAG, stepsize2]=GD_DGD_LAG(X,y, num_workers, num_feature, Hmax, num_iter, obj0, X_fede, y_fede, num_sample);
% 
% 
% 
% 
% 
% %num_iter=500000;
% acc = 1E-4;
% 
% [obj_dualAvg, loss_dualAvg, Iter_dualAvg, dualAvg_time] = dual_averaging(X_fede,y_fede, num_workers, num_feature, per_split, num_iter, obj0, acc, stepsize2);
% 


%% Group-ADMM

num_iter=20000;

rho = 0.1;
acc = 1E-4;


coherence_Time=1E9;

[path, pathCost]=findPath(num_workers);
%path=1:num_workers;
%pathCost=ones(1,num_workers);

% [obj_GADMM_rho3, loss_GADMM_rho3, Iter_3, gadmm_time3, com_cost] = dynamic_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
%     , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);
% 
coherence_Time=50;

[obj_GADMM_rho3_d, loss_GADMM_rho3_d, Iter_3_d, gadmm_time3_d, com_cost_d] = dynamic_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);




%coherence_Time=50;

[obj_GADMM_rho3_d2, loss_GADMM_rho3_d2, Iter_3_d2, gadmm_time3_d2, com_cost_d2] = dynamic_group_ADMM_closedForm_v2(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);

% coherence_Time=10;
% 
% [obj_GADMM_rho3_d, loss_GADMM_rho3_d, Iter_3_d, gadmm_time3_d, com_cost_d] = dynamic_group_ADMM_cvx_v2(X_fede,y_fede, rho, num_workers, num_feature...
%     , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);



semilogy(loss_GADMM_rho3_d,'r-','LineWidth',3);
hold on

semilogy(loss_GADMM_rho3_d2,'b--','LineWidth',3);
hold on

kkk=1;