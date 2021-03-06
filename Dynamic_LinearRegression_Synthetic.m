
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
num_workers=50;
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

% %% DG, DGD, and LAG baselines
% [obj_GD, loss_GD, time_GD, obj_DGD, loss_DGD, time_DGD,...
%     obj_LAG_PS, loss_LAG_PS, time_lagPS, comm_iter_final_LAG_PS,...
%     obj_LAG_WK, loss_LAG_WK, time_lagWK, comm_iter_final_LAG_WK,...
%     obj_cyclic_IAG, loss_cyclic_IAG, obj_R_IAG, loss_R_IAG, stepsize2]=GD_DGD_LAG(X,y, num_workers, num_feature, Hmax, num_iter, obj0, X_fede, y_fede, num_sample);
% 
% 
% 
% 
% %% Dual Averaging baseline
% num_iter=50000;
% acc = 1E-4;
% 
% [obj_dualAvg, loss_dualAvg, Iter_dualAvg, dualAvg_time] = dual_averaging(X_fede,y_fede, num_workers, num_feature, num_sample, num_iter, obj0, acc, stepsize2);
% 
%% Group-ADMM, (The proposed algorithm)
num_iter=5000;
rho = 3;
acc = 1E-4;

%% Compare D_GADMM with GADMM
coherence_Time=10;

%[path, pathCost, grid]=findPath(num_workers);

%costPath_static = calc_cost(grid);
[path0, pathCost0, grid]=findPath(num_workers);
pathCost_static = calc_cost(grid, path0);

path_matrix=[path0];
pathCost_matrix=[pathCost0];
pathCost_static_matrix=[pathCost_static];

for i=1:1000
    [path, pathCost, grid]=findPath(num_workers);
    pathCost_static = calc_cost(grid, path0);
    pathCost_static_matrix=[pathCost_static_matrix;pathCost_static];
    path_matrix=[path_matrix;path];
    pathCost_matrix=[pathCost_matrix;pathCost];
end
    
    
[obj_GADMM_rho3s, loss_GADMM_rho3s, Iter_3s, gadmm_time3s, com_cost_s] = static_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, coherence_Time, pathCost_static_matrix);


  
% 
[obj_GADMM_rho3_d0, loss_GADMM_rho3_d0, Iter_3_d0, gadmm_time3_d0, com_cost_d0] = dynamic_group_ADMM_closedForm_v0(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path_matrix, pathCost_matrix, coherence_Time);


num_iter=Iter_3s;
for iter=1:num_iter
    cumulative_com_GADMM_rho3s(iter)=com_cost_s(iter);%iter*num_workers;   
    %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
end

num_iter=Iter_3_d0;
for iter=1:num_iter
    %cumulative_com_GADMM_rho3_d0_1(iter)=com_cost_d0(iter);%iter*num_workers; 
    cumulative_com_GADMM_rho3_d0(iter)=com_cost_d0(iter)+5/15*com_cost_d0(iter);%30/100*com_cost_d0(iter);
    %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100); 
    gadmm_time3_d0(iter)=gadmm_time3_d0(iter)+5/15*gadmm_time3_d0(iter);
end


% [obj_GADMM_rho3_dr, loss_GADMM_rho3_dr, Iter_3_dr, gadmm_time3_dr, com_cost_dr] = dynamic_group_ADMM_closedForm_random(X_fede,y_fede, rho, num_workers, num_feature...
%     , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);

% semilogy(loss_GADMM_rho3s,'k-','LineWidth',3);
% hold on
% 
% semilogy(loss_GADMM_rho3_d0,'b--','LineWidth',3);
% hold on
% semilogy(loss_GADMM_rho3_dr,'r--','LineWidth',3);


%% Evaluate D-GADMM under different refresh (coherence time) periods.

coherence_Time=1E9;

[path, pathCost, ~]=findPath(num_workers);
% %path=1:num_workers;
% %pathCost=ones(1,num_workers);
% 
[obj_GADMM_rho3, loss_GADMM_rho3, Iter_3, gadmm_time3, com_cost] = dynamic_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);

  coherence_Time=1;
% 
[obj_GADMM_rho3_d1, loss_GADMM_rho3_d1, Iter_3_d1, gadmm_time3_d1, com_cost_d1] = dynamic_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);


% 
  coherence_Time=10;
% 
[obj_GADMM_rho3_d, loss_GADMM_rho3_d, Iter_3_d, gadmm_time3_d, com_cost_d] = dynamic_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);


  coherence_Time=50;
% 
[obj_GADMM_rho3_d2, loss_GADMM_rho3_d2, Iter_3_d2, gadmm_time3_d2, com_cost_d2] = dynamic_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);


  coherence_Time=100;
% 
[obj_GADMM_rho3_d3, loss_GADMM_rho3_d3, Iter_3_d3, gadmm_time3_d3, com_cost_d3] = dynamic_group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature...
    , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);






% coherence_Time=10;
% 
% [obj_GADMM_rho3_d2, loss_GADMM_rho3_d2, Iter_3_d2, gadmm_time3_d2, com_cost_d2] = dynamic_group_ADMM_closedForm_test(X_fede,y_fede, rho, num_workers, num_feature...
%     , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);
% 


% coherence_Time=10;
% 
% [obj_GADMM_rho3_d, loss_GADMM_rho3_d, Iter_3_d, gadmm_time3_d, com_cost_d] = dynamic_group_ADMM_cvx_v2(X_fede,y_fede, rho, num_workers, num_feature...
%     , num_sample, num_iter, obj0, acc, path, pathCost, coherence_Time);



% semilogy(loss_GADMM_rho3_d,'r-','LineWidth',3);
% hold on
% 
% semilogy(loss_GADMM_rho3_d2,'b--','LineWidth',3);
% hold on
% 
% % 
%  kkk=1;
% rho = 5;
% 
% [obj_GADMM_rho5, loss_GADMM_rho5, Iter_5, gadmm_time5] = group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature, num_sample, num_iter, obj0, acc);
% 
% 
% 
% rho = 7;
% 
% [obj_GADMM_rho7, loss_GADMM_rho7, Iter_7, gadmm_time7] = group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature, num_sample, num_iter, obj0, acc);
% 


% num_iter=num_iter1;
% 
% for iter=1:num_iter
%     cumulative_com_GD(iter)=iter*num_workers+iter; 
%     errorPer_GD(iter) = abs(loss_GD(iter)/opt_obj(iter)*100);        
% end
% 
% for iter=1:num_iter
%     cumulative_com_DGD(iter)=iter*num_workers; 
%     %errorPer_GD(iter) = abs(loss_GD(iter)/opt_obj(iter)*100);        
% end
% 
% for iter=1:num_iter
%     cumulative_com_LAG_PS(iter)=comm_iter_final_LAG_PS(iter)+iter;  
%     errorPer_LAG_PS(iter) = abs(loss_LAG_PS(iter)/opt_obj(iter)*100);        
% end
% 
% for iter=1:num_iter
%     cumulative_com_LAG_WK(iter)=comm_iter_final_LAG_WK(iter)+iter;  
%     errorPer_LAG_WK(iter) = abs(loss_LAG_WK(iter)/opt_obj(iter)*100);        
% end
% 
% num_iter=Iter_dualAvg;
% for iter=1:num_iter
%     cumulative_com_dualAvg(iter)=iter*num_workers;   
%     %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
% end

% num_iter=Iter_3s;
% for iter=1:num_iter
%     cumulative_com_GADMM_rho3s(iter)=com_cost_s(iter);%iter*num_workers;   
%     %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
% end
% 
% num_iter=Iter_3_d0;
% for iter=1:num_iter
%     cumulative_com_GADMM_rho3_d0(iter)=com_cost_d0(iter);%iter*num_workers;   
%     %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
% end


num_iter=Iter_3;
for iter=1:num_iter
    cumulative_com_GADMM_rho3(iter)=com_cost(iter);%iter*num_workers;   
    %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
end



num_iter=Iter_3_d;
for iter=1:num_iter
    cumulative_com_GADMM_rho3_d(iter)=com_cost_d(iter);%iter*num_workers;   
    %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
end

num_iter=Iter_3_d1;
for iter=1:num_iter
    cumulative_com_GADMM_rho3_d1(iter)=com_cost_d1(iter);%iter*num_workers;   
    %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
end

num_iter=Iter_3_d2;
for iter=1:num_iter
    cumulative_com_GADMM_rho3_d2(iter)=com_cost_d2(iter);%iter*num_workers;   
    %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
end

num_iter=Iter_3_d3;
for iter=1:num_iter
    cumulative_com_GADMM_rho3_d3(iter)=com_cost_d3(iter);%iter*num_workers;   
    %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
end


% num_iter=Iter_5;
% for iter=1:num_iter
%     cumulative_com_GADMM_rho5(iter)=iter*num_workers;   
%     errorPer_GADMM_rho5(iter) = abs(loss_GADMM_rho5(iter)/opt_obj(iter)*100);        
% end
% 
% num_iter=Iter_7;
% for iter=1:num_iter
%     cumulative_com_GADMM_rho7(iter)=iter*num_workers;   
%     errorPer_GADMM_rho7(iter) = abs(loss_GADMM_rho7(iter)/opt_obj(iter)*100);        
% end
% 



% figure(1);
% semilogy(obj_GADMM_rho3,'k-','LineWidth',3);
% hold on
% semilogy(obj_GADMM_rho3_d,'r--','LineWidth',3);
% hold on
% 
% semilogy(obj_GADMM_rho3_d2,'b--','LineWidth',3);
% hold on
% 
% semilogy(obj_GADMM_rho3_d3,'b--','LineWidth',3);
% hold on
% 
% 
% xlabel({'Number of Iterations';'(a)'},'fontsize',16,'fontname','Times New Roman')
% ylabel('Objective Value','fontsize',16,'fontname','Times New Roman')
% legend('GADMM', 'D-GADMM, coh-time=10s', 'D-GADMM, coh-time=50s', 'D-GADMM, coh-time=100s');%,'Batch-GD')
% 
% 
% set(gca,'fontsize',14,'fontweight','bold');
% 
% 



figure(1);
subplot(1,3,1);
semilogy(loss_GADMM_rho3s,'k-','LineWidth',3);
hold on

semilogy(loss_GADMM_rho3_d0,'r--','LineWidth',3);
hold on

% semilogy(loss_GADMM_rho3_d0_2,'r--','LineWidth',3);
% hold on


xlabel({'Number of Iterations';'(a)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GADMM',  'D-GADMM, coherence time=15iter');%,'Batch-GD')
%ylim([10^-4 10^3])
%xlim([10 60000])

set(gca,'fontsize',14,'fontweight','bold');



subplot(1,3,2);
semilogy(cumulative_com_GADMM_rho3s,loss_GADMM_rho3s, 'k-','LineWidth',3);
hold on

%semilogy(cumulative_com_GADMM_rho3_d0_1,loss_GADMM_rho3_d0, 'b--','LineWidth',3);
%hold on

 semilogy(cumulative_com_GADMM_rho3_d0,loss_GADMM_rho3_d0,'r--','LineWidth',3);
 hold on

xlabel({'Cumulative Communication Cost';'(b)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GADMM',  'D-GADMM, coherence time = 15iter');%,'Batch-GD')

%xlim([200 400000])
set(gca,'fontsize',14,'fontweight','bold');


subplot(1,3,3);
semilogy(gadmm_time3s,loss_GADMM_rho3s, 'k-','LineWidth',3);
hold on

%semilogy(cumulative_com_GADMM_rho3_d0_1,loss_GADMM_rho3_d0, 'b--','LineWidth',3);
%hold on

 semilogy(gadmm_time3_d0,loss_GADMM_rho3_d0,'r--','LineWidth',3);
 hold on

xlabel({'Cumulative Communication Cost';'(b)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GADMM',  'D-GADMM, coherence time = 15iter');%,'Batch-GD')

%xlim([200 400000])
set(gca,'fontsize',14,'fontweight','bold');









figure(2);
%subplot(1,2,1);

semilogy(loss_GADMM_rho3,'k-','LineWidth',3);
hold on

semilogy(loss_GADMM_rho3_d1,'LineWidth',3);
hold on

semilogy(loss_GADMM_rho3_d,'r--','LineWidth',3);
hold on

semilogy(loss_GADMM_rho3_d2,'b--','LineWidth',3);
hold on

semilogy(loss_GADMM_rho3_d3,'m--','LineWidth',3);
hold on


xlabel({'Number of Iterations';'(a)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GADMM', 'D-GADMM, coh-time=1iteration', 'D-GADMM, coh-time=10iterations', 'D-GADMM, coh-time=50iterations', 'D-GADMM, coh-time=100iterations');%,'Batch-GD')
%ylim([10^-4 10^3])
%xlim([10 60000])

set(gca,'fontsize',14,'fontweight','bold');



% subplot(1,2,2);
% semilogy(cumulative_com_GADMM_rho3,loss_GADMM_rho3, 'k-','LineWidth',3);
% hold on
% semilogy(cumulative_com_GADMM_rho3_d1,loss_GADMM_rho3_d1,'LineWidth',3);
% hold on
% semilogy(cumulative_com_GADMM_rho3_d,loss_GADMM_rho3_d, 'r--','LineWidth',3);
% hold on
% semilogy(cumulative_com_GADMM_rho3_d2,loss_GADMM_rho3_d2, 'b--','LineWidth',3);
% hold on
% semilogy(cumulative_com_GADMM_rho3_d3,loss_GADMM_rho3_d3, 'm--','LineWidth',3);
% hold on
% 
% xlabel({'Cumulative Communication Cost';'(b)'},'fontsize',16,'fontname','Times New Roman')
% ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
% legend('GADMM', 'D-GADMM, coh-time=1iteration', 'D-GADMM, coh-time=10iterations', 'D-GADMM, coh-time=50iterations', 'D-GADMM, coh-time=100iterations');%,'Batch-GD')
% %ylim([10^-4 10^3])
% %xlim([200 400000])
% set(gca,'fontsize',14,'fontweight','bold');




% figure(3);
% %subplot(1,2,1);
% semilogy(loss_GADMM_rho3,'k-','LineWidth',3);
% hold on
% semilogy(loss_GADMM_rho3_d,'r--','LineWidth',3);
% hold on
% semilogy(loss_GADMM_rho3_d2,'b--','LineWidth',3);
% hold on
% semilogy(loss_GADMM_rho3_d3,'m--','LineWidth',3);
% hold on
% 
% xlabel({'Number of Iterations';'(a)'},'fontsize',16,'fontname','Times New Roman')
% ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
% legend('GADMM', 'D-GADMM, coherence time=10 iter', 'D-GADMM, coherence time=50 iter', 'D-GADMM, coherence time=100 iter');%,'Batch-GD')
% %ylim([10^-4 10^3])
% %xlim([10 60000])
% 
% set(gca,'fontsize',14,'fontweight','bold');
% 


% subplot(1,2,2);
% semilogy(cumulative_com_GADMM_rho3,loss_GADMM_rho3, 'k-','LineWidth',3);
% hold on
% semilogy(cumulative_com_GADMM_rho3_d,loss_GADMM_rho3_d, 'r--','LineWidth',3);
% hold on
% semilogy(cumulative_com_GADMM_rho3_d2,loss_GADMM_rho3_d2, 'b--','LineWidth',3);
% hold on
% semilogy(cumulative_com_GADMM_rho3_d3,loss_GADMM_rho3_d3, 'm--','LineWidth',3);
% hold on
% xlabel({'Cumulative Communication Cost';'(b)'},'fontsize',16,'fontname','Times New Roman')
% ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
% legend('GADMM', 'D-GADMM, coherence time=10 iter', 'D-GADMM, coherence time=50 iter', 'D-GADMM, coherence time=100 iter');%,'Batch-GD')
% %ylim([10^-4 10^3])
% %xlim([200 400000])
% set(gca,'fontsize',14,'fontweight','bold');





