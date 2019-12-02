
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
[obj_GD, loss_GD, time_GD, obj_DGD, loss_DGD, time_DGD,...
    obj_LAG_PS, loss_LAG_PS, time_lagPS, comm_iter_final_LAG_PS,...
    obj_LAG_WK, loss_LAG_WK, time_lagWK, comm_iter_final_LAG_WK,...
    obj_cyclic_IAG, loss_cyclic_IAG, obj_R_IAG, loss_R_IAG, stepsize2]=GD_DGD_LAG(X,y, num_workers, num_feature, Hmax, num_iter, obj0, X_fede, y_fede, num_sample);




%% Dual Averaging baseline
num_iter=50000;
acc = 1E-4;

[obj_dualAvg, loss_dualAvg, Iter_dualAvg, dualAvg_time] = dual_averaging(X_fede,y_fede, num_workers, num_feature, num_sample, num_iter, obj0, acc, stepsize2);

%% Group-ADMM, (The proposed algorithm)
num_iter=1000;
rho = 3;
acc = 1E-4;


[obj_GADMM_rho3, loss_GADMM_rho3, Iter_3, gadmm_time3] = group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature, num_sample, num_iter, obj0, acc);


rho = 5;

[obj_GADMM_rho5, loss_GADMM_rho5, Iter_5, gadmm_time5] = group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature, num_sample, num_iter, obj0, acc);



rho = 7;

[obj_GADMM_rho7, loss_GADMM_rho7, Iter_7, gadmm_time7] = group_ADMM_closedForm(X_fede,y_fede, rho, num_workers, num_feature, num_sample, num_iter, obj0, acc);



num_iter=num_iter1;

for iter=1:num_iter
    cumulative_com_GD(iter)=iter*num_workers+iter; 
    errorPer_GD(iter) = abs(loss_GD(iter)/opt_obj(iter)*100);        
end

for iter=1:num_iter
    cumulative_com_DGD(iter)=iter*num_workers; 
    %errorPer_GD(iter) = abs(loss_GD(iter)/opt_obj(iter)*100);        
end

for iter=1:num_iter
    cumulative_com_LAG_PS(iter)=comm_iter_final_LAG_PS(iter)+iter;  
    errorPer_LAG_PS(iter) = abs(loss_LAG_PS(iter)/opt_obj(iter)*100);        
end

for iter=1:num_iter
    cumulative_com_LAG_WK(iter)=comm_iter_final_LAG_WK(iter)+iter;  
    errorPer_LAG_WK(iter) = abs(loss_LAG_WK(iter)/opt_obj(iter)*100);        
end

num_iter=Iter_dualAvg;
for iter=1:num_iter
    cumulative_com_dualAvg(iter)=iter*num_workers;   
    %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
end

num_iter=Iter_3;
for iter=1:num_iter
    cumulative_com_GADMM_rho3(iter)=iter*num_workers;   
    errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
end

num_iter=Iter_5;
for iter=1:num_iter
    cumulative_com_GADMM_rho5(iter)=iter*num_workers;   
    errorPer_GADMM_rho5(iter) = abs(loss_GADMM_rho5(iter)/opt_obj(iter)*100);        
end

num_iter=Iter_7;
for iter=1:num_iter
    cumulative_com_GADMM_rho7(iter)=iter*num_workers;   
    errorPer_GADMM_rho7(iter) = abs(loss_GADMM_rho7(iter)/opt_obj(iter)*100);        
end



figure(2);
subplot(1,3,1);
semilogy(loss_GD,'r-','LineWidth',3);
hold on
semilogy(loss_cyclic_IAG,'c-','LineWidth',3);
hold on
semilogy(loss_R_IAG,'g--','LineWidth',3);
hold on
semilogy(loss_LAG_PS,'r--','LineWidth',3);
hold on
semilogy(loss_LAG_WK,'m-','LineWidth',3);
hold on
semilogy(loss_DGD,'y-','LineWidth',3);
hold on
semilogy(loss_dualAvg,'b-','LineWidth',3);
hold on
semilogy(loss_GADMM_rho3,'k-','LineWidth',3);
hold on
semilogy(loss_GADMM_rho5,'k--','LineWidth',3);
hold on
% semilogy(loss_GADMM_rho6,'r--','LineWidth',3);
% hold on

semilogy(loss_GADMM_rho7,'b--','LineWidth',3);
hold on

xlabel({'Number of Iterations';'(a)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GD','cyclic-IAG','R-IAG','LAG-PS','LAG-WK','DGD', 'Dual-Avg','GADMM, \rho=3','GADMM, \rho=5','GADMM, \rho=7');%,'Batch-GD')
ylim([10^-4 10^3])
xlim([10 60000])

set(gca,'fontsize',14,'fontweight','bold');



subplot(1,3,2);
semilogy(cumulative_com_GD, loss_GD,'r-','LineWidth',3);
hold on
semilogy(cumulative_com_LAG_PS, loss_LAG_PS,'r--','LineWidth',3);
hold on
semilogy(cumulative_com_LAG_WK, loss_LAG_WK,'m-','LineWidth',3);
hold on
semilogy(cumulative_com_DGD, loss_DGD,'y-','LineWidth',3);
hold on
semilogy(cumulative_com_dualAvg, loss_dualAvg,'b-','LineWidth',3);
hold on
semilogy(cumulative_com_GADMM_rho3,loss_GADMM_rho3, 'k-','LineWidth',3);
hold on
semilogy(cumulative_com_GADMM_rho5,loss_GADMM_rho5, 'k--','LineWidth',3);
hold on
semilogy(cumulative_com_GADMM_rho7,loss_GADMM_rho7, 'b--','LineWidth',3);
hold on



xlabel({'Cumulative Communication Cost';'(b)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GD','LAG-PS','LAG-WK','DGD', 'Dual-Avg','GADMM, \rho=3','GADMM, \rho=5','GADMM, \rho=7');
ylim([10^-4 10^3])
xlim([200 400000])
set(gca,'fontsize',14,'fontweight','bold');


subplot(1,3,3);
semilogy(time_GD, loss_GD,'r-','LineWidth',3);
hold on
semilogy(time_lagPS, loss_LAG_PS,'r--','LineWidth',3);
hold on
semilogy(time_lagWK, loss_LAG_WK,'m-','LineWidth',3);
hold on
semilogy(time_DGD, loss_DGD,'y-','LineWidth',3);
hold on
semilogy(dualAvg_time, loss_dualAvg,'b-','LineWidth',3);
hold on
semilogy(gadmm_time3,loss_GADMM_rho3, 'k-','LineWidth',3);
hold on
semilogy(gadmm_time5,loss_GADMM_rho5, 'k--','LineWidth',3);
hold on
semilogy(gadmm_time7,loss_GADMM_rho7, 'b--','LineWidth',3);
hold on



xlabel({'Clock Time';'(b)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GD','LAG-PS','LAG-WK','DGD', 'Dual-Avg','GADMM, \rho=3','GADMM, \rho=5','GADMM, \rho=7');
%ylim([10^-4 10^3])
xlim([0 3])
set(gca,'fontsize',14,'fontweight','bold');



figure(4);

semilogy(loss_GADMM_rho3,'k-','LineWidth',3);
hold on
semilogy(loss_GADMM_rho5,'k--','LineWidth',3);
hold on
semilogy(loss_GADMM_rho7,'b--','LineWidth',3);
ylim([10^-4 10^3])
xlim([10 1000])
set(gca,'fontsize',14,'fontweight','bold');

