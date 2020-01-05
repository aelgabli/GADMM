
clear
close all

%% data allocation for linear regression
%  [Xdata_29] = load('data2/data.txt'); % ionosphere dataset
%  [ydata_29] = load('data2/y.txt'); 
% [Xdata_29] = load('data9/data.txt'); % adult dataset
% [ydata_29] = load('data9/y.txt');   
[Xdata_29] = load('data11/data.txt'); % derm dataset
[ydata_29] = load('data11/y.txt');   


num_iter1=100000;
num_iter=num_iter1;
accuracy=1E-4;
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
%Hmax=zeros(num_workers,1);
for i=1:num_workers
   X{i}=1^(i-1)*Q(:,i)*Q(:,i)'+diag(ones(num_sample,1));
   %Hmax(i)=max(eig(X{i}'*X{i})); 
   y{i}=ydata;
end

num_feature=size(X{1},2);

%Hmax_sum=sum(Hmax);
lambda=0.001;
lambda=0.00001;

Hmax=zeros(num_workers,1);
for i=1:num_workers
   Hmax(i)=0.25*max(abs(eig(X{i}'*X{i})))+lambda; 
end


%% data pre-analysis for GD and LAG algorithms that we compare with

%lambda=0;


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

%obj0 = opt_sol_logistic(XX,YY, num_feature, lambda, num_workers);%(0.5*sum_square(XX*z1 - YY));

%opt_obj = obj0*ones(1,num_iter);


%% GD, DGD, LAG
[obj_GD, loss_GD, time_GD, obj_DGD, loss_DGD, time_DGD,...
    obj_LAG_PS, loss_LAG_PS, time_lagPS, comm_iter_final_LAG_PS,...
    obj_LAG_WK, loss_LAG_WK, time_lagWK, comm_iter_final_LAG_WK,...
    obj_cyclic_IAG, loss_cyclic_IAG, obj_R_IAG, loss_R_IAG, stepsize2, obj0]=GD_DGD_LAG_logistic(X,y, num_workers, num_feature, Hmax, num_iter, X_fede, y_fede, num_sample, lambda, accuracy);


%obj0=obj1;


%% Dual Averaging
acc = accuracy;%1E-3;
[obj_dualAvg, loss_dualAvg, Iter_dualAvg, dualAvg_time]=dual_averaging_logisticReg(X_fede,y_fede, num_workers, num_feature, num_sample, num_iter, obj0, acc, stepsize2, lambda);

%% GADMM

num_iter=400;
gdStep=2.2;%2.5;
acc = 1E-4;%accuracy;%1E-3;
rho=0.0003;
%rho=0.0300;
[obj_GADMM_rho0003, loss_GADMM_rho0003, Iter_0003, gadmm_time0003] = group_ADMM_logistic_GD(X_fede,y_fede, rho, num_workers, num_feature, num_sample, num_iter, obj0, lambda, acc, gdStep);
rho



rho=0.0002;
[obj_GADMM_rho0002, loss_GADMM_rho0002, Iter_0002, gadmm_time0002] = group_ADMM_logistic_GD(X_fede,y_fede, rho, num_workers, num_feature, num_sample, num_iter, obj0, lambda, acc, gdStep);
rho



num_iter=num_iter1;
%num_iter = 40000;

for iter=1:num_iter
    cumulative_com_GD(iter)=iter*num_workers+iter; 
    %errorPer_GD(iter) = abs(loss_GD(iter)/opt_obj(iter)*100);        
end

for iter=1:num_iter
    cumulative_com_DGD(iter)=iter*num_workers; 
    %errorPer_GD(iter) = abs(loss_GD(iter)/opt_obj(iter)*100);        
end

for iter=1:length(comm_iter_final_LAG_PS)
    cumulative_com_LAG_PS(iter)=comm_iter_final_LAG_PS(iter)+iter;  
    %errorPer_LAG_PS(iter) = abs(loss_LAG_PS(iter)/opt_obj(iter)*100);        
end

for iter=1:length(comm_iter_final_LAG_WK)
    cumulative_com_LAG_WK(iter)=comm_iter_final_LAG_WK(iter)+iter;  
    %errorPer_LAG_WK(iter) = abs(loss_LAG_WK(iter)/opt_obj(iter)*100);        
end

num_iter=Iter_dualAvg;
for iter=1:num_iter
    cumulative_com_dualAvg(iter)=iter*num_workers;   
    %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
end

num_iter=Iter_0003;
for iter=1:num_iter
    cumulative_com_GADMM_rho0003(iter)=iter*num_workers;   
    %errorPer_GADMM_rho3(iter) = abs(loss_GADMM_rho3(iter)/opt_obj(iter)*100);        
end



num_iter=Iter_0002;
for iter=1:num_iter
    cumulative_com_GADMM_rho0002(iter)=iter*num_workers;   
    %errorPer_GADMM_rho5(iter) = abs(loss_GADMM_rho5(iter)/opt_obj(iter)*100);        
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
semilogy(loss_GADMM_rho0002,'k-','LineWidth',3);
hold on
semilogy(loss_GADMM_rho0003,'b--','LineWidth',3);
hold on
% semilogy(loss_GADMM_rho0007,'b--','LineWidth',3);
% hold on
%semilogy(loss_GADMM_rho0009,'b--','LineWidth',3);
%hold on

xlabel({'Number of Iterations';'(a)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GD','cyclic-IAG','R-IAG','LAG-PS','LAG-WK','DGD','dualAvg', 'GADMM, \rho=2E-3'...
    ,'GADMM, \rho=3E-3');
%ylim([10^-4 10^3])
%xlim([10 30000])
ylim([10^-4 10^3])
set(gca,'fontsize',14,'fontweight','bold');



subplot(1,3,2);
semilogy(cumulative_com_GD(1:length(loss_GD)), loss_GD,'r-','LineWidth',3);
hold on
semilogy(cumulative_com_LAG_PS, loss_LAG_PS,'r--','LineWidth',3);
hold on
semilogy(cumulative_com_LAG_WK, loss_LAG_WK,'m-','LineWidth',3);
hold on
semilogy(cumulative_com_DGD, loss_DGD,'y-','LineWidth',3);
hold on
semilogy(cumulative_com_dualAvg, loss_dualAvg,'b-','LineWidth',3);
hold on
semilogy(cumulative_com_GADMM_rho0002,loss_GADMM_rho0002, 'k-','LineWidth',3);
hold on
semilogy(cumulative_com_GADMM_rho0003,loss_GADMM_rho0003, 'b--','LineWidth',3);
hold on
% semilogy(cumulative_com_GADMM_rho0007,loss_GADMM_rho0007, 'b--','LineWidth',3);
% hold on
%semilogy(cumulative_com_GADMM_rho0009,loss_GADMM_rho0009, 'b-','LineWidth',3);
%hold on

xlabel({'Cumulative Communication Cost';'(b)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GD','LAG-PS','LAG-WK','DGD','dualAvg', 'GADMM, \rho=2E-3'...
    ,'GADMM, \rho=3E-3');
%ylim([10^-4 10^3])
xlim([0 1E6])
ylim([10^-4 10^3])
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
semilogy(gadmm_time0002,loss_GADMM_rho0002, 'k-','LineWidth',3);
hold on
semilogy(gadmm_time0003,loss_GADMM_rho0003, 'b--','LineWidth',3);
hold on
% semilogy(gadmm_time0007,loss_GADMM_rho0007, 'b--','LineWidth',3);
% hold on
%semilogy(gadmm_time0009,loss_GADMM_rho0009, 'b-','LineWidth',3);
%hold on
xlabel({'Clock Time';'(b)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GD','LAG-PS','LAG-WK','DGD','dualAvg', 'GADMM, \rho=2E-3'...
    ,'GADMM, \rho=3E-3');
%ylim([10^-4 10^3])
xlim([0 1.5])
ylim([10^-4 10^3])
set(gca,'fontsize',14,'fontweight','bold');






figure(4);
semilogy(loss_GADMM_rho0002,'k-','LineWidth',3);
hold on
semilogy(loss_GADMM_rho0003,'b--','LineWidth',3);
hold on
% semilogy(loss_GADMM_rho0007,'b--','LineWidth',3);
% hold on
%semilogy(loss_GADMM_rho0009,'b-','LineWidth',3);
ylim([10^-4 10^1])
xlim([1 80])
set(gca,'fontsize',14,'fontweight','bold');


figure(5);
%semilogy(cumulative_com_LAG_WK, loss_LAG_WK,'m-','LineWidth',3);
%hold on
semilogy(cumulative_com_GADMM_rho0002,loss_GADMM_rho0002, 'k-','LineWidth',3);
hold on
semilogy(cumulative_com_GADMM_rho0003,loss_GADMM_rho0003, 'b--','LineWidth',3);
hold on
% semilogy(cumulative_com_GADMM_rho0007,loss_GADMM_rho0007, 'b--','LineWidth',3);
% hold on
%semilogy(cumulative_com_GADMM_rho0009,loss_GADMM_rho0009, 'b-','LineWidth',3);
%hold on
ylim([10^-4 10^1])
xlim([1 2000])
set(gca,'fontsize',14,'fontweight','bold');

