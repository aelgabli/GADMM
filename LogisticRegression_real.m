
clear
close all

%% data allocation for linear regression
%  [Xdata_29] = load('data2/data.txt'); % ionosphere dataset
%  [ydata_29] = load('data2/y.txt'); 
 %[Xdata_28] = load('data9/data.txt'); % adult dataset
%[ydata_28] = load('data9/y.txt');   
[Xdata_28] = load('data11/data.txt'); % derm dataset
[ydata_28] = load('data11/y.txt');   


num_iter1=100000;
num_iter=num_iter1;
accuracy=1E-4;
num_workers=10;
X=cell(num_workers);
y=cell(num_workers);

num_feature=size(Xdata_28,2);
num_sample=size(Xdata_28,1);
%per_split=sort(randperm(num_sample,num_workers));
per_split=floor(num_sample/num_workers);
% for n=1:num_workers
%     per_split=per_split_in*ones(1,num_workers);
% end
for n=1:num_workers
    first = (n-1)*per_split+1;
    last = first+per_split-1;
    X{n}=Xdata_28(first:last,1:num_feature);
    y{n}=ydata_28(first:last);

end



X_fede=[];
y_fede=[];
for i=1:num_workers
  X_fede=[X_fede;X{i}];
  y_fede=[y_fede;y{i}];
end

%% data pre-analysis for GD and LAG algorithms that we compare with
lambda=0.001;
lambda=0.00001;

Hmax=zeros(num_workers,1);
for i=1:num_workers
   Hmax(i)=0.25*max(abs(eig(X{i}'*X{i})))+lambda; 
end


Hmax_sum=sum(Hmax);


[obj_GD, loss_GD, time_GD, obj_DGD, loss_DGD, time_DGD,...
    obj_LAG_PS, loss_LAG_PS, time_lagPS, comm_iter_final_LAG_PS,...
    obj_LAG_WK, loss_LAG_WK, time_lagWK, comm_iter_final_LAG_WK,...
    obj_cyclic_IAG, loss_cyclic_IAG, obj_R_IAG, loss_R_IAG, stepsize2, obj0]=GD_DGD_LAG_logistic(X,y, num_workers, num_feature, Hmax, num_iter, X_fede, y_fede, per_split, lambda, accuracy);


%num_sample=per_split;
num_iter2=500000;
num_iter=num_iter2;
acc = accuracy;%1E-3;
%% Dual Averaging
[obj_dualAvg, loss_dualAvg, Iter_dualAvg, dualAvg_time]=dual_averaging_logisticReg(X_fede,y_fede, num_workers, num_feature, per_split, num_iter, obj0, acc, stepsize2, lambda);


%% GADMM
num_iter=1000;
gdStep=0.08;
%gdStep=0.01;
acc = accuracy;%1E-3;
rho=0.03;
[obj_GADMM_rho0003, loss_GADMM_rho0003, Iter_0003, gadmm_time0003] = group_ADMM_logistic_GD(X_fede,y_fede, rho, num_workers, num_feature, per_split, num_iter, obj0, lambda, acc, gdStep);
rho



rho=0.02;
[obj_GADMM_rho0002, loss_GADMM_rho0002, Iter_0002, gadmm_time0002] = group_ADMM_logistic_GD(X_fede,y_fede, rho, num_workers, num_feature, per_split, num_iter, obj0, lambda, acc, gdStep);
rho




num_iter = num_iter1;

for iter=1:num_iter
    cumulative_com_GD(iter)=iter*num_workers+iter; 
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

%num_iter=num_iter2;

for iter=1:num_iter
    cumulative_com_DGD(iter)=iter*num_workers; 
    %errorPer_GD(iter) = abs(loss_GD(iter)/opt_obj(iter)*100);        
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
% semilogy(loss_GADMM_rho0005,'k--','LineWidth',3);
% hold on
semilogy(loss_GADMM_rho0003,'b--','LineWidth',3);
hold on


xlabel({'Number of Iterations';'(a)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GD','cyclic-IAG','R-IAG','LAG-PS','LAG-WK','DGD','dualAvg', 'GADMM, \rho=2E-2'...
    ,'GADMM, \rho=3E-2');%,'Batch-GD')
%ylim([10^-4 10^3])
xlim([10 1.1E5])
ylim([10^-4 10^2])
set(gca,'fontsize',14,'fontweight','bold');



subplot(1,3,2);
semilogy(cumulative_com_GD, loss_GD(1:length(cumulative_com_GD)),'r-','LineWidth',3);
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
% semilogy(cumulative_com_GADMM_rho0005,loss_GADMM_rho0005, 'k-','LineWidth',3);
% hold on
semilogy(cumulative_com_GADMM_rho0003,loss_GADMM_rho0003, 'b--','LineWidth',3);
hold on

xlabel({'Cumulative Communication Cost';'(b)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GD','LAG-PS','LAG-WK','DGD','dualAvg', 'GADMM, \rho=2E-2'...
    ,'GADMM, \rho=3E-2');
%ylim([10^-4 10^3])
xlim([0 1.1E6])
ylim([10^-4 10^2])
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
% semilogy(gadmm_time0005,loss_GADMM_rho0005, 'k--','LineWidth',3);
% hold on
semilogy(gadmm_time0003,loss_GADMM_rho0003, 'b--','LineWidth',3);
hold on
%semilogy(gadmm_time0009,loss_GADMM_rho0009, 'b-','LineWidth',3);
%hold on
xlabel({'Clock Time';'(c)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Objective Error','fontsize',16,'fontname','Times New Roman')
legend('GD','LAG-PS','LAG-WK','DGD','dualAvg', 'GADMM, \rho=2E-2'...
    ,'GADMM, \rho=3E-2');
%ylim([10^-4 10^3])
xlim([0 3])
ylim([10^-4 10^2])
set(gca,'fontsize',14,'fontweight','bold');



figure(4);

semilogy(loss_GADMM_rho0002,'k-','LineWidth',3);
hold on
% semilogy(loss_GADMM_rho0005,'k--','LineWidth',3);
% hold on
semilogy(loss_GADMM_rho0003,'b--','LineWidth',3);
ylim([10^-4 10^0])
%xlim([1 63])
set(gca,'fontsize',14,'fontweight','bold');


figure(5);
%semilogy(cumulative_com_LAG_WK, loss_LAG_WK,'m-','LineWidth',3);
%hold on
semilogy(cumulative_com_GADMM_rho0002,loss_GADMM_rho0002, 'k-','LineWidth',3);
hold on
% semilogy(cumulative_com_GADMM_rho0005,loss_GADMM_rho0005, 'k--','LineWidth',3);
% hold on
semilogy(cumulative_com_GADMM_rho0003,loss_GADMM_rho0003, 'b--','LineWidth',3);
hold on
ylim([10^-4 10^0])
%xlim([1 1500])
set(gca,'fontsize',14,'fontweight','bold');

