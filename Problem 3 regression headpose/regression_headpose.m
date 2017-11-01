% @author Sio Regression headpose data script

% constants
k_fold_cnt = 10; % Number of k-folds cross validation
use_gpu = false; % use GPU to train NN instead?

% Create data
load facialPoints.mat
load headpose.mat
X = reshape(points, [size(points, 1)*size(points, 2) length(points)]); % features
Y = pose(:, 6)'; % labels

X_train = X;
Y_train = Y;
train_indices = kfoldcross(X_train, k_fold_cnt);

% init NN
net_p = minmax(X);
net_t = Y;
net_si = [132]; % NN hidden layer and nodes
net_tfi = {'tansig' 'purelin'}; % NN transfer function
net_btf = 'traingda'; % NN training function
net_blf = 'learngdm'; % NN weight/bias learning function
net_pf = 'msereg'; % NN performance function
net_ipf = {}; % NN row cell array input processing function
net_opf = {}; % NN row cell array output processing function
net_ddf = {}; % NN data diversion function

net = newff(net_p, net_t, net_si, net_tfi, net_btf, net_blf, net_pf);

% NN general params
net.trainParam.showWindow = true;
net.trainParam.showCommandLine = false;
net.trainParam.show = 25;
net.trainParam.epochs = 1000;
net.trainParam.time = inf;
net.trainParam.goal = 0;
net.trainParam.min_grad = 1e-07;
net.trainParam.max_fail = 10;

% NN data division params
net.divideParam.trainRatio = 0.8;
net.divideParam.testRatio = 0;
net.divideParam.valRatio = 0.2;

% NN training params
switch net_btf
    case {'trainlm'}
        net.efficiency.memoryReduction = 2; % Reduce memory usage of trainlm
        net.trainParam.mu = 0.001;
        net.trainParam.mu_dec = 0.1;
        net.trainParam.mu_inc = 10;
        net.trainParam.mu_max = 10000000000;
    case {'traingd' 'traingda' 'traingdx' 'traingdm'}
        net.trainParam.lr = 0.01;
    case {'traingda'}
        net.trainParam.lr_inc = 1.05;
        net.trainParam.lr_dec = 0.7;
        net.trainParam.max_perf_inc = 1.04;
end

% train NN
perf = zeros([1 k_fold_cnt]);
vperf = zeros([1 k_fold_cnt]);

for i = 1:k_fold_cnt
    % data segmentation
    k_test_indices = (train_indices == i);
    k_train_indices = ~k_test_indices;

    % NN train
    if use_gpu
        [net,tr] = train(net, X_train(:, k_train_indices), Y_train(:, k_train_indices), 'useGPU', 'yes');
    else
        [net,tr] = train(net, X_train(:, k_train_indices), Y_train(:, k_train_indices));
    end
    output_label = net(X_train(:, k_test_indices));
    perf(i) = tr.best_perf;
    vperf(i) = tr.best_vperf;

    %plotconfusion(Y_train(:, k_test_indices), output_label);
    %print(['plotconfusion_' num2str(i)], '-dpng');
end

iter = (1:k_fold_cnt);
plot(iter, perf, iter, vperf);
xlabel('Iteration');
ylabel('Mean Squared Error');
legend('Train', 'Validation');

% MSE plot
% figure
% plot(iter, perf, 'b');
% title('Mean Squared Error Plot');
% xlabel('Iterations');
% ylabel('Mean Squared Error (MSE)');
% print('mseplot', '-dpng');

mean_mse = mean(perf);

% % Confusion matrix rates plot
% figure
% plot(iter, Accuracy, 'b', ...
%     iter, Recall, '--xk', ...
%     iter, Precision, ':sm', ...
%     iter, F1_measure, '-.dr');
% legend('Accuracy', 'Recall', 'Precision', 'F1 Measure');
% title('Confusion Matrix Rates');
% xlabel('Iterations');
% print('cmr', '-dpng');
%
% % Confusion matrix plot
% avg_confusionrates = [mean(Accuracy); mean(Recall); mean(Precision); mean(F1_measure)];
% confusionrates_table = uitable(figure, 'Data', avg_confusionrates, ...
%     'RowName', {'Accuracy' 'Recall' 'Precision' 'F1 Measure'}, ...
%     'ColumnName', {'Rates'});
% print('confusionrates', '-dpng');
