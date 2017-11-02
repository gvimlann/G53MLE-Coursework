% @author Boey Emotions data script

% constants
k_fold_cnt = 5; % Number of k-folds cross validation
use_gpu = false; % use GPU to train NN instead?
do_holdout = false; % do train test split
holdout_test_ratio = 0.2; % test ratio amount

% Create data
load facialPoints.mat
load labels.mat
X = reshape(points, [size(points, 1)*size(points, 2) length(points)]); % features
Y = labels'; % labels

X_train = X;
Y_train = Y;

if do_holdout
    [X_train,Y_train,X_test,Y_test] = holdout(X_train, Y_train, holdout_test_ratio);
end

train_indices = kfoldcross(X_train, k_fold_cnt);

% init NN
net_p = minmax(X);
net_t = Y;
net_si = [132]; % NN hidden layer and nodes
net_tfi = {'tansig' 'tansig'}; % NN transfer function
net_btf = 'traingd'; % NN training function
net_blf = 'learngd'; % NN weight/bias learning function
net_pf = 'msereg'; % NN performance function
net_ipf = {}; % NN row cell array input processing function
net_opf = {}; % NN row cell array output processing function
net_ddf = {}; % NN data diversion function

net = newff(net_p, net_t, net_si, net_tfi, net_btf, net_blf, net_pf);

% NN general params
net.trainParam.showWindow = true;
net.trainParam.showCommandLine = false;
net.trainParam.show = 25;
net.trainParam.epochs = 3000;
net.trainParam.time = inf;
net.trainParam.goal = 0;
net.trainParam.min_grad = 1e-05;
net.trainParam.max_fail = 30;

% NN data division params
net.divideFcn = 'divideind';
% net.divideParam.trainRatio = 0.8;
% net.divideParam.testRatio = 0;
% net.divideParam.valRatio = 0.2;

% NN training params
switch net_btf
    case {'trainlm'}
        net.efficiency.memoryReduction = 2; % Reduce memory usage of trainlm
        net.trainParam.mu = 0.001;
        net.trainParam.mu_dec = 0.1;
        net.trainParam.mu_inc = 10;
        net.trainParam.mu_max = 10000000000;
    case {'traingd' 'traingda' 'traingdx'}
        net.trainParam.lr = 0.02;
    case {'traingda'}
        net.trainParam.lr_inc = 1.05;
        net.trainParam.lr_dec = 0.7;
        net.trainParam.max_perf_inc = 1.04;
end

% train NN
perf = zeros([1 k_fold_cnt]);
vperf = zeros([1 k_fold_cnt]);

misclassified = zeros([1 k_fold_cnt]);
confusion_mat = zeros([2 2 k_fold_cnt]);
percentage = zeros([2 4 k_fold_cnt]);
sum_confusion_mat = zeros([2 2]);

for i = 1:k_fold_cnt
    % data segmentation
    k_test_indices = (train_indices == i);
    k_train_indices = ~k_test_indices;
    
    % Manually divide dataset
    [~,~,~,~,trind,tsind] = holdout(X_train(:, k_train_indices), Y_train(:, k_train_indices), (1/(k_fold_cnt - 1)));
    net.divideParam.trainInd = find(trind);
    net.divideParam.valInd = find(tsind);
    net.divideParam.testInd = [];
    
    % NN train
    if use_gpu
        [net,tr] = train(net, X_train(:, k_train_indices), Y_train(:, k_train_indices), 'useGPU', 'yes');
    else
        [net,tr] = train(net, X_train(:, k_train_indices), Y_train(:, k_train_indices));
    end
    output_label = net(X_train(:, k_test_indices));
    perf(i) = tr.best_perf;
    vperf(i) = tr.best_vperf;

    [misclassified(i),confusion_mat(:, :, i),~,percentage(:, :, i)] = confusion(Y_train(:, k_test_indices), output_label);
    sum_confusion_mat = sum_confusion_mat + confusion_mat(:, :, i);
    
    net = init(net);
end

iter = (1:k_fold_cnt);
plot(iter, perf, iter, vperf);
xlabel('Iteration');
ylabel('Mean Squared Error');
legend('Train', 'Validation');

[acc,rec,pre,f1] = confusion_rates(sum_confusion_mat);
mean_mse = mean(perf);
