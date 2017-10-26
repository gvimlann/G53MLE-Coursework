% @author Boey Emotions data script

% constants
k_fold_cnt = 10; % Number of k-folds cross validation
train_test_split_ratio = 0.2; % ratio for splitting train and test set

% Create data
load emotions_data.mat
X = x'; % features
Y = label_encode(y)'; % labels

% cross validation
[train test] = crossvalind('HoldOut', length(X), train_test_split_ratio);
X_train = X(:, train);
Y_train = Y(:, train);
X_test = X(:, test);
Y_test = Y(:, test);
train_indices = crossvalind('Kfold', length(train), k_fold_cnt);

% init NN
net_p = minmax(X);
net_t = Y;
net_si = [136 68 34 17 8]; % NN hidden layer and nodes
net_tfi = {'logsig' 'logsig' 'logsig' 'logsig' 'logsig'}; % NN transfer function
net_btf = 'traingda'; % NN training function
net_blf = 'learngdm'; % NN weight/bias learning function
net_pf = 'softmax'; % NN performance function
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
net.trainParam.max_fail = 6;
% NN training params
switch net_btf
    case {'trainlm'}
        %net.efficiency.memoryReduction = 2; % Reduce memory usage of trainlm
        net.trainParam.mu = 0.001;
        net.trainParam.mu_dec = 0.1;
        net.trainParam.mu_inc = 10;
        net.trainParam.mu_max = 10000000000;
    case {'traingd' 'traingda'}
        net.trainParam.lr = 0.01;
    case {'traingda'}
        net.trainParam.lr_inc = 1.05;
        net.trainParam.lr_dec = 0.7;
        net.trainParam.max_perf_inc = 1.04;
end

% train NN
% for i = 1:k_fold_cnt
%     k_test_indices = (train_indices == i);
%     k_train_indices = ~k_test_indices;
%     [net tr] = train(net, X_train(:, k_train_indices), Y_train(:, k_train_indices));
%     plotperform(tr);
% end

