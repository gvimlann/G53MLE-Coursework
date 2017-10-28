% @author Boey Emotions data script

% constants
k_fold_cnt = 10; % Number of k-folds cross validation
holdout_ratio = 0.2; % ratio for splitting train and test set
do_holdout = true; % apply train test split?

% Create data
load emotions_data.mat % load datasets (change)
X = x'; % features (change)
Y = label_encode(y)'; % labels (change)

% initialise training
X_train = X;
Y_train = Y;
% cross validation
if do_holdout
    [d_train,d_test] = holdout(X, holdout_ratio);
    X_train = X(:, d_train);
    Y_train = Y(:, d_train);
    X_test = X(:, d_test);
    Y_test = Y(:, d_test);
end
train_indices = kfoldcross(X_train, k_fold_cnt);

% init NN (change anything)
net_p = minmax(X);
net_t = Y;
net_si = [142]; % NN hidden layer and nodes
net_tfi = {'softmax'}; % NN transfer function
net_btf = 'traingda'; % NN training function
net_blf = 'learngdm'; % NN weight/bias learning function
net_pf = 'mse'; % NN performance function
net_ipf = {}; % NN row cell array input processing function
net_opf = {}; % NN row cell array output processing function
net_ddf = {}; % NN data diversion function

net = newff(net_p, net_t, net_si, net_tfi, net_btf, net_blf, net_pf);

% NN general params (change anything)
net.trainParam.showWindow = true;
net.trainParam.showCommandLine = false;
net.trainParam.show = 25;
net.trainParam.epochs = 1500;
net.trainParam.time = inf;
net.trainParam.goal = 0;
net.trainParam.min_grad = 1e-07;
net.trainParam.max_fail = 15;
% NN training params (change anything)
switch net_btf
    case {'trainlm'}
        %net.efficiency.memoryReduction = 2; % Reduce memory usage of trainlm
        net.trainParam.mu = 0.001;
        net.trainParam.mu_dec = 0.1;
        net.trainParam.mu_inc = 10;
        net.trainParam.mu_max = 10000000000;
    case {'traingd' 'traingda'}
        net.trainParam.lr = 0.02;
    case {'traingda'}
        net.trainParam.lr_inc = 1.05;
        net.trainParam.lr_dec = 0.7;
        net.trainParam.max_perf_inc = 1.04;
end
% =========usually nothing to change after this line=======================
% train NN
Accuracy = zeros([1 k_fold_cnt]);
Recall = zeros([1 k_fold_cnt]);
Precision = zeros([1 k_fold_cnt]);
F1_measure = zeros([1 k_fold_cnt]);
for i = 1:k_fold_cnt
    % data segmentation
    k_test_indices = (train_indices == i);
    k_train_indices = ~k_test_indices;
    % NN train
    [net,tr] = train(net, X_train(:, k_train_indices), Y_train(:, k_train_indices));
    % average confusion matrix rates
    [Accuracy(i) Recall(i) Precision(i) F1_measure] = confusion_rates(gen_confusionmat(Y_train(:, k_train_indices), net(X_train(:, k_train_indices))));
end

if do_holdout
    % NN test
    [net,tr] = train(net, X_test, Y_test);
    netsim = sim(net, X_test);
    
    y_confusion = net(X_test);
    [Accuracy(k_fold_cnt+1) Recall(k_fold_cnt+1) Precision(k_fold_cnt+1) F1_measure(k_fold_cnt+1)] = confusion_rates(gen_confusionmat(Y_test, y_confusion));
    plotconfusion(Y_test, y_confusion);
else
    y_confusion = net(X_train);
    plotconfusion(Y_train, y_confusion);
end
print('plotconfusion', '-dpng');
Accuracy = mean(Accuracy);
Recall = mean(Recall);
Precision = mean(Precision);
F1_measure = mean(F1_measure);
% TODO: uitable cannot be viewed for some reason
confusionrates_table = uitable(figure, 'Data', [Accuracy Recall Precision F1_measure], 'ColumnName', {'Accuracy' 'Recall' 'Precision' 'F1_measure'});

plotperform(tr);
print('plotperform', '-dpng');
