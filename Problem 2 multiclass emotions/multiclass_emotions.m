% @author Boey Emotions data script

% constants
k_fold_cnt = 10; % Number of k-folds cross validation
holdout_ratio = 0.2; % ratio for splitting train and test set
do_holdout = true; % apply train test split?
do_adapt = true; % Adapt NN to current iteration dataset?
adapt_max_iter = 1000; % adaptation maximum iterations
adapt_perf_inc = 0.1; % adaptation termination performance increment
use_gpu = true; % use GPU to train NN instead?

% Create data
load emotions_data.mat
X = x'; % features
Y = label_encode(y)'; % labels
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

% init NN
net_p = minmax(X);
net_t = Y;
net_si = [284 142]; % NN hidden layer and nodes
net_tfi = {'tansig' 'softmax'}; % NN transfer function
net_btf = 'traingda'; % NN training function
net_blf = 'learngdm'; % NN weight/bias learning function
net_pf = 'mse'; % NN performance function
net_ipf = {}; % NN row cell array input processing function
net_opf = {}; % NN row cell array output processing function
net_ddf = {}; % NN data diversion function

net = newff(net_p, net_t, net_si, net_tfi, net_btf, net_blf, net_pf);

% NN general params
net.trainParam.showWindow = true;
net.trainParam.showCommandLine = false;
net.trainParam.show = 25;
net.trainParam.epochs = 1500;
net.trainParam.time = inf;
net.trainParam.goal = 0;
net.trainParam.min_grad = 1e-07;
net.trainParam.max_fail = 15;
% NN training params
switch net_btf
    case {'trainlm'}
        net.efficiency.memoryReduction = 2; % Reduce memory usage of trainlm
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

% train NN
Accuracy = zeros([1 k_fold_cnt]);
Recall = zeros([1 k_fold_cnt]);
Precision = zeros([1 k_fold_cnt]);
F1_measure = zeros([1 k_fold_cnt]);
perf = zeros([1 k_fold_cnt]);

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
    perf(i) = mse(net, Y_train(:, k_train_indices), net(X_train(:, k_train_indices)));
    
    % Adaptation
    if do_adapt && i < floor(k_fold_cnt / 2)
        j = 0;
%         figure; hold on;
%         plot(j, tr.best_perf, '--g', 'LineWidth', 2);
        perf_threshold = tr.best_perf * (1 - adapt_perf_inc);
        % Terminating condition: MSE(CUR) < MEAN(MSE(1:CUR - 1)) || ITER * ITER
        while(j < adapt_max_iter && perf(i) > perf_threshold)
            [net,y,e] = adapt(net, X_train(:, k_train_indices), Y_train(:, k_train_indices));
            perf(i) = mse(e);
            j = j + 1;
%             plot(j, perf(i), '--g', 'LineWidth', 2);
        end
    end
%     hold off;
    
    % average confusion matrix rates
    [Accuracy(i),Recall(i),Precision(i),F1_measure(i)] = confusion_rates(...
        gen_confusionmat(Y_train(:, k_train_indices), net(X_train(:, k_train_indices))));
end

if do_holdout
    k_fold_cnt = k_fold_cnt + 1;
    % NN test
    [net,tr] = train(net, X_test, Y_test);
    
    y_confusion = net(X_test);
    [Accuracy(k_fold_cnt),Recall(k_fold_cnt),Precision(k_fold_cnt),F1_measure(k_fold_cnt)] = confusion_rates(...
        gen_confusionmat(Y_test, y_confusion));
    plotconfusion(Y_test, y_confusion);
    perf(k_fold_cnt) = mse(net, Y_test, y_confusion);
else
    y_confusion = net(X_train);
    plotconfusion(Y_train, y_confusion);
end
print('plotconfusion', '-dpng'); % Inversed plot, require consideration

iter = (1:k_fold_cnt);

% MSE plot
figure
plot(iter, perf, '--gs', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', [0.5 0.5 0.5]);
title('Mean Squared Error Plot');
xlabel('Iterations');
ylabel('Mean Squared Error (MSE)');
print('mseplot', '-dpng');

% Confusion matrix rates plot
figure
plot(iter, Accuracy, 'b', ...
    iter, Recall, '--xk', ...
    iter, Precision, ':sm', ...
    iter, F1_measure, '-.dr');
legend('Accuracy', 'Recall', 'Precision', 'F1 Measure');
title('Confusion Matrix Rates');
xlabel('Iterations');
print('cmr', '-dpng');

% Confusion matrix plot
avg_confusionrates = [mean(Accuracy); mean(Recall); mean(Precision); mean(F1_measure)];
confusionrates_table = uitable(figure, 'Data', avg_confusionrates, ...
    'RowName', {'Accuracy' 'Recall' 'Precision' 'F1 Measure'}, ...
    'ColumnName', {'Rates'});
print('confusionrates', '-dpng');