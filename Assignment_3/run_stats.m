% constants
is_regression = 1;
kfolds = 5;
CANN = 1;
CSVML = 2;
CSVMP = 3;
CSVMG = 4;
CDT = 5;

% load and transform data
load('facialPoints.mat');
if is_regression
    load('headpose.mat');
    Y = pose(:,6);
else
    load('labels.mat');
    Y = labels;
end
spoints = size(points);
X = reshape(points, [spoints(1)*spoints(2) spoints(3)])';

X_ann = X';
Y_ann = Y';

% data splitting
kindices = kfoldcross(X', kfolds, false);

% hyperparameters
if is_regression
    % ANN hyperparameters
    % init NN
    net_p = minmax(X_ann);
    net_t = Y_ann;
    net_si = [132]; % NN hidden layer and nodes
    net_tfi = {'tansig' 'purelin'}; % NN transfer function
    net_btf = 'traingda'; % NN training function
    net_blf = 'learngdm'; % NN weight/bias learning function
    net_pf = 'msereg'; % NN performance function
    ann.trainParam.lr = 0.01;

    ann = newff(net_p, net_t, net_si, net_tfi, net_btf, net_blf, net_pf);

    % NN general params
    ann.trainParam.epochs = 1000;
    ann.trainParam.min_grad = 1e-05;
    ann.trainParam.max_fail = 10;
    %===================================
    % SVM hyperparameters
    PolyOrder = 3;
    KernScale = 21;
    Epsilon_lin = 0.6147;
    Epsilon_poly = 0.07;
    Epsilon_g = 0.38;
else
    % ANN hyperparameters
    % init NN
    net_p = minmax(X_ann);
    net_t = Y_ann;
    net_si = [132]; % NN hidden layer and nodes
    net_tfi = {'tansig' 'tansig'}; % NN transfer function
    net_btf = 'traingd'; % NN training function
    net_blf = 'learngd'; % NN weight/bias learning function
    net_pf = 'msereg'; % NN performance function

    ann = newff(net_p, net_t, net_si, net_tfi, net_btf, net_blf, net_pf);

    % NN general params
    ann.trainParam.epochs = 3000;
    ann.trainParam.min_grad = 1e-05;
    ann.trainParam.max_fail = 300;
    ann.trainParam.lr = 0.02;
    %===================================
    % SVM hyperparameters
    PolyOrder = 3;
    KernScale = 22;
end

% shared params for all models
% ANN
% NN data division params
ann.divideFcn = 'divideind';
ann.trainParam.showWindow = false;
ann.trainParam.showCommandLine = false;
ann.trainParam.show = 25;
ann.trainParam.time = inf;
ann.trainParam.goal = 0;
%===================================

% train model
% svml = Linear SVM
% svmp = Polynomial SVM
% svmg = RBF/Gaussian SVM
if is_regression
    STORE_TRUE_Y = 5;
else
    STORE_TRUE_Y = 6;
end
output_allmodel = zeros([length(Y(kindices == 1)) 1 STORE_TRUE_Y kfolds]);

disp('Start cv')

for li = 1:kfolds
    disp(num2str(li));
    % data segmentation
    k_test_indices = (kindices == li);
    k_train_indices = ~k_test_indices;
    
    % Manually divide dataset
    [~,~,~,~,trind,tsind] = holdout(X_ann(:, k_train_indices), Y_ann(:, k_train_indices), (1/(kfolds - 1)));
    ann.divideParam.trainInd = find(trind);
    ann.divideParam.valInd = find(tsind);
    ann.divideParam.testInd = [];
    
    % train all model
    [ann,tr] = train(ann, X_ann(:, k_train_indices), Y_ann(:, k_train_indices));
    disp('ANN done')
    if is_regression
        svml = SVM(X(k_train_indices, :), Y(k_train_indices, :), 'linear_regression', 'Epsilon', Epsilon_lin);
        disp('SVML done')
        svmp = SVM(X(k_train_indices, :), Y(k_train_indices, :), 'polynomial_regression', 'PolynomialOrder', PolyOrder, 'Epsilon', Epsilon_poly);
        disp('SVMP done')
        svmg = SVM(X(k_train_indices, :), Y(k_train_indices, :), 'rbf_regression', 'KernelScale', KernScale, 'Epsilon', Epsilon_g);
        disp('SVMG done')
    else
        svml = SVM(X(k_train_indices, :), Y(k_train_indices, :), 'linear_classification');
        disp('SVML done')
        svmp = SVM(X(k_train_indices, :), Y(k_train_indices, :), 'polynomial_classification', 'PolynomialOrder', PolyOrder);
        disp('SVMP done')
        svmg = SVM(X(k_train_indices, :), Y(k_train_indices, :), 'rbf_classification', 'KernelScale', KernScale);
        disp('SVMG done')
        dt = decision_tree_learning(X(k_train_indices, :), Y(k_train_indices, :));
        output_allmodel(:, :, CDT, li) = evaluate_tree(dt, X(k_test_indices, :));
    end
    
    % test model
    output_allmodel(:, :, STORE_TRUE_Y, li) = Y(k_test_indices, :);
    output_allmodel(:, :, CANN, li) = ann(X_ann(:, k_test_indices));
    output_allmodel(:, :, CSVML, li) = predict(svml, X(k_test_indices, :));
    output_allmodel(:, :, CSVMP, li) = predict(svmp, X(k_test_indices, :));
    output_allmodel(:, :, CSVMG, li) = predict(svmg, X(k_test_indices, :));
    
    ann = init(ann);
end

disp('End cv')

disp('Start ttest2')
% compare each model with against each other using ttest2
num_model = size(output_allmodel, 3) - 1;
model_cmp = (num_model*(num_model-1)/2);
% 1 - h-value; 2 - p-value
% CLASSIFICATION
% 1 - ANN-SVML; 2 - ANN-SVMP; 3 - ANN-SVMG; 4 - ANN-DT;
% 5 - SVML-SVMP; 6 - SVML-SVMG; 7 - SVML-DT;
% 8 - SVMP-SVMG; 9 - SVMP-DT; 10 - SVMG-DT
% REGRESSION
% 1 - ANN-SVML; 2 - ANN-SVMP; 3 - ANN-SVMG;
% 4 - SVML-SVMP; 5 - SVML-SVMG;
% 6 - SVMP-SVMG;
% stats_score
% 1 - H; 2 - P;
stats_store = zeros([2 model_cmp kfolds]);
for li = 1:kfolds
    compare = 2;
    pointer = 1;
    for ji = 1:model_cmp
        if compare > num_model
            pointer = pointer + 1;
            compare = pointer + 1;
        end
        [stats_store(1, ji, li),stats_store(2, ji, li),~,~] = ttest2(output_allmodel(:, :, pointer, li), output_allmodel(:, :, compare, li));
        compare = compare + 1;
    end
end

disp('End ttest2')

disp('Start performance metric')

if is_regression
    % Get MSE for each model
    % 1 - ANN; 2 - SVML; 3 - SVMP; 4 - SVMG
    ANS_mses = zeros([1 num_model]);
    parfor li = 1:num_model
        all_mse = zeros([1 kfolds]);
        for k = 1:kfolds
            all_mse(:, k) = immse(Y(kindices == k), output_allmodel(:, :, STORE_TRUE_Y, k));
        end
        ANS_mses(:, li) = mean(all_mse);
    end
else
    % Get accuracy, recall, precision and f1 for each model
    % 1 - ANN; 2 - SVML; 3 - SVMP; 4 - SVMG; 5 - DT
    ANS_conf_rates = zeros([4 num_model]);
    for li = 1:num_model
        sum_cm = zeros([2 2]);
        for k = 1:kfolds
            [~,cm,~,~] = confusion(output_allmodel(:, :, STORE_TRUE_Y, k)', output_allmodel(:, :, li, k)');
            sum_cm = sum_cm + cm;
        end
        [ANS_conf_rates(1, li),ANS_conf_rates(2, li),ANS_conf_rates(3, li),ANS_conf_rates(4, li)] = confusion_rates(sum_cm);
    end
end

disp('End performance metric')

% Get majority count for significance
ANS_mode_model_significance = zeros([1 model_cmp]);
parfor li = 1:model_cmp
    ANS_mode_model_significance(:, li) = mode(stats_store(1, li, :));
end