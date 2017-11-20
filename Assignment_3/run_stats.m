% constants
is_regression = 0;
test_ratio = 0.3;

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

% data splitting
[X_train,Y_train,X_test,Y_test] = holdout(X', Y', test_ratio);

% hyperparameters
if is_regression
    % ANN hyperparameters
    % init NN
    net_p = minmax(X);
    net_t = Y;
    net_si = [132]; % NN hidden layer and nodes
    net_tfi = {'tansig' 'purelin'}; % NN transfer function
    net_btf = 'traingda'; % NN training function
    net_blf = 'learngdm'; % NN weight/bias learning function
    net_pf = 'msereg'; % NN performance function
    net.trainParam.lr = 0.01;

    ann = newff(net_p, net_t, net_si, net_tfi, net_btf, net_blf, net_pf);

    % NN general params
    ann.trainParam.epochs = 1000;
    ann.trainParam.min_grad = 1e-05;
    ann.trainParam.max_fail = 10;
    %===================================
    % SVM hyperparameters
    
else
    % ANN hyperparameters
    % init NN
    net_p = minmax(X);
    net_t = Y;
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
    
end

% shared params for all models
% ANN
% NN data division params
ann.divideFcn = 'divideind';
ann.divideParam.trainRatio = 0.8;
ann.divideParam.testRatio = 0;
ann.divideParam.valRatio = 0.2;
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
[ann,tr] = train(ann, X_train, Y_train);
if is_regression
    % svml =
    % svmp = 
    % svmg = 
else
    dt = decision_tree_learning(X_train, Y_train);
    % svml =
    % svmp = 
    % svmg = 
end

% statistical analysis (student T test)
if is_regression
    output_allmodel = zeros([length(Y_test) 4]); % Length of test output * 4 models (excl DT)
else
    output_allmodel = zeros([length(Y_test) 5]); % Length of test output * 5 models (incl DT)
end

% test all model and store its output
output_allmodel(:, 1) = ann(X_test);
output_allmodel(:, 2) = svml.predict(X_test);
output_allmodel(:, 3) = svmp.predict(X_test);
output_allmodel(:, 4) = svmg.predict(X_test);
if ~is_regression
    output_allmodel(:, 5) = evaluate_tree(dt, X_test);
end

% compare each model with against each other using ttest2
num_model = size(output_allmodel, 2);
compare = 2;
pointer = 1;
ticker = 1;
ttest_store = cell(4, 10);
while pointer < num_model-1
    if compare == num_model
        pointer = pointer + 1;
        compare = pointer + 1;
    end
    [H,P,CI,STATS] = ttest2(output_allmodel(:, pointer), output_allmodel(:, compare));
    ttest_score{ticker}.h = H;
    ttest_score{ticker}.p = P;
    ttest_score{ticker}.ci = CI;
    ttest_score{ticker}.stats = STATS;
    compare = compare + 1;
    ticker = ticker + 1;
end