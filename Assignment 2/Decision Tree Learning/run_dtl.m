% Data creation
% load('labels.mat');
% load('facialPoints.mat');
p_size = size(points);
X = reshape(points, [p_size(1)*p_size(2) p_size(3)])';
Y = labels;

% constants
kfolds = 5;

% k-fold indicing
k_ind = kfoldcross(X, kfolds);

sum_confusion = zeros([2 2]);
for i = 1:kfolds
    % train
    test_ind = (k_ind == i);
    train_ind = ~test_ind;
    trained_tree = decision_tree_learning(X(train_ind,:), Y(train_ind,:));
    DrawDecisionTree(trained_tree, ['Tree_' num2str(i)]);
    % test
    test_output = evaluate_tree(trained_tree, X(test_ind,:));
    [misclassified,cm,~,rates] = confusion(Y(test_ind)', test_output);
    sum_confusion = sum_confusion + cm;
end

% accuracy, recall, precision, f1
[acc,rec,pre,f1] = confusion_rates(sum_confusion);
