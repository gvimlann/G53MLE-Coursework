% constants
is_regression = 0;
outer_kfold = 10;
inner_kfold = 10;

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

% variables
svm_store = {};
if is_regression
    % Vars for regression
else
    % Vars for binary
end

% cross-valdiate
