% This is the MAIN script that generates one SVM via inner-crossfold validation.
% Random search is performed for hyperparameter optimisation

% constants
IS_REGRESSION = 0;
OUTER_kFOLD = 10;
INNER_kFOLD = 10;
% Kernel to be used: 'linear' OR 'polynomial' OR 'rbf'
KERNEL_TYPE = 'linear';

% load and transform data
load('facialPoints.mat');
if IS_REGRESSION
    load('headpose.mat');
    Y = pose(:,6);
else
    load('labels.mat');
    Y = labels;
end
spoints = size(points);
X = reshape(points, [spoints(1)*spoints(2) spoints(3)])';

% variables
% SVM storage during hyperparameter optimisation
svm_store = {};
if IS_REGRESSION
    % Vars for regression
    
else
    % Vars for binary
    
end

% cross-validate
