% This is the MAIN script that generates THREE SVMs via inner-crossfold validation.
% 
% Random search is performed for hyperparameter optimisation

% constants
IS_REGRESSION = 0;
OUTER_kFOLD = 10;
INNER_kFOLD = 10;

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

% Default values for SVM
defaultKernalScale = 1;
defaultPolynomialOrder = 3;
defaultEpsilon = iqr(Y) / 13.49;

% Additional variables for polynomial and rbg
linear_input = inputParser;
poly_input = inputParser;
rbg_input = inputParser;
if IS_REGRESSION
    % Vars for regression
    linear_Kernel = 'linear_regression';
    poly_Kernel = 'polynomial_regression';
    rbg_Kernel = 'rbg_regression';

    % Additional vars for rbg
    addParameter(rbg_input, 'KernelScale', defaultKernalScale, @isnumeric);
    addParameter(rbg_input, 'Epsilon', defaultEpsilon, @isnumeric);

    % Additional vars for polynomial
    addParameter(poly_input, 'PolynomialOrder', defaultPolynomialOrder, @isnumeric);
    addParameter(poly_input, 'Epsilon', defaultEpsilon, @isnumeric);

    % Additional vars for linear
    addParameter(linear_input, 'Epsilon', defaultEpsilon, @isnumeric);
else
    % Vars for binary
    linear_Kernel = 'linear_classification';
    poly_Kernel = 'polynomial_classification';
    rbg_Kernel = 'rbf_classification';

    % Additional vars for rbg
    addParameter(rbg_input, 'KernelScale', defaultKernalScale, @isnumeric);

    % Additional vars for polynomial
    addParameter(poly_input, 'PolynomialOrder', defaultPolynomialOrder, @isnumeric);
end

parse(linear_input);
parse(poly_input);
parse(rbg_input);

% cross-validate
for i=1:OUTER_kFOLD
    % Set up model in this loop
    svm_linear = SVM(X, Y, linear_Kernel, linear_input.Results);
    svm_poly = SVM(X, Y, poly_Kernel, poly_input.Results);
    svm_rbg = SVM(X, Y, rbg_Kernel, rbg_input.Results);

    for j=1:INNER_kFOLD
        % Test out model in this loop
        
    end

    % Random Search algorithm
end
