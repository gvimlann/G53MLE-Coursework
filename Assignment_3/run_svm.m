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

% Split data into k folds
train_indices = kfoldcross(X, INNER_kFOLD, 0);

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

% To keep track and modify additional vars
linear_val = {};
poly_val = {};
rbg_val = {};

% To keep track of confusion matrices
confusion_linear = zeros([2 2 INNER_kFOLD]);
confusion_poly = zeros([2 2 INNER_kFOLD]);
confusion_rbg = zeros([2 2 INNER_kFOLD]);
sum_confusion_linear = zeros([2 2]);
sum_confusion_poly = zeros([2 2]);
sum_confusion_rbg = zeros([2 2]);

% cross-validate
for i=1:OUTER_kFOLD
    % Set up model in this loop

    % Parse the additional variables
    parse(linear_input, linear_val{:});
    parse(poly_input, poly_val{:});
    parse(rbg_input, rbg_val{:});

    for j=1:INNER_kFOLD
        % Test out model in this loop
        % data segmentation
        k_test_indices = (train_indices == j);
        k_train_indices = ~k_test_indices;

        % Train SVMs
        svm_linear = SVM(X(k_train_indices, :), Y(k_train_indices, :), linear_Kernel, linear_input.Results);
        svm_poly = SVM(X(k_train_indices, :), Y(k_train_indices, :), poly_Kernel, poly_input.Results);
        svm_rbg = SVM(X(k_train_indices, :), Y(k_train_indices, :), rbg_Kernel, rbg_input.Results);

        % Test the SVMs
        output_linear = predict(svm_linear, X(k_test_indices, :));
        output_poly = predict(svm_poly, X(k_test_indices, :));
        output_rbg = predict(svm_rbg, X(k_test_indices, :));

        if IS_REGRESSION
            % (Regression) 

        else
            % (Classification) Accuracy calculation using confusion matrix - sum up all confusion matrix during k fold
            [~, confusion_linear(:, :, j), ~, ~] = confusion(Y(k_test_indices, :)', output_linear');
            sum_confusion_linear = sum_confusion_linear + confusion_linear(:, :, j);

            [~,confusion_poly(:, :, j),~,~] = confusion(Y(k_test_indices, :)', output_poly');
            sum_confusion_poly = sum_confusion_poly + confusion_poly(:, :, j);

            [~,confusion_rbg(:, :, j),~,~] = confusion(Y(k_test_indices, :)', output_rbg');
            sum_confusion_rbg = sum_confusion_rbg + confusion_rbg(:, :, j);

        end
    end

    % Compute the required performance metrics
    [acc_linear,rec_linear,pre_linear,f1_linear] = confusion_rates(sum_confusion_linear);
    [acc_poly,rec_poly,pre_poly,f1_poly] = confusion_rates(sum_confusion_poly);
    [acc_rbg,rec_rbg,pre_rbg,f1_rbg] = confusion_rates(sum_confusion_rbg);

    disp(acc_linear);
    disp(acc_poly);
    disp(acc_rbg);
    % Random Search algorithm

end
