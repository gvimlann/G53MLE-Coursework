% This is the MAIN script that generates THREE SVMs via inner-crossfold validation.
% 
% Random search is performed for hyperparameter optimisation

% constants
IS_REGRESSION = 1;
OUTER_kFOLD = 10;
INNER_kFOLD = 10;
EPSILON_DELTA = 0.1;
POLY_DELTA = 4;
KS_DELTA = 2;

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
defaultPolynomialOrder = 50;
defaultEpsilon = 2;

% Additional variables for polynomial and rbg
linear_input = inputParser;
poly_input = inputParser;
rbg_input = inputParser;
if IS_REGRESSION
    % Vars for regression
    linear_Kernel = 'linear_regression';
    poly_Kernel = 'polynomial_regression';
    rbg_Kernel = 'rbf_regression';

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

% To keep track of missclasification
missclassification_linear = zeros([INNER_kFOLD 1]);
missclassification_poly = zeros([INNER_kFOLD 1]);
missclassification_rbg = zeros([INNER_kFOLD 1]);

% to keep track of the best SVMs
svm_linear_best = {};
svm_poly_best = {};
svm_rbg_best = {};

% cross-validate
for i=1:OUTER_kFOLD
    % Set up model in this loop

    % Parse the additional variables
    parse(linear_input, linear_val{:});
    parse(poly_input, poly_val{:});
    parse(rbg_input, rbg_val{:});

    %disp(linear_input.Results);
    %disp(poly_input.Results);
    %disp(rbg_input.Results);

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
            % (Regression) Accuracy calculation using Mean Squared error
            regression_output_linear(k_test_indices, :, j) = output_linear;
            regression_output_poly(k_test_indices, :, j) = output_poly;
            regression_output_rbg(k_test_indices, :, j) = output_rbg;
            
            % Compute the mean squared error
            mean_square_err_linear(j) = immse(regression_output_linear(k_test_indices, :, j), Y(k_test_indices, :));
            mean_square_err_poly(j) = immse(regression_output_poly(k_test_indices, :, j), Y(k_test_indices, :));
            mean_square_err_rbg(j) = immse(regression_output_rbg(k_test_indices, :, j), Y(k_test_indices, :));

        else
            % (Classification) Accuracy calculation using confusion matrix
            % sum up all confusion matrix during k fold
            % keep track of misclasifcation as well
            [missclassification_linear(j), confusion_linear(:, :, j), ~, ~] = confusion(Y(k_test_indices, :)', output_linear');
            sum_confusion_linear = sum_confusion_linear + confusion_linear(:, :, j);

            [missclassification_poly(j),confusion_poly(:, :, j),~,~] = confusion(Y(k_test_indices, :)', output_poly');
            sum_confusion_poly = sum_confusion_poly + confusion_poly(:, :, j);

            [missclassification_rbg(j),confusion_rbg(:, :, j),~,~] = confusion(Y(k_test_indices, :)', output_rbg');
            sum_confusion_rbg = sum_confusion_rbg + confusion_rbg(:, :, j);

        end
    end

    if IS_REGRESSION
        % Calculate the average mean squared error
        average_MSE_linear = mean(mean_square_err_linear);
        average_MSE_poly = mean(mean_square_err_poly);
        average_MSE_rbg = mean(mean_square_err_rbg);

        %disp(average_MSE_linear);
        %disp(average_MSE_poly);
        %disp(average_MSE_rbg);

        % to keep track of the best accuracy/score
        if i == 1
            score_linear_best = average_MSE_linear;
            score_poly_best = average_MSE_poly;
            score_rbg_best = average_MSE_rbg; 
        else 
            % Check if it is a better model
            if average_MSE_linear < score_linear_best
                score_linear_best = average_MSE_linear;
                svm_linear_best = svm_linear;
            end
            if average_MSE_poly < score_poly_best
                score_poly_best = average_MSE_poly;
                svm_poly_best = svm_poly;
            end
            if average_MSE_rbg < score_rbg_best
                score_rbg_best = average_MSE_rbg;
                svm_rbg_best = svm_rbg;
            end
        end
    else
        % Compute the required performance metrics for Classification
        [acc_linear,rec_linear,pre_linear,f1_linear] = confusion_rates(sum_confusion_linear);
        [acc_poly,rec_poly,pre_poly,f1_poly] = confusion_rates(sum_confusion_poly);
        [acc_rbg,rec_rbg,pre_rbg,f1_rbg] = confusion_rates(sum_confusion_rbg);

        %disp(acc_linear);
        %disp(acc_poly);
        %disp(acc_rbg);

        % average misclassification error
        average_MCE_linear = mean(missclassification_linear);
        average_MCE_poly = mean(missclassification_poly);
        average_MCE_rbg = mean(missclassification_rbg);

        %disp(average_MCE_linear);
        %disp(average_MCE_poly);
        %disp(average_MCE_rbg);

        % to keep track of the best accuracy/score
        if i == 1
            score_linear_best = average_MCE_linear;
            score_poly_best = average_MCE_poly;
            score_rbg_best = average_MCE_rbg; 
        else 
            % Check if it is a better model
            if average_MCE_linear < score_linear_best
                score_linear_best = average_MCE_linear;
                svm_linear_best = svm_linear;
            end
            if average_MCE_poly < score_poly_best
                score_poly_best = average_MCE_poly;
                svm_poly_best = svm_poly;
            end
            if average_MCE_rbg < score_rbg_best
                score_rbg_best = average_MCE_rbg;
                svm_rbg_best = svm_rbg;
            end
        end

    end 

    disp(['best linear: ' string(score_linear_best)]);
    disp(['best poly: ' string(score_poly_best)]);
    disp(['best rbg: ' string(score_rbg_best)]);

    % Random Search algorithm for the next SVM configuration
    if IS_REGRESSION
        % Randomise/Tweak Epsilon
        epsilon = linear_input.Results.Epsilon - EPSILON_DELTA;
        if epsilon <= 0
            epsilon = 0.1;
        end

        % Set up new linear SVM arguments       
        linear_val = {'Epsilon', epsilon};

        % Randomise/Tweak polynomial order
        polynomial_order = poly_input.Results.PolynomialOrder - POLY_DELTA;
        if polynomial_order <= 3
            polynomial_order = 3;
        end

        % set up new polynomial SVM configuration
        poly_val = {'Epsilon', epsilon, 'PolynomialOrder', polynomial_order};

        % Randomise/Tweak rbg Kernel Scale
        kernel_scale = KS_DELTA + rbg_input.Results.KernelScale;

        % set up new rbg SVM configuration
        rbg_val = {'Epsilon', epsilon, 'KernelScale', kernel_scale};
    else        
        % Randomise/Tweak polynomial order
        polynomial_order = poly_input.Results.PolynomialOrder - POLY_DELTA;
        if polynomial_order <= 3
            polynomial_order = 3;
        end

        % set up new polynomia SVM configuration
        poly_val = {'PolynomialOrder', polynomial_order};

        % Randomise/Tweak rbg Kernel Scale
        kernel_scale = KS_DELTA + rbg_input.Results.KernelScale;
        
        % set up new rbg SVM configuration
        rbg_val = {'KernelScale', kernel_scale};
    end

end
