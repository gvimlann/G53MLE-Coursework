% This is the MAIN script that generates THREE optimal SVM hyperparameters
% for 3 kernels {linear, polynomial, rbf} via inner-crossfold validation.
% 
% Random search is performed for hyperparameter optimisation

% constants
IS_REGRESSION = 0;
NUM_OF_TRIALS = 10;
INNER_kFOLD = 5;

% Min Constants
MIN_EPSILON = 0.1;
MIN_POLY = 3;
MIN_KS = 1;

% Max Constants
MAX_EPSILON = 2;
MAX_POLY = 64;
MAX_KS = 64;

SEED = 432;
% Initialise random number generator with seed
rng(SEED);

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

% Variables to interface with SVM.m
if IS_REGRESSION
    % Vars for regression
    linear_Kernel = 'linear_regression';
    poly_Kernel = 'polynomial_regression';
    rbg_Kernel = 'rbf_regression';
else
    % Vars for binary
    linear_Kernel = 'linear_classification';
    poly_Kernel = 'polynomial_classification';
    rbg_Kernel = 'rbf_classification';
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

% To keep track of parameters
history_epsilon = zeros([NUM_OF_TRIALS 1]);
history_polyOrder = zeros([NUM_OF_TRIALS 1]);
history_kernelScale = zeros([NUM_OF_TRIALS 1]);

% To keep track of average Errors
average_err_linear = zeros([NUM_OF_TRIALS 1]);
average_err_poly = zeros([NUM_OF_TRIALS 1]);
average_err_rbg = zeros([NUM_OF_TRIALS 1]);

% OUTER FOLD = Number of Trials
for i=1:NUM_OF_TRIALS

    % Random Search algorithm for SVM configurations    
    % Randomly generate parameters for each SVM {linear, poly, rbf}
    if IS_REGRESSION
        % Randomise/Tweak Epsilon
        epsilon = MIN_EPSILON + rand()*(MAX_EPSILON-MIN_EPSILON);
        history_epsilon(i) = epsilon;
        % Set up new linear SVM arguments       
        linear_val = {'Epsilon', epsilon};

        % Randomise/Tweak polynomial order
        polynomial_order = ceil(MIN_POLY + rand()*(MAX_POLY-MIN_POLY));
        history_polyOrder(i) = polynomial_order;
        % set up new polynomial SVM configuration
        poly_val = {'Epsilon', epsilon, 'PolynomialOrder', polynomial_order};

        % Randomise/Tweak rbg Kernel Scale
        kernel_scale = ceil(MIN_KS + rand()*(MAX_KS-MIN_KS));
        history_kernelScale(i) = kernel_scale;
        % set up new rbg SVM configuration
        rbg_val = {'Epsilon', epsilon, 'KernelScale', kernel_scale};
    else        
        % Randomise/Tweak polynomial order
        polynomial_order = ceil(MIN_POLY + rand()*(MAX_POLY-MIN_POLY));
        history_polyOrder(i) = polynomial_order;
        % set up new polynomia SVM configuration
        poly_val = {'PolynomialOrder', polynomial_order};
        
        % Randomise/Tweak rbg Kernel Scale
        kernel_scale = ceil(MIN_KS + rand()*(MAX_KS-MIN_KS));
        history_kernelScale(i) = kernel_scale;
        % set up new rbg SVM configuration
        rbg_val = {'KernelScale', kernel_scale};
    end

    for j=1:INNER_kFOLD
        % Train and Test out models in this loop
        % data segmentation
        k_test_indices = (train_indices == j);
        k_train_indices = ~k_test_indices;

        % Train SVMs
        svm_linear = SVM(X(k_train_indices, :), Y(k_train_indices, :), linear_Kernel, linear_val{:});
        svm_poly = SVM(X(k_train_indices, :), Y(k_train_indices, :), poly_Kernel, poly_val{:});
        svm_rbg = SVM(X(k_train_indices, :), Y(k_train_indices, :), rbg_Kernel, rbg_val{:});

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

    % Determine the performance of this randomly generated model
    if IS_REGRESSION
        % Calculate the average mean squared error
        average_err_linear(i) = mean(mean_square_err_linear);
        average_err_poly(i) = mean(mean_square_err_poly);
        average_err_rbg(i) = mean(mean_square_err_rbg);

        % to keep track of the best accuracy/score
        if i == 1
            score_linear_best = average_err_linear(i);
            score_poly_best = average_err_poly(i);
            score_rbg_best = average_err_rbg(i);
            svm_linear_best = svm_linear;
            svm_poly_best = svm_poly;
            svm_rbg_best = svm_rbg;
        else 
            % Check if it is a better model
            if average_err_linear(i) < score_linear_best
                score_linear_best = average_err_linear(i);
                svm_linear_best = svm_linear;
            end
            if average_err_poly(i) < score_poly_best
                score_poly_best = average_err_poly(i);
                svm_poly_best = svm_poly;
            end
            if average_err_rbg(i) < score_rbg_best
                score_rbg_best = average_err_rbg(i);
                svm_rbg_best = svm_rbg;
            end
        end
    else
        % Compute the required performance metrics for Classification
        [acc_linear,rec_linear,pre_linear,f1_linear] = confusion_rates(sum_confusion_linear);
        [acc_poly,rec_poly,pre_poly,f1_poly] = confusion_rates(sum_confusion_poly);
        [acc_rbg,rec_rbg,pre_rbg,f1_rbg] = confusion_rates(sum_confusion_rbg);

        % average misclassification error
        average_err_linear(i) = mean(missclassification_linear);
        average_err_poly(i) = mean(missclassification_poly);
        average_err_rbg(i) = mean(missclassification_rbg);

        % to keep track of the best accuracy/score
        if i == 1
            score_linear_best = average_err_linear(i);
            score_poly_best = average_err_poly(i);
            score_rbg_best = average_err_rbg(i);
            svm_linear_best = svm_linear;
            svm_poly_best = svm_poly;
            svm_rbg_best = svm_rbg;
        else 
            % Check if it is a better model, save it if it is
            if average_err_linear(i) < score_linear_best
                score_linear_best = average_err_linear(i);
                svm_linear_best = svm_linear;
            end
            if average_err_poly(i) < score_poly_best
                score_poly_best = average_err_poly(i);
                svm_poly_best = svm_poly;
            end
            if average_err_rbg(i) < score_rbg_best
                score_rbg_best = average_err_rbg(i);
                svm_rbg_best = svm_rbg;
            end
        end
    end 
end

% Plot parameters and accuracy
if IS_REGRESSION
    figure();
    scatter(history_epsilon, average_err_linear);
    title('Error rate (Linear) vs Epsilon');
    xlabel('Epsilon');
    ylabel('Error rate');

    figure();
    scatter(history_epsilon, average_err_poly);
    title('Error rate (Poly) vs Epsilon');
    xlabel('Epsilon');
    ylabel('Error rate');

    figure();
    scatter(history_epsilon, average_err_rbg);
    title('Error rate (RBF) vs Epsilon');
    xlabel('Epsilon');
    ylabel('Error rate');
end

figure();
scatter(history_polyOrder, average_err_poly);
title('Error rate (Poly) vs Polynomial Order (q)');
xlabel('Polynomial Order (q)');
ylabel('Error rate');

figure();
scatter(history_kernelScale, average_err_rbg);
title('Error rate (RBF) vs KernelScale (sigma)');
xlabel('KernelScale (sigma)')
ylabel('Error rate');
