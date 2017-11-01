function [accuracy recall precision f1_measure] = confusion_rates(confusion_matrix)
%CONFUSION_RATES Compute the required rates based on confusion matrix
    s_cm = size(confusion_matrix);

    if prod(s_cm) < 9
        disp('Confusion matrix must have at least 3 columns and rows');
        return;
    end

    accuracy = trace(confusion_matrix) / sum(sum(confusion_matrix)); % Accuracy
    if s_cm(2) == 2 % binary classification
        recall = confusion_matrix(2, 2) / sum(confusion_matrix(:, 2)); % Recall
        precision = confusion_matrix(2, 2) / sum(confusion_matrix(2, :)); % Precision
        f1_measure = 2 * ((precision * recall) / (precision + recall)); % F1-measure
    elseif s_cm(2) > 2 % multiclass classification
        recall = zeros([length(confusion_matrix) 1]);
        precision = zeros([length(confusion_matrix) 1]);
        f1_measure = zeros([length(confusion_matrix) 1]);

        for i = 1:s_cm(2)
            recall(i) = confusion_matrix(i, i) / sum(confusion_matrix(:, i)); % Recall
            precision(i) = confusion_matrix(i, i) / sum(confusion_matrix(i, :)); % Precision
            f1_measure(i) = 2 * (precision(i) * recall(i) / (precision(i) + recall(i))); % F1-measure
        end
        
        recall(isnan(recall)) = 0;
        precision(isnan(precision)) = 0;
        f1_measure(isnan(f1_measure)) = 0;

        recall = mean(recall);
        precision = mean(precision);
        f1_measure = mean(f1_measure);
    end
end

