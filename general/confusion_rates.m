function [accuracy recall precision f1_measure] = confusion_rates(confusion_matrix)
%CONFUSION_RATES Compute the required rates based on confusion matrix
    rate_row_cnt = 4;
    s_cm = size(confusion_matrix);
    total = sum(sum(confusion_matrix));

    if prod(s_cm) < 4
        disp('Confusion matrix must have at least 2 columns and rows');
        return;
    end
    
    accuracy = trace(confusion_matrix) / total; % Accuracy
    if s_cm(2) == 2 % binary classification
        recall = confusion_matrix(2, 2) / sum(confusion_matrix(2, :)); % Recall
        precision = confusion_matrix(2, 2) / sum(confusion_matrix(:, 2)); % Precision
        f1_measure = 2 / ((1 / accuracy + (1 / precision))); % F1-measure
    elseif s_cm(2) > 2 % multiclass classification
        recall = zeros([length(confusion_matrix) 1]);
        precision = zeros([length(confusion_matrix) 1]);
        f1_measure = zeros([length(confusion_matrix) 1]);
        
        for i = 1:s_cm(2)
            for j = 1:s_cm(1)
                recall(i) = confusion_matrix(j, i) / sum(confusion_matrix(j, :)); % Recall
                recall(isnan(recall)) = 0;
                precision(i) = confusion_matrix(j, i) / sum(confusion_matrix(:, i)); % Precision
                precision(isnan(precision)) = 0;
                f1_measure(i) = 2 / ((1 / recall(i) + (1 / precision(i)))); % F1-measure
                %f1_measure(isnan(f1_measure)) = 0;
            end
        end
        
        recall = mean(recall);
        precision = mean(precision);
        f1_measure = mean(f1_measure);
    end
end

