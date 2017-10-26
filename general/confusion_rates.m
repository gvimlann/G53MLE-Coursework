function [accuracy recall precision f1_measure] = confusion_rates(confusion_matrix)
%CONFUSION_RATES Compute the required rates based on confusion matrix

rate_row_cnt = 4;
s_cm = size(confusion_matrix);
total = sum(sum(confusion_matrix));

if prod(s_cm) < 4
    disp('Confusion matrix must have at least 2 columns and rows');
    return;
end

if s_cm(2) == 2 % binary classification
    accuracy = trace(confusion_matrix) / total; % Accuracy
    recall = confusion_matrix(2, 2) / sum(confusion_matrix(2, :)); % Recall
    precision = confusion_matrix(2, 2) / sum(confusion_matrix(:, 2)); % Precision
    f1_measure = 2 / ((1 / accuracy + (1 / precision))); % F1-measure
elseif s_cm(2) > 2
    for i = 1:s_cm(2)
        for j = 1:s_cm(1)
            accuracy = trace(confusion_matrix) / total; % Accuracy
            recall = confusion_matrix(j, i) / sum(confusion_matrix(j, :)); % Recall
            precision = confusion_matrix(j, i) / sum(confusion_matrix(:, i)); % Precision
            f1_measure = 2 / ((1 / recall + (1 / precision))); % F1-measure
        end
    end
end

end

