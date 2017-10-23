function rates = confusion_rates(confusion_matrix)
%CONFUSION_RATES Compute the required rates based on confusion matrix
%                   Classes (column)
%   (Row)
%   Accuracy
%   Recall
%   Precision
%   F1-measure

rate_row_cnt = 4;
s_cm = size(confusion_matrix);
rates = zeros([rate_row_cnt s_cm(2)]);
total = sum(sum(confusion_matrix));

if s_cm(2) == 2 % binary classification
    rates = [trace(confusion_matrix) / total]; % Accuracy
    rates = [rates; confusion_matrix(2, 2) / sum(confusion_matrix(2, :))]; % Recall
    rates = [rates; confusion_matrix(2, 2) / sum(confusion_matrix(:, 2))]; % Precision
    rates = [rates; 2 / ((1 / rates(2, :) + (1 / rates(3, :))))]; % F1-measure
elseif s_cm(2) > 2
    for i = 1:s_cm(2)
        for j = 1:s_cm(1)
            rates = [trace(confusion_matrix) / total]; % Accuracy
            rates = [rates; confusion_matrix(j, i) / sum(confusion_matrix(j, :))]; % Recall
            rates = [rates; confusion_matrix(j, i) / sum(confusion_matrix(:, i))]; % Precision
            rates = [rates; 2 / ((1 / rates(2, i) + (1 / rates(3, i))))]; % F1-measure
        end
    end
else
    disp('Confusion matrix must have at least 2 columns');
end

end

