% DEPRECATED: Succeded by built in function 'confusion'
% @author Boey
% Generates binary and multiclass confusion matrix
% Structure of matrix is identical to plotconfusion
function confusion_mat = gen_confusionmat(test_label, output_label)
    yc = zeros([length(output_label) 1]);
    if size(output_label, 1) == 1 % binary classification
        for i = 1:length(yc)
            yc(i) = round(output_label(i));
        end
        confusion_mat = confusionmat(test_label, yc);
    else % multiclass classification
        yt = zeros([length(output_label) 1]);
        for i = 1:length(yc)
            [val yc(i)] = max(output_label(:, i));
            [val yt(i)] = max(test_label(:, i));
        end
        confusion_mat = confusionmat(yt, yc);
    end
end

