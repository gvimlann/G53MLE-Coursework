function [gain,best_index] = feature_gain(data,labels,feature_index,feature_list)
    remainder = inf;
    best_index = 0;
    for feat = 1: size(feature_list)
        center = feature_list(feat);
        left_side = [];
        right_side = [];
        %split them to left and right node based on the feat value
        for temp_feat = 1 : size(feature_list)
            if(data(temp_feat,feature_index) < center)
                left_side = [left_side,labels(temp_feat)];
            else
                right_side = [right_side,labels(temp_feat)];
            end 
        end
        %Remainder
        %disp();
        temp_remainder = (size(left_side,1)/size(data,1))*calculateEntropy(left_side) + (size(right_side,1)/size(data,1))*calculateEntropy(right_side);
        %Finds the feature index with the lowest remainder - to use the
        %value as threshold
        if temp_remainder < remainder
            remainder = temp_remainder;
            best_index = feat;
        end
    end
    gain = calculateEntropy(labels) - remainder;
end