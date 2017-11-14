function [feat,threshold] = choose_attribute(features,targets)
    for feature = 1:size(features,2)
        for sample = 1:size(features,1)
            center = features(sample,feature);
            left = find((features(:,feature) < center) == 1);
            right = find((features(:,feature) >= center) == 1);
            remainder_out(sample) = remainder(targets(left),targets(right));
        end
        [~,low_remainder_index] = min(remainder_out);
        low_index(feature) = low_remainder_index;
        gain_out(feature) = gain(targets,remainder_out(low_remainder_index));
    end
    [~,feat] = max(gain_out);
    low = low_index(feat);
    threshold = features(low,feat);
end