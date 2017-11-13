function [best_feature,best_threshold] = choose_attr(examples,targets)
    total_features = size(examples,2);
    
    for feat = 1:total_features
        feature_list = examples(:,feat);
        [gain(feat),best_index(feat)] = feature_gain(examples,targets,feat,feature_list);
    end
    [~,index] = max(gain);
    best_feature = index;
    best_threshold = examples(best_index(index),best_feature);
    
end