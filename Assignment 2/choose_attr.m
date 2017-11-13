function ent = choose_attr(examples,targets)
    total_features = size(examples,2);
    
    for feat = 1:total_features
        feature_list = examples(:,feat);
        [gain(feat),best_index(feat)] = split_data(examples,targets,feat,feature_list);
    end
    [~,index] = max(gain);
    
    
end