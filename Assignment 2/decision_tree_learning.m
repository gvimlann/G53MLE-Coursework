%Author Yipin Jin

function  tree = decision_tree_learning(features, labels)
%-----
if all(~diff(labels))
    tree.op = [];
    tree.kids = [];
    tree.class = majority_value(labels);
    return;
%-----
else
%     disp(size(features));
    [best_feature, best_threshold] = choose_attribute(features, labels);
    examples1 = [];
    examples2 = [];
    tree.op = best_feature;
    tree.kids = [];
    tree.class = [];
    
    for sampleIndex = 1:size(features,1)
        if features(sampleIndex,best_feature) < best_threshold
            examples1 = [examples1 , sampleIndex];
        else
            examples2 = [examples2 , sampleIndex];
        end
    end
    targets1 = labels(examples1);
    targets2 = labels(examples2);
    examples1 = features(examples1,:);
    examples2 = features(examples2,:);
    
    if(isempty(examples1))
        tree.kids(0) = majority_value(targets1);
    else
        tree.kids(0) = decision_tree_learning(examples1,targets1);
    end
    
    if(isempty(examples2))
        tree.kids(1) = majority_value(targets2);
    else
        tree.kids(1) = decision_tree_learning(examples2,targets2);
    end
    
    return
end