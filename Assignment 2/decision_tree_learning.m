%Author Yipin Jin

function  tree = decision_tree_learning(features, labels)
%-----
if all(~diff(labels))
    tree.op = [];
    tree.kids = [];
    tree.class = majority_value(labels);
%-----
else
    [best_feature, best_threshold] = choose_attribute(features, labels);
    examples1 = [];
    examples2 = [];
    if all(labels == labels(1))
        tree.op = [];
        tree.kids = [];
        tree.class = majority_value(labels);

    else
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
        disp(size(examples1));
        tree.kids = [decision_tree_learning(examples1, targets1), decision_tree_learning(examples2, targets2)];
        tree.op = best_feature;
        tree.class = [];
    end
end