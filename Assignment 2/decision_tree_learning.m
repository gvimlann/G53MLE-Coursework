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
    if isempty(labels)
        tree.op = [];
        tree.kids = [];
        tree.class = majority_value(labels);
    else
        af = labels < best_threshold;
        labelA = labels(af);
        labelB = labels(~af);
        for tempFeature = 1:length(features(:, 1))
            features((features(tempFeature, :) == best_feature), : ) = [];
        end
        featureA = features(af);
        featureB = features(~af);
        tree.kids = [decision_tree_learning(featureA, labelA), decision_tree_learning(featureB, labelB)];
        tree.op = best_feature;
        tree.class = [];
    end
end