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
        A = labels(af);
        B = labels(~af);
        tree.kids = [decision_tree_learning(best_feature, A), decision_tree_learning(best_feature, B)];
        tree.op = best_feature;
        tree.class = [];
    end
end