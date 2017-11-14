function tree = decision_tree_learning(features,labels)
    if all(~diff(labels))
    	tree.class = majority_value(labels);
        tree.kids = [];
        tree.op = [];
    	return;
  	else
    	[best_feature,best_threshold] = choose_attribute(features,labels);
        tree = struct('op',[],'class',[]);
        tree.kids = {};
        tree.op = best_feature;
        for i = 1:2
            if(i == 1)
                left = find((features(:,best_feature) < best_threshold) == 1);
                examples1 = features(left,:);
                targets1 = labels(left);
                if isempty(examples1)
                    tree.kids(1) = {majority_value(targets1)};
                    return 
                else
                    tree.kids(1) = {decision_tree_learning(examples1,targets1)};
                end
            elseif i == 2
                right = find((features(:,best_feature) >= best_threshold) == 1);
                examples2 = features(right,:);
                targets2 = labels(right);
                if isempty(examples2)
                    tree.kids(2) = {majority_value(targets2)};
                    return 
                else
                    tree.kids(2) = {decision_tree_learning(examples2,targets2)};
                end
            end
        end
       
    end
end