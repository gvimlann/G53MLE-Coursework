function tree = decision_tree_learning(features,labels)
    if all(~diff(features))
    	tree.class = majority_value(labels);
    	return;
  	else
    	[best_feature,best_threshold] = choose_attribute(features,labels);
        disp(best_feature);
        tree.op = best_feature;
        tree.kids = {};
        for i = 1:2
            if(i == 1)
                left = find((features(:,best_feature) < best_threshold) == 1);
                examples1 = features(left);
                targets1 = labels(left);
                if isempty(examples1)
                    tree.kids.left = majority_value(targets1);
                    return 
                else
                    tree.kids.left = [decision_tree_learning(examples1,targets1)];
                end
            else
                right = find((features(:,best_feature) >= best_threshold) == 1);
                examples2 = features(right);
                targets2 = labels(right);
                if isempty(examples2)
                    tree.kids.right = majority_value(targets2);
                    return 
                else
                    tree.kids.right = [decision_tree_learning(examples2,targets2)];
                end
            end
        end
       
    end
end