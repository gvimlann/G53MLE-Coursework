function output = evaluate_tree_single_sample(tree,sample)
    if(~isempty(tree.class))
        output = tree.class;
        return;
    end
     
    if(sample(tree.op) < tree.threshold)
        output = evaluate_tree_single_sample(tree.kids{1},sample);
    else
        output = evaluate_tree_single_sample(tree.kids{2},sample);
    end
end