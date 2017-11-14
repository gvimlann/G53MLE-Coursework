function outputs = evaluate_tree(tree,samples)
    for sampleIndex = 1 : size(samples,1)
        outputs(sampleIndex) = evaluate_tree_single_sample(tree,samples(sampleIndex,:));
        outputs = outputs';
    end
end