% Author Vimlan G
function entropy = calculateEntropy(labels)
    
    positive = sum(labels(:) == 1);
    negative = sum(labels(:) == 0);
    
    pp_p = positive/size(labels,1);
    pn_p = negative/size(labels,1);
    
    pp_eq = -1 * pp_p * log2(pp_p);
    pn_eq = -1 * pn_p * log2(pn_p);
    %Entropy
    entropy = pp_eq + pn_eq;
end