% Author Vimlan G
function entropy = calculateEntropy(positive,negative)
    pp_p = positive/(positive+negative);
    pn_p = negative/(positive+negative);
    
    pp_eq = -1 * pp_p * log2(pp_p);
    pn_eq = -1 * pn_p * log2(pn_p);
    
    %Entropy
    entropy = pp_eq + pn_eq;
end