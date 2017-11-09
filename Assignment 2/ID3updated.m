% Author Vimlan G
function [bestThresholdFeature,bestThresholdFeatureIndex] = ID3updated(examples,labels)
    tree = struct('value','null','kids',[],'class','null');
    numberOfLabels = length(labels);
    numberOfFeatures = size(examples,2);
    
   
    
    sumOfLabels = sum(labels(:,1));
    if(sumOfLabels == numberOfLabels || sumOfLabels == 0)
         tree.class = sumOfLabels * 1;
    end
    
    
%     % probability positive points
%     pp_p = sumOfLabels/numberOfLabels;
%     % probability negative points
%     pn_p = (numberOfLabels - sumOfLabels)/numberOfLabels;
%     
%     pp_eq = -1 * pp_p * log2(pp_p);
%     pn_eq = -1 * pn_p * log2(pn_p);
%     
%     %Entropy
%     entropy = pp_eq + pn_eq;
    
    [bestThresholdFeature,bestThresholdFeatureIndex] = choose_attribute(examples,labels);
end