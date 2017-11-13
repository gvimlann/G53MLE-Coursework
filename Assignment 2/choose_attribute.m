% Author Vimlan Ganesan
function [indexLowestFeatureEnt,bestThreshold] = choose_attribute(examples,samples)
    
    if(isempty(examples))
        return
    end
    totalSamples = length(samples);
    totalFeatures = size(examples,2);
    sampleIndex = zeros(totalFeatures,1);
    featureGain = zeros(totalFeatures,1);
    base_entropy = calculateEntropy(samples);
    
    for feature=1:totalFeatures
            base_remainder = inf;
            %iterate through the examples of each features
        for sample=1:totalSamples
            left = [];
            right = [];
            center = examples(sample,feature);
            %sets the center of the feature split, iterates each sample for
            %new center for each feature
            for tempSample=1:totalSamples
                %if sample value of feature is smaller than center, assign
                %to left side of the node else right
                if(examples(tempSample,feature) < center)
                    left = [left,samples(tempSample,1)];
                else
                    right = [right,samples(tempSample,1)];
                end
            end
            %Remainder function
            remainder = size(left,1)/totalSamples * calculateEntropy(left);
            remainder = remainder + size(right,1)/totalSamples*calculateEntropy(right);
            %Stores the lowest entropy to get the highest gain
            %Compares the entropy for each feature
            if(remainder < base_remainder)
                base_remainder = remainder;
                sampleIndex(feature) = sample;
                featureGain(feature) = base_entropy - remainder;
            end
        end
    end
    %returns the feature index with the lowest ent
    [~,indexLowestFeatureEnt] = max(featureGain);
    %returns the index of the feature examples in the SAMPLE with lowest
    %ent, finding the best threshold for the specific feature
    bestThresholdSampleIndex = sampleIndex(indexLowestFeatureEnt);
    bestThreshold = examples(bestThresholdSampleIndex,indexLowestFeatureEnt);
end