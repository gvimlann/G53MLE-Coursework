% Author Vimlan Ganesan
function [indexLowestFeatureEnt,bestThreshold] = choose_attribute(examples,samples)
    
    totalSamples = length(samples);
    totalFeatures = size(examples,2);
    
    sampleIndex = zeros(totalFeatures,1);
    featureEntropy = zeros(totalFeatures,1);
    
    for feature=1:totalFeatures
            ent = inf;
            %iterate through the examples of each features
        for sample=1:totalSamples
            positiveRight = 0;
            positiveLeft = 0;
            negativeRight = 0;
            negativeLeft = 0; 
            center = examples(sample,feature);
            %sets the center of the feature split, iterates each sample for
            %new center for each feature
            for tempsample=1:totalSamples
                %if sample value of feature is smaller than center, assign
                %to left side of the node else right
                if(examples(tempsample,feature) < center)
                    if(samples(tempsample,1) == 1)
                        positiveLeft = positiveLeft + 1;
                    else
                        negativeLeft = negativeLeft + 1;
                    end
                else
                    if(samples(tempsample,1) == 1)
                        positiveRight = positiveRight + 1;
                    else
                        negativeRight = negativeRight + 1;
                    end
                end
            end
            %Remainder function
            tempEnt = (positiveLeft+negativeLeft)/totalSamples * calculateEntropy(positiveLeft,negativeLeft);
            tempEnt = tempEnt + (positiveRight+negativeRight)/totalSamples*calculateEntropy(positiveRight,negativeRight);
            %Stores the lowest entropy to get the highest gain
            if(tempEnt < ent)
                ent = tempEnt;
                sampleIndex(feature) = sample;
                featureEntropy(feature) = tempEnt;
            end
        end
    end
    %returns the feature index with the lowest ent
    [~,indexLowestFeatureEnt] = min(featureEntropy);
    %returns the index of the feature examples in the SAMPLE with lowest
    %ent, finding the best threshold for the specific feature
    bestThresholdSampleIndex = sampleIndex(indexLowestFeatureEnt);
    bestThreshold = examples(bestThresholdSampleIndex,indexLowestFeatureEnt);
end