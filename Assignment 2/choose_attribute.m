% Author Vimlan Ganesan
function [bestThresholdFeature,bestThreshold] = choose_attribute(examples,labels)
    
    numberOfLabels = length(labels);
    numberOfFeatures = size(examples,2);
    
    bestThresholdIndex = zeros(numberOfFeatures,1);
    bestThresholdGain = zeros(numberOfFeatures,1);
    
    for i=1:numberOfFeatures
            ent = inf;
            %iterate through the examples of each features
        for j=1:numberOfLabels
            numOfPositiveRight = 0;
            numOfPositiveLeft = 0;
            numOfNegativeRight = 0;
            numOfNegativeLeft = 0; 
            center = examples(j,i);
            %sets the center of the feature split, iterates each sample for
            %new center for each feature
            for k=1:numberOfLabels
                %if sample value of feature is smaller than center, assign
                %to left side of the node else right
                if(examples(k,i) < center)
                    if(labels(k,1) == 1)
                        numOfPositiveLeft = numOfPositiveLeft + 1;
                    else
                        numOfNegativeLeft = numOfNegativeLeft + 1;
                    end
                else
                    if(labels(k,1) == 1)
                        numOfPositiveRight = numOfPositiveRight + 1;
                    else
                        numOfNegativeRight = numOfNegativeRight + 1;
                    end
                end
            end
            %Remainder function
            tempEnt = (numOfPositiveLeft+numOfNegativeLeft)/numberOfLabels * calculateEntropy(numOfPositiveLeft,numOfNegativeLeft);
            tempEnt = tempEnt + (numOfPositiveRight+numOfNegativeRight)/numberOfLabels*calculateEntropy(numOfPositiveRight,numOfNegativeRight);
            %Stores the lowest entropy to get the highest gain
            if(tempEnt < ent)
                ent = tempEnt;
                bestThresholdIndex(i) = j;
                bestThresholdGain(i) = tempEnt;
            end
        end
    end
    %returns the feature with the lowest ent
    [~,bestThresholdFeature] = min(bestThresholdGain);
    %returns the index of the feature examples with lowest ent
    bestThresholdFeatureIndex = bestThresholdIndex(bestThresholdFeature);
    bestThreshold = examples(bestThresholdFeatureIndex,bestThresholdFeature);
end