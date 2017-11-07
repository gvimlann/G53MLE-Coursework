% Author Vimlan G
function [bestThresholdFeature,bestThresholdFeatureIndex] = choose_attribute(examples,labels)
    
    numberOfLabels = length(labels);
    numberOfFeatures = size(examples,2);
    
    bestThresholdIndex = zeros(numberOfFeatures,1);
    bestThresholdGain = zeros(numberOfFeatures,1);
    
    for i=1:numberOfFeatures
            ent = inf;
        for j=1:numberOfLabels
            numOfPositiveRight = 0;
            numOfPositiveLeft = 0;
            numOfNegativeRight = 0;
            numOfNegativeLeft = 0;
            center = examples(j,i);
            for k=1:numberOfLabels
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
            tempEnt = (numOfPositiveLeft+numOfNegativeLeft)/numberOfLabels * calculateEntropy(numOfPositiveLeft,numOfNegativeLeft);
            tempEnt = tempEnt + (numOfPositiveRight+numOfNegativeRight)/numberOfLabels*calculateEntropy(numOfPositiveRight,numOfNegativeRight);
            if(tempEnt < ent)
                ent = tempEnt;
                bestThresholdIndex(i) = j;
                bestThresholdGain(i) = tempEnt;
            end
        end
    end
    [~,bestThresholdFeature] = min(bestThresholdGain);
    bestThresholdFeatureIndex = bestThresholdIndex(bestThresholdFeature);
end