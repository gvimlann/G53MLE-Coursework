% Author: VIMLAN G
% parameters(data:dimension(totalExamples,examplesFeatures),kFold)
% returns two matrices with dimension - training matrix[kFold,150-batchSize,132] and
% testing matrix [kFold,batchSize,132]
function [training_batches,training_labels,validation_batches,validation_labels] = kFoldClassification(points,labels,kFold)
    batchSize = floor(size(points,1)/kFold);
    for c = 0:kFold-1
        tempData = points;
        tempLabel = labels;
        validation(c+1,:,:) = (tempData(:,(c*batchSize)+1:((c+1)*batchSize)));
        validation_l(c+1,:,:) = (tempLabel(:,(c*batchSize)+1:((c+1)*batchSize)));
        tempData(:,(c*batchSize)+1:((c+1)*batchSize)) = [];
        tempLabel(:,(c*batchSize)+1:((c+1)*batchSize)) = [];
        training(c+1,:,:) = tempData;
        training_l(c+1,:,:) = tempLabel;
    end
    training_batches = training;
    validation_batches =  validation;
    training_labels = training_l;
    validation_labels = validation_l;
end