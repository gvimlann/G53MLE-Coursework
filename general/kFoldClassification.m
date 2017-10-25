% Author: VIMLAN G
% parameters(data,kFold)
% returns two matrices with dimension - training matrix[kFold,150-batchSize,132] and
% testing matrix [kFold,batchSize,132]
function [training_batches,validation_batches] = kFoldClassification(data,kFold)
    batchSize = floor(size(data,1)/kFold);
    for c = 0:kFold-1
        tempData = data;
        validation(c+1,:,:) = (tempData((c*batchSize)+1:((c+1)*batchSize),:));
        tempData((c*batchSize)+1:((c+1)*batchSize),:) = [];
        training(c+1,:,:) = tempData;
    end
    training_batches = training;
    validation_batches =  validation;
end