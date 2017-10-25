% Author: VIMLAN G
% parameters(index, training_batches, validation_batches) - returns
% training batch and validation batch based on index
function [training_batch,validation_batch] = splitBatchesintoSingleBatch(index,training_batches,validation_batches)
    training_batch = training_batches(index,:,:);
    training_batch = reshape(training_batch,[size(training_batch,2),size(training_batch,3)]);
    validation_batch = validation_batches(index,:,:);
    validation_batch = reshape(validation_batch,[size(validation_batch,2),size(validation_batch,3)]);
end